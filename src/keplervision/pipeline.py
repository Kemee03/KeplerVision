"""Light-curve classification pipeline and artifact handling."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import FLUX_COLUMNS

ARTIFACT_VERSION = "lightcurve-v1"


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build a class-weighted baseline for the strongly imbalanced source labels."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_multiscale_cnn(random_state: int = 42):
    """Build the multi-kernel 1D CNN from the research notebook."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "The CNN needs TensorFlow. Use Python 3.10–3.13 and install the 'cnn' extra."
        ) from exc
    tf.keras.utils.set_random_seed(random_state)
    inputs = tf.keras.Input(shape=(len(FLUX_COLUMNS), 1), name="light_curve")
    branches = [
        tf.keras.layers.Conv1D(32, kernel_size=size, padding="same", activation="relu")(
            inputs
        )
        for size in (3, 5, 11, 21)
    ]
    merged = tf.keras.layers.Concatenate(name="multi_scale_features")(branches)
    pooled = tf.keras.layers.MaxPooling1D(2)(merged)
    pooled = tf.keras.layers.MaxPooling1D(2)(pooled)
    flattened = tf.keras.layers.Flatten()(pooled)
    hidden = tf.keras.layers.Dense(64, activation="relu")(flattened)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="keplervision_multiscale_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class KerasLightCurvePredictor:
    """Adapt a saved Keras model and its scaler to the sklearn prediction interface."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_proba(self, features) -> np.ndarray:
        values = self.scaler.transform(features).astype(np.float32, copy=False)
        planet_probability = self.model.predict(values[..., np.newaxis], verbose=0).reshape(-1)
        return np.column_stack((1 - planet_probability, planet_probability))

    def predict(self, features) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def save_model(model, path: str | Path, *, scaler=None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".keras":
        if scaler is None:
            raise ValueError("A fitted scaler is required to save a CNN model")
        model.save(destination)
        joblib.dump(scaler, destination.with_suffix(".scaler.joblib"))
        return destination
    joblib.dump(
        {"model": model, "features": list(FLUX_COLUMNS), "format": ARTIFACT_VERSION},
        destination,
    )
    return destination


def load_model(path: str | Path):
    source = Path(path)
    if source.suffix == ".keras":
        scaler_path = source.with_suffix(".scaler.joblib")
        if not scaler_path.is_file():
            raise ValueError(f"CNN scaler artifact not found: {scaler_path}")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Serving this CNN requires TensorFlow") from exc
        return KerasLightCurvePredictor(tf.keras.models.load_model(source), joblib.load(scaler_path))
    artifact = joblib.load(source)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Model artifact has an unsupported format")
    if artifact.get("format") != ARTIFACT_VERSION or artifact.get("features") != list(FLUX_COLUMNS):
        raise ValueError("Model artifact does not match the KeplerVision light-curve schema")
    return artifact["model"]
