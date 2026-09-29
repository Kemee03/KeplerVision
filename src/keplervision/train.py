"""Train and evaluate a KeplerVision model using labeled flux light curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .pipeline import build_multiscale_cnn, build_pipeline, save_model
from .schema import FLUX_COLUMNS, LABEL, binary_labels, validate_observations
from .tracking import log_run, require_mlflow


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> list[float]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def train(
    csv_path: str | Path,
    output: str | Path,
    *,
    test_csv: str | Path | None = None,
    model_type: str = "multiscale_cnn",
    random_state: int = 42,
) -> dict:
    train_frame = validate_observations(pd.read_csv(csv_path), require_label=True)
    x_train = train_frame.loc[:, list(FLUX_COLUMNS)]
    y_train = binary_labels(train_frame[LABEL])
    if test_csv:
        test_frame = validate_observations(pd.read_csv(test_csv), require_label=True)
        x_test = test_frame.loc[:, list(FLUX_COLUMNS)]
        y_test = binary_labels(test_frame[LABEL])
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=y_train,
        )

    scaler = None
    if model_type == "multiscale_cnn":
        scaler = StandardScaler().fit(x_train)
        x_fit = scaler.transform(x_train).astype(np.float32)[..., np.newaxis]
        x_eval = scaler.transform(x_test).astype(np.float32)[..., np.newaxis]
        y_fit = y_train.to_numpy(dtype=np.float32)
        labels, counts = np.unique(y_fit, return_counts=True)
        if len(labels) != 2:
            raise ValueError("Training data must contain both label classes")
        rng = np.random.default_rng(random_state)
        majority_count = int(counts.max())
        sampled_indices = [np.flatnonzero(y_fit == label) for label in labels]
        balanced_indices = np.concatenate(
            [
                np.concatenate(
                    [indices, rng.choice(indices, majority_count - len(indices), replace=True)]
                )
                for indices in sampled_indices
            ]
        )
        rng.shuffle(balanced_indices)
        model = build_multiscale_cnn(random_state)
        model.fit(x_fit[balanced_indices], y_fit[balanced_indices], epochs=3, batch_size=32, verbose=2)
        probabilities = model.predict(x_eval, batch_size=32, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        artifact_path = save_model(model, output, scaler=scaler)
        model_name = "multi-scale 1D CNN (3 epochs; minority class oversampled in training only)"
    elif model_type == "logistic":
        model = build_pipeline(random_state)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        artifact_path = save_model(model, output)
        model_name = "class-weighted logistic regression"
    else:
        raise ValueError("model_type must be 'multiscale_cnn' or 'logistic'")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    true_planets = int(np.sum(np.asarray(y_test) == 1))
    found_planets = int(np.sum((np.asarray(y_test) == 1) & (predictions == 1)))
    evaluation_warnings = []
    if true_planets < 30:
        evaluation_warnings.append(
            f"Only {true_planets} planet examples are in this evaluation split; "
            "planet recall has high uncertainty and should not be treated as a reliable estimate."
        )
    metrics = {
        "model": model_name,
        "train_rows": int(len(x_train)),
        "evaluation_rows": int(len(x_test)),
        "positive_train_rows": int(y_train.sum()),
        "positive_evaluation_rows": true_planets,
        "prediction_threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_test, predictions)),
        "planet_recall_wilson_95_ci": wilson_interval(found_planets, true_planets),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["no_planet", "planet"],
            output_dict=True, zero_division=0,
        ),
        "evaluation_warnings": evaluation_warnings,
        "artifact": str(artifact_path),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Labeled exoTrain.csv")
    parser.add_argument("--test-csv", help="Optional held-out exoTest.csv")
    parser.add_argument("--model", choices=["multiscale_cnn", "logistic"], default="multiscale_cnn")
    parser.add_argument("--output", default="models/multiscale_cnn.keras")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--track-mlflow", action="store_true", help="Log this training run to local MLflow")
    parser.add_argument("--experiment", default="KeplerVision", help="MLflow experiment name")
    args = parser.parse_args()
    if args.track_mlflow:
        try:
            require_mlflow()
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        result = train(
            args.csv, args.output, test_csv=args.test_csv,
            model_type=args.model, random_state=args.random_state
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.track_mlflow:
        model_path = Path(args.output)
        artifact_paths = [model_path]
        scaler_path = model_path.with_suffix(".scaler.joblib")
        if scaler_path.is_file():
            artifact_paths.append(scaler_path)
        source_files = [Path(args.csv)]
        if args.test_csv:
            source_files.append(Path(args.test_csv))
        result["mlflow_run_id"] = log_run(
            experiment=args.experiment,
            run_name=f"{args.model}-training",
            metrics=result,
            parameters={
                "model_type": args.model,
                "random_state": args.random_state,
                "train_csv": args.csv,
                "test_csv": args.test_csv or "stratified split from train CSV",
                "decision_threshold": 0.5,
            },
            artifact_paths=artifact_paths,
            source_files=source_files,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
