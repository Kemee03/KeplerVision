"""Evaluate a saved KeplerVision model on a labeled CSV without retraining it."""

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

from .pipeline import load_model
from .schema import FLUX_COLUMNS, LABEL, binary_labels, validate_observations
from .train import wilson_interval
from .tracking import log_run, require_mlflow

THRESHOLD = 0.5


def evaluate(model_path: str | Path, csv_path: str | Path) -> dict:
    """Score an existing artifact against labeled data; this never fits a model."""
    frame = validate_observations(pd.read_csv(csv_path), require_label=True)
    features = frame.loc[:, list(FLUX_COLUMNS)]
    labels = binary_labels(frame[LABEL]).to_numpy(dtype=int)
    model = load_model(model_path)
    probabilities = np.asarray(model.predict_proba(features))[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    positive_count = int(np.sum(labels == 1))
    true_positives = int(np.sum((labels == 1) & (predictions == 1)))
    warnings = []
    if positive_count < 30:
        warnings.append(
            f"Only {positive_count} planet examples are in this evaluation file; "
            "planet recall has high uncertainty and should not be treated as a reliable estimate."
        )

    return {
        "model_artifact": str(model_path),
        "evaluation_file": str(csv_path),
        "evaluation_rows": int(len(labels)),
        "positive_evaluation_rows": positive_count,
        "prediction_threshold": THRESHOLD,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(labels, predictions)),
        "planet_recall_wilson_95_ci": wilson_interval(true_positives, positive_count),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["no_planet", "planet"],
            output_dict=True,
            zero_division=0,
        ),
        "evaluation_warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/multiscale_cnn.keras", help="Saved model artifact")
    parser.add_argument("--csv", default="data/exoTest.csv", help="Labeled evaluation CSV")
    parser.add_argument("--report", help="Optional path to save the JSON report")
    parser.add_argument("--track-mlflow", action="store_true", help="Log this evaluation to local MLflow")
    parser.add_argument("--experiment", default="KeplerVision", help="MLflow experiment name")
    args = parser.parse_args()
    if args.track_mlflow:
        try:
            require_mlflow()
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        result = evaluate(args.model, args.csv)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.track_mlflow:
        model_path = Path(args.model)
        artifact_paths = [model_path]
        scaler_path = model_path.with_suffix(".scaler.joblib")
        if scaler_path.is_file():
            artifact_paths.append(scaler_path)
        if args.report:
            artifact_paths.append(Path(args.report))
        result["mlflow_run_id"] = log_run(
            experiment=args.experiment,
            run_name="saved-model-evaluation",
            metrics=result,
            parameters={
                "model_artifact": args.model,
                "evaluation_csv": args.csv,
                "decision_threshold": THRESHOLD,
            },
            artifact_paths=artifact_paths,
            source_files=[Path(args.csv)],
        )
        if args.report:
            report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = json.dumps(result, indent=2)
    print(report)


if __name__ == "__main__":
    main()
