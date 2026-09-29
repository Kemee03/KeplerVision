"""Optional MLflow logging helpers for local, reproducible runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_mlflow():
    """Import MLflow before starting expensive work so missing setup fails early."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is optional. Install it with: pip install -e '.[mlops]'"
        ) from exc
    return mlflow


def log_run(
    *,
    experiment: str,
    run_name: str,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    artifact_paths: list[str | Path],
    source_files: list[str | Path],
) -> str:
    """Record scalar metrics, parameters, artifact files, and source file hashes."""
    mlflow = require_mlflow()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        params = {key: str(value) for key, value in parameters.items() if value is not None}
        for path in source_files:
            source = Path(path)
            params[f"sha256_{source.name}"] = _sha256(source)
        mlflow.log_params(params)

        scalar_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for class_name in ("no_planet", "planet", "macro avg", "weighted avg"):
            report = metrics.get("classification_report", {}).get(class_name, {})
            for metric_name in ("precision", "recall", "f1-score"):
                if metric_name in report:
                    scalar_metrics[f"{class_name.replace(' ', '_')}_{metric_name.replace('-', '_')}"] = float(
                        report[metric_name]
                    )
        if "planet_recall_wilson_95_ci" in metrics:
            lower, upper = metrics["planet_recall_wilson_95_ci"]
            scalar_metrics["planet_recall_ci_lower"] = float(lower)
            scalar_metrics["planet_recall_ci_upper"] = float(upper)
        mlflow.log_metrics(scalar_metrics)

        for path in artifact_paths:
            artifact = Path(path)
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path=artifact.parent.name or "artifacts")
        return run.info.run_id
