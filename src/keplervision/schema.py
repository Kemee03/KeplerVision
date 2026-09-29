"""Validation for Kepler labelled light-curve rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

FLUX_COLUMNS = tuple(f"FLUX.{index}" for index in range(1, 3198))
LABEL = "LABEL"
TARGET_NAMES = {0: "no_planet", 1: "planet"}


def validate_observations(frame: pd.DataFrame, *, require_label: bool = False) -> pd.DataFrame:
    """Validate flux columns and, for training/evaluation, the dataset's 1/2 labels."""
    required = list(FLUX_COLUMNS) + ([LABEL] if require_label else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns; expected LABEL (for labeled data) "
            "and FLUX.1 through FLUX.3197."
        )
    if frame.empty:
        raise ValueError("Observation table must contain at least one row")
    result = frame.loc[:, required].copy()
    try:
        result.loc[:, list(FLUX_COLUMNS)] = result.loc[:, list(FLUX_COLUMNS)].apply(
            pd.to_numeric, errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Flux measurements must all be numeric") from exc
    fluxes = result.loc[:, list(FLUX_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(fluxes).all():
        raise ValueError("Flux measurements cannot contain missing or infinite values")
    if require_label:
        labels = pd.to_numeric(result[LABEL], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({1, 2}):
            raise ValueError("LABEL must contain only 1 (no planet) or 2 (planet)")
        if labels.nunique() != 2:
            raise ValueError("Labeled data must contain both LABEL classes 1 and 2")
        result[LABEL] = labels.astype(int)
    return result


def binary_labels(labels: pd.Series) -> pd.Series:
    """Convert the source dataset's labels (1/2) to model labels (0/1)."""
    return labels.astype(int).map({1: 0, 2: 1})
