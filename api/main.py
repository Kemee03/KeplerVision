"""FastAPI inference service for Kepler time-series light curves."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from api.database import engine, initialize_database, recent_predictions, save_predictions
from keplervision.pipeline import load_model
from keplervision.schema import FLUX_COLUMNS, TARGET_NAMES, validate_observations


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="KeplerVision API", version="0.1.0", lifespan=lifespan)


class LightCurve(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str | None = Field(default=None, max_length=255)
    flux: list[float] = Field(
        min_length=3197,
        max_length=3197,
        description="Exactly 3,197 flux measurements ordered FLUX.1 through FLUX.3197",
    )


class PredictionRequest(BaseModel):
    observations: list[LightCurve] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _model(path: str):
    return load_model(path)


@app.get("/health")
def health() -> dict:
    model_path = Path(os.getenv("KEPLERVISION_MODEL_PATH", "models/multiscale_cnn.keras"))
    return {
        "status": "ok",
        "model_loaded": model_path.is_file(),
        "database_configured": engine is not None,
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "features": {"count": len(FLUX_COLUMNS), "format": "FLUX.1 through FLUX.3197"},
        "source_labels": {"1": "no_planet", "2": "planet"},
    }


@app.get("/predictions")
def get_recent_predictions(limit: int = Query(default=20, ge=1, le=500)) -> dict:
    try:
        records = recent_predictions(limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable.") from exc
    return {
        "predictions": [
            {
                "observation_id": record.observation_id,
                "object_id": record.object_id,
                "prediction": record.prediction,
                "planet_probability": record.planet_probability,
                "confidence": record.confidence,
                "model_version": record.model_version,
                "created_at": record.created_at,
            }
            for record in records
        ]
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    model_path = os.getenv("KEPLERVISION_MODEL_PATH", "models/multiscale_cnn.keras")
    if not Path(model_path).is_file():
        raise HTTPException(status_code=503, detail="No trained model artifact is available")
    frame = pd.DataFrame(
        [row.flux for row in request.observations], columns=FLUX_COLUMNS
    )
    try:
        frame = validate_observations(frame)
        model = _model(model_path)
        probabilities = model.predict_proba(frame.loc[:, list(FLUX_COLUMNS)])[:, 1]
        predictions = model.predict(frame.loc[:, list(FLUX_COLUMNS)])
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_version = os.getenv("KEPLERVISION_MODEL_VERSION", Path(model_path).stem)
    records = [
        {
            "observation_id": str(uuid4()),
            "object_id": observation.object_id,
            "features": observation.flux,
            "prediction": TARGET_NAMES[int(label)],
            "planet_probability": float(probability),
            "confidence": float(probability if label else 1 - probability),
            "model_version": model_version,
        }
        for observation, label, probability in zip(
            request.observations, predictions, probabilities, strict=True
        )
    ]
    try:
        save_predictions(records)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Predictions could not be saved. Check that PostgreSQL is running.",
        ) from exc
    return {
        "predictions": [
            {
                "observation_id": record["observation_id"],
                "label": int(label),
                "class_name": TARGET_NAMES[int(label)],
                "planet_probability": record["planet_probability"],
                "confidence": record["confidence"],
                "model_version": model_version,
            }
            for record, label in zip(records, predictions, strict=True)
        ]
    }


WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
