"""Optional PostgreSQL persistence for predictions."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, Float, JSON, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    observation_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    object_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    features: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    prediction: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    planet_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None


def initialize_database() -> None:
    if engine is not None:
        Base.metadata.create_all(engine)


def save_predictions(records: list[dict]) -> None:
    if engine is None or not records:
        return
    with Session(engine) as session, session.begin():
        session.add_all(PredictionRecord(**record) for record in records)


def recent_predictions(limit: int) -> list[PredictionRecord]:
    if engine is None:
        raise RuntimeError(
            "PostgreSQL is not configured. Set the DATABASE_URL environment variable."
        )
    with Session(engine) as session:
        statement = (
            select(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
        )
        return list(session.scalars(statement))
