from __future__ import annotations

from datetime import datetime
from sqlalchemy import UUID, Text, DateTime, Enum, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import mapped_column, Mapped

import uuid
from backend.app.src.db.models.base import Base
from backend.app.src.db.experiment_status import ExperimentStatus


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    standardized_input_smiles: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(
            ExperimentStatus,
            name="experiment_status_enum",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        default=ExperimentStatus.PENDING,
        nullable=False,
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    results: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
