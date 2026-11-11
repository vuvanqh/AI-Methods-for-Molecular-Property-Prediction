from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import UUID, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .molecule import Molecule


class MoleculeEmbedding(Base):
    __tablename__ = "molecule_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    molecule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("molecules.id", ondelete="CASCADE"),
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(Text, index=True, nullable=False)

    embedding_model_version: Mapped[str] = mapped_column(
        Text,
        index=True,
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector,
        nullable=False,
    )

    molecule: Mapped[Molecule] = relationship(
        back_populates="molecule_embedding",
    )

    __table_args__ = (
        UniqueConstraint(
            "molecule_id",
            "embedding_model",
            "embedding_model_version",
            name="unique_molecule_embedding",
        ),
    )
