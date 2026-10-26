from __future__ import annotations

from pgvector.sqlalchemy import BIT
from sqlalchemy import UUID, Text
from sqlalchemy.orm import mapped_column, Mapped, relationship
import uuid

from typing import TYPE_CHECKING
from backend.app.src.db.models.base import Base

if TYPE_CHECKING:
    from .molecule_embedding import MoleculeEmbedding
    from .molecule_source import MoleculeSource


class Molecule(Base):
    __tablename__ = "molecules"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    standardized_smiles: Mapped[str] = mapped_column(
        Text,
        unique=True,
        index=True,
        nullable=False,
    )

    morgan_fingerprint: Mapped[str] = mapped_column(
        BIT(1024),
        nullable=False,
    )

    molecule_embedding: Mapped[list["MoleculeEmbedding"]] = relationship(
        back_populates="molecule",
        cascade="all, delete-orphan",
    )

    molecule_sources: Mapped[list["MoleculeSource"]] = relationship(
        back_populates="molecule",
        cascade="all, delete-orphan",
    )
