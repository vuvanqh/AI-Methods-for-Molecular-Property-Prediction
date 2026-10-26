from __future__ import annotations

from sqlalchemy import UUID, Text, ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship

from typing import TYPE_CHECKING
from backend.app.src.db.models.base import Base
import uuid

if TYPE_CHECKING:
    from .molecule import Molecule


class MoleculeSource(Base):
    __tablename__ = "molecule_sources"

    provider: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        index=True,
    )

    molecule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("molecules.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )

    source_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    molecule: Mapped[Molecule] = relationship(
        back_populates="molecule_sources",
    )
