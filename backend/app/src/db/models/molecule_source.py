from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import UUID, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

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

    source_identifier: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    molecule: Mapped[Molecule] = relationship(
        back_populates="molecule_sources",
    )
