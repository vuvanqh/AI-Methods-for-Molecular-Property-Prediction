from datetime import datetime

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis, Molecule, MoleculeEmbedding, MoleculeSource
from tests.integration.conftest import seed_molecule

pytestmark = pytest.mark.integration


def test_analyses_orm_enum_values_default_pending_and_server_timestamp(
    db_session: Session,
) -> None:
    # Default status (omitted) -> AnalysisStatus.PENDING; plus explicit check for every enum label
    default_row = Analysis(
        standardized_input_smiles="O=Cc1ccccc1",
        config={"input_smiles": "O=Cc1ccccc1"},
        results=None,
    )
    status_rows = [
        Analysis(
            standardized_input_smiles="CCO",
            status=status,
            config={"input_smiles": "CCO"},
            results={"ok": True} if status == AnalysisStatus.COMPLETED else None,
        )
        for status in AnalysisStatus
    ]
    db_session.add_all([default_row, *status_rows])
    db_session.commit()

    for row in (default_row, *status_rows):
        db_session.refresh(row)
        assert row.id is not None
        assert isinstance(row.created_at, datetime) and row.created_at.tzinfo

    assert default_row.status == AnalysisStatus.PENDING
    assert [row.status for row in status_rows] == list(AnalysisStatus)


def test_analyses_jsonb_none_as_null_enforces_config_not_null(
    db_session: Session,
) -> None:
    # Exercises JSONB(none_as_null=True): Python None must bind as SQL NULL, not JSON 'null'::jsonb
    db_session.add(
        Analysis(
            standardized_input_smiles="CCO",
            status=AnalysisStatus.COMPLETED,
            config=None,  # type: ignore[arg-type]
            results=None,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_deleting_molecule_cascades_to_embeddings_and_sources_in_postgres(
    db_session: Session,
) -> None:
    mol = seed_molecule(
        db_session,
        "CCO",
        sources=[("ChEMBL", "CHEMBL545"), ("PubChem", "702")],
        embeddings=[("unimol", "unimolv1", [0.1] * 8)],
    )

    # SQL-level DELETE exercises PostgreSQL ON DELETE CASCADE (bypassing ORM cascade)
    db_session.execute(delete(Molecule).where(Molecule.id == mol.id))
    db_session.commit()

    assert db_session.scalars(select(MoleculeEmbedding)).all() == []
    assert db_session.scalars(select(MoleculeSource)).all() == []
