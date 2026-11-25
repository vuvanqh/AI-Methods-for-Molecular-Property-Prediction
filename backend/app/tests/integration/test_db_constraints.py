import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis, Molecule, MoleculeEmbedding, MoleculeSource
from src.rag.morgan_fingerprint_helper import (
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
)
from tests.pg_fixtures import seed_molecule

pytestmark = pytest.mark.integration


def _fp(smiles: str) -> str:
    return fingerprint_to_bitstring(generate_morgan_fingerprint(smiles))


def _assert_integrity_error(db_session: Session, row: object) -> None:
    db_session.add(row)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_molecules_standardized_smiles_unique_constraint_rejects_duplicate(
    db_session: Session,
) -> None:
    seed_molecule(db_session, "CCO")
    _assert_integrity_error(
        db_session, Molecule(standardized_smiles="CCO", morgan_fingerprint=_fp("CCO"))
    )


def test_molecule_sources_composite_pk_and_fk_constraints(
    db_session: Session,
) -> None:
    mol = seed_molecule(db_session, "CCO", sources=[("ChEMBL", "CHEMBL545")])

    # Composite PK (provider, molecule_id) rejects duplicate provider per molecule
    _assert_integrity_error(
        db_session,
        MoleculeSource(
            provider="ChEMBL", molecule_id=mol.id, source_identifier="CHEMBL999999"
        ),
    )
    # FK rejects nonexistent molecule_id
    _assert_integrity_error(
        db_session,
        MoleculeSource(
            provider="ChEMBL",
            molecule_id=uuid.uuid4(),
            source_identifier="CHEMBL_ORPHAN",
        ),
    )


def test_molecule_embeddings_unique_and_fk_constraints(
    db_session: Session,
) -> None:
    mol = seed_molecule(
        db_session, "CCO", embeddings=[("unimol", "unimolv1", [0.1, 0.2, 0.3, 0.4])]
    )

    # Unique (molecule_id, embedding_model, embedding_model_version)
    _assert_integrity_error(
        db_session,
        MoleculeEmbedding(
            molecule_id=mol.id,
            embedding_model="unimol",
            embedding_model_version="unimolv1",
            embedding=[0.9, 0.8, 0.7, 0.6],
        ),
    )
    # FK rejects nonexistent molecule_id
    _assert_integrity_error(
        db_session,
        MoleculeEmbedding(
            molecule_id=uuid.uuid4(),
            embedding_model="unimol",
            embedding_model_version="unimolv1",
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
    )


def test_sql_delete_molecule_cascades_only_its_own_embeddings_and_sources(
    db_session: Session,
) -> None:
    deleted_mol = seed_molecule(
        db_session,
        "CCO",
        sources=[("ChEMBL", "CHEMBL545"), ("PubChem", "702")],
        embeddings=[("unimol", "unimolv1", [0.1] * 8)],
    )
    kept_mol = seed_molecule(
        db_session,
        "c1ccccc1",
        sources=[("ChEMBL", "CHEMBL277500")],
        embeddings=[("unimol", "unimolv1", [0.2] * 8)],
    )

    db_session.execute(delete(Molecule).where(Molecule.id == deleted_mol.id))
    db_session.commit()

    assert [e.molecule_id for e in db_session.scalars(select(MoleculeEmbedding))] == [
        kept_mol.id
    ]
    assert [s.molecule_id for s in db_session.scalars(select(MoleculeSource))] == [
        kept_mol.id
    ]


def test_analyses_jsonb_none_as_null_enforces_config_and_persists_sql_null_results(
    db_session: Session,
) -> None:
    # JSONB(none_as_null=True): Python None binds as SQL NULL, not JSON 'null'::jsonb
    _assert_integrity_error(
        db_session,
        Analysis(
            standardized_input_smiles="CCO",
            status=AnalysisStatus.COMPLETED,
            config=None,  # type: ignore[arg-type]
            results=None,
        ),
    )

    row = Analysis(
        standardized_input_smiles="CCO",
        status=AnalysisStatus.FAILED,
        config={"input_smiles": "CCO"},
        results=None,
    )
    db_session.add(row)
    db_session.commit()

    is_sql_null = db_session.scalar(
        text("SELECT results IS NULL FROM analyses WHERE id = :id"), {"id": row.id}
    )
    assert is_sql_null is True


def test_analyses_enum_values_default_pending_and_server_timestamp(
    db_session: Session,
) -> None:
    before = datetime.now(UTC)
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
        assert isinstance(row.created_at, datetime)
        assert row.created_at.tzinfo is not None

    stored_labels = dict(
        db_session.execute(text("SELECT id, status::text FROM analyses")).all()
    )
    assert default_row.status == AnalysisStatus.PENDING
    assert stored_labels[default_row.id] == AnalysisStatus.PENDING.value
    assert [stored_labels[r.id] for r in status_rows] == [
        s.value for s in AnalysisStatus
    ]
    assert abs((default_row.created_at - before).total_seconds()) < 60


def test_morgan_fingerprint_orm_round_trip_preserves_exact_2048_bits(
    db_session: Session,
) -> None:
    expected_fp = _fp("O=C(O)c1ccccc1O")
    assert len(expected_fp) == 2048 and "1" in expected_fp

    mol = seed_molecule(db_session, "O=C(O)c1ccccc1O")
    db_session.expire_all()
    stored_fp = db_session.scalar(
        select(Molecule.morgan_fingerprint).where(Molecule.id == mol.id)
    )

    assert stored_fp == expected_fp
    assert len(stored_fp) == 2048
