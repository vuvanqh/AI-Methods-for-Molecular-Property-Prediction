import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url

from src.db.models import Base
from tests.integration.conftest import make_alembic_config

pytestmark = pytest.mark.integration

APP_TABLES = {"analyses", "molecules", "molecule_embeddings", "molecule_sources"}


@pytest.fixture
def empty_db_url(pg_url: str) -> Iterator[str]:
    """Create a uniquely named empty database on the test Postgres server for migration tests."""
    base = make_url(pg_url)
    admin = create_engine(
        base.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    db_name = f"migrations_verify_{uuid.uuid4().hex[:8]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {db_name}"))
        yield base.set(database=db_name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)"))
        admin.dispose()


def test_alembic_upgrade_downgrade_and_data_preserving_rename(
    empty_db_url: str,
) -> None:
    cfg = make_alembic_config(empty_db_url)
    engine = create_engine(empty_db_url, pool_pre_ping=True)
    try:
        # 1. Upgrade to initial revision and seed a row to verify 5b1f0c2e9a7d preserves data
        command.upgrade(cfg, "08d3e13077d6")
        mol_id = uuid.uuid4()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO molecules (id, standardized_smiles, morgan_fingerprint) "
                    "VALUES (:id, 'CCO', :fp)"
                ),
                {"id": mol_id, "fp": "0" * 2048},
            )
            conn.execute(
                text(
                    "INSERT INTO molecule_sources (provider, molecule_id, source_url) "
                    "VALUES ('ChEMBL', :id, 'CHEMBL545')"
                ),
                {"id": mol_id},
            )

        # 2. Upgrade across 5b1f0c2e9a7d: verify renamed column preserves data
        command.upgrade(cfg, "5b1f0c2e9a7d")
        inspector = inspect(engine)
        source_cols = {c["name"] for c in inspector.get_columns("molecule_sources")}
        assert "source_identifier" in source_cols and "source_url" not in source_cols
        with engine.connect() as conn:
            assert (
                conn.scalar(
                    text(
                        "SELECT source_identifier FROM molecule_sources WHERE molecule_id = :id"
                    ),
                    {"id": mol_id},
                )
                == "CHEMBL545"
            )

        # 3. Downgrade back to 08d3e13077d6 preserves data in source_url
        command.downgrade(cfg, "08d3e13077d6")
        with engine.connect() as conn:
            assert (
                conn.scalar(
                    text(
                        "SELECT source_url FROM molecule_sources WHERE molecule_id = :id"
                    ),
                    {"id": mol_id},
                )
                == "CHEMBL545"
            )

        # 4. Full head -> base -> head lifecycle: verify tables, pgvector extension, and enum cleanup
        command.upgrade(cfg, "head")
        assert APP_TABLES <= set(inspect(engine).get_table_names())
        with engine.connect() as conn:
            assert (
                conn.scalar(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )
                == "vector"
            )
            labels = conn.scalars(
                text(
                    "SELECT e.enumlabel FROM pg_enum e "
                    "JOIN pg_type t ON e.enumtypid = t.oid "
                    "WHERE t.typname = 'analysis_status_enum' ORDER BY e.enumsortorder"
                )
            ).all()
            assert list(labels) == ["Pending", "In Progress", "Completed", "Failed"]

        command.downgrade(cfg, "base")
        assert set(inspect(engine).get_table_names()) - {"alembic_version"} == set()
        with engine.connect() as conn:
            assert (
                conn.scalar(
                    text(
                        "SELECT count(*) FROM pg_type WHERE typname = 'analysis_status_enum'"
                    )
                )
                == 0
            )

        command.upgrade(cfg, "head")
        assert APP_TABLES <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_orm_models_match_alembic_migrated_schema(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_server_default": True})
        assert compare_metadata(ctx, Base.metadata) == []
