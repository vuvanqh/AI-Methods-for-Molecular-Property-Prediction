import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url

from src.db.models import Base
from tests.pg_fixtures import make_alembic_config

pytestmark = pytest.mark.integration

INITIAL_REV = "08d3e13077d6"
RENAME_REV = "5b1f0c2e9a7d"
APP_TABLES = {"analyses", "molecules", "molecule_embeddings", "molecule_sources"}


@pytest.fixture
def empty_db(pg_url: str) -> Iterator[tuple[str, Engine]]:
    """Create a uniquely named empty database and engine on the test Postgres server."""
    base = make_url(pg_url)
    admin = create_engine(
        base.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    db_name = f"migrations_verify_{uuid.uuid4().hex[:8]}"
    db_url = base.set(database=db_name).render_as_string(hide_password=False)
    engine: Engine | None = None
    try:
        with admin.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {db_name}"))
        engine = create_engine(db_url, pool_pre_ping=True)
        yield db_url, engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)"))
        admin.dispose()


def test_upgrade_to_head_creates_tables_pgvector_extension_and_status_enum(
    empty_db: tuple[str, Engine],
) -> None:
    db_url, engine = empty_db
    command.upgrade(make_alembic_config(db_url), "head")

    tables = set(inspect(engine).get_table_names())
    assert APP_TABLES <= tables and "alembic_version" in tables
    with engine.connect() as conn:
        ext = conn.scalar(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        labels = conn.scalars(
            text(
                "SELECT e.enumlabel FROM pg_enum e "
                "JOIN pg_type t ON e.enumtypid = t.oid "
                "WHERE t.typname = 'analysis_status_enum' ORDER BY e.enumsortorder"
            )
        ).all()
    assert ext == "vector"
    assert list(labels) == ["Pending", "In Progress", "Completed", "Failed"]


def test_rename_migration_round_trip_preserves_source_rows(
    empty_db: tuple[str, Engine],
) -> None:
    db_url, engine = empty_db
    cfg = make_alembic_config(db_url)
    command.upgrade(cfg, INITIAL_REV)

    rows = [
        (uuid.uuid4(), "CCO", "CHEMBL545"),
        (uuid.uuid4(), "c1ccccc1", "CHEMBL277500"),
    ]
    expected = {(mid, "ChEMBL", ident) for mid, _, ident in rows}
    with engine.begin() as conn:
        for mid, smiles, ident in rows:
            conn.execute(
                text(
                    "INSERT INTO molecules (id, standardized_smiles, morgan_fingerprint) "
                    "VALUES (:id, :smiles, :fp)"
                ),
                {"id": mid, "smiles": smiles, "fp": "0" * 2048},
            )
            conn.execute(
                text(
                    "INSERT INTO molecule_sources (provider, molecule_id, source_url) "
                    "VALUES ('ChEMBL', :id, :url)"
                ),
                {"id": mid, "url": ident},
            )

    command.upgrade(cfg, RENAME_REV)
    cols = {c["name"]: c for c in inspect(engine).get_columns("molecule_sources")}
    assert "source_identifier" in cols and "source_url" not in cols
    assert cols["source_identifier"]["nullable"] is False
    with engine.connect() as conn:
        upgraded = conn.execute(
            text(
                "SELECT molecule_id, provider, source_identifier FROM molecule_sources"
            )
        ).all()
    assert set(upgraded) == expected

    command.downgrade(cfg, INITIAL_REV)
    col_names = {c["name"] for c in inspect(engine).get_columns("molecule_sources")}
    assert "source_url" in col_names and "source_identifier" not in col_names
    with engine.connect() as conn:
        restored = conn.execute(
            text("SELECT molecule_id, provider, source_url FROM molecule_sources")
        ).all()
    assert set(restored) == expected


def test_full_downgrade_to_base_drops_tables_and_enum_and_reupgrades_cleanly(
    empty_db: tuple[str, Engine],
) -> None:
    db_url, engine = empty_db
    cfg = make_alembic_config(db_url)
    command.upgrade(cfg, "head")
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
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_server_default": True})
        assert compare_metadata(ctx, Base.metadata) == []


def test_orm_models_match_alembic_migrated_schema(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_server_default": True})
        assert compare_metadata(ctx, Base.metadata) == []


def test_migrated_schema_enforces_pg_types_and_structural_keys(
    pg_engine: Engine,
) -> None:
    inspector = inspect(pg_engine)
    assert inspector.get_pk_constraint("molecule_sources")["constrained_columns"] == [
        "provider",
        "molecule_id",
    ]
    for table in ("molecule_embeddings", "molecule_sources"):
        assert any(
            fk["constrained_columns"] == ["molecule_id"]
            and fk["referred_table"] == "molecules"
            and fk["referred_columns"] == ["id"]
            and fk["options"].get("ondelete") == "CASCADE"
            for fk in inspector.get_foreign_keys(table)
        )
    assert any(
        idx["unique"] and idx["column_names"] == ["standardized_smiles"]
        for idx in inspector.get_indexes("molecules")
    )
    assert any(
        set(uq["column_names"])
        == {"molecule_id", "embedding_model", "embedding_model_version"}
        for uq in inspector.get_unique_constraints("molecule_embeddings")
    )

    with pg_engine.connect() as conn:
        pg_types = dict(
            conn.execute(
                text(
                    "SELECT c.table_name || '.' || c.column_name, "
                    "format_type(a.atttypid, a.atttypmod) "
                    "FROM information_schema.columns c "
                    "JOIN pg_attribute a ON a.attrelid = "
                    "(quote_ident(c.table_schema) || '.' || quote_ident(c.table_name))::regclass "
                    "AND a.attname = c.column_name "
                    "WHERE c.table_schema = 'public' "
                    "AND c.table_name IN ('molecules', 'molecule_embeddings', 'analyses')"
                )
            ).all()
        )
    assert pg_types["molecules.morgan_fingerprint"] == "bit(2048)"
    assert pg_types["molecule_embeddings.embedding"] == "vector"
    assert pg_types["analyses.config"] == "jsonb"
    assert pg_types["analyses.results"] == "jsonb"
    assert pg_types["analyses.created_at"] == "timestamp with time zone"
