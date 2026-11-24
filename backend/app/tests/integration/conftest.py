import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.community.postgres import PostgresContainer

from src.db.models import Base, Molecule, MoleculeEmbedding, MoleculeSource
from src.descriptors import standardize_smiles
from src.rag.config import retrievalSettings, unimolSettings
from src.rag.embeddings.unimol_embeddings import UniMolEmbeddings
from src.rag.morgan_fingerprint_helper import (
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
)

PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
BACKEND_DIR = Path(__file__).resolve().parents[3]


def make_alembic_config(db_url: str) -> Config:
    """Build an Alembic `Config` targeting `db_url` without touching `.env` or pytest loggers."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return cfg


@pytest.fixture
def isolated_rag_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin RAG settings so local `.env` overrides cannot perturb integration tests."""
    monkeypatch.setattr(unimolSettings, "model_name", "unimolv1")
    monkeypatch.setattr(UniMolEmbeddings, "version", "unimolv1")
    monkeypatch.setattr(retrievalSettings, "fingerprint_weight", 0.5)
    monkeypatch.setattr(retrievalSettings, "semantic_weight", 0.5)


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """Spin up an ephemeral `pgvector/pgvector:pg17` container and apply Alembic migrations once."""
    with PostgresContainer(
        image=PGVECTOR_IMAGE,
        driver="psycopg",
        username="postgres",
        password="postgres",
        dbname="molecules_test",
    ) as container:
        parsed = make_url(container.get_connection_url())
        if parsed.host == "localhost":
            parsed = parsed.set(host="127.0.0.1")
        url = parsed.render_as_string(hide_password=False)
        command.upgrade(make_alembic_config(url), "head")
        yield url


@pytest.fixture(scope="session")
def pg_engine(pg_url: str) -> Iterator[Engine]:
    engine = create_engine(pg_url, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def pg_session_factory(pg_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)


@pytest.fixture
def clean_db(pg_engine: Engine) -> None:
    """Truncate every ORM table before each test so tests stay independent while allowing real commits."""
    tables = ", ".join(Base.metadata.tables)
    with pg_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {tables} CASCADE"))


@pytest.fixture
def db_session(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> Iterator[Session]:
    with pg_session_factory() as session:
        yield session


@pytest.fixture
def db_dependency(
    pg_session_factory: sessionmaker[Session],
) -> Callable[[], Iterator[Session]]:
    """Zero-arg FastAPI `get_db` override yielding real PostgreSQL sessions per request."""

    def _get_test_db() -> Iterator[Session]:
        with pg_session_factory() as session:
            yield session

    return _get_test_db


def seed_molecule(
    session: Session,
    smiles: str,
    *,
    sources: Sequence[tuple[str, str]] = (),
    embeddings: Sequence[tuple[str, str, Sequence[float]]] = (),
) -> Molecule:
    """Insert a standardized `Molecule` with optional sources and embeddings into Postgres."""
    standardized = standardize_smiles(smiles)
    mol = Molecule(
        standardized_smiles=standardized,
        morgan_fingerprint=fingerprint_to_bitstring(
            generate_morgan_fingerprint(standardized)
        ),
        molecule_sources=[
            MoleculeSource(provider=p, source_identifier=i) for p, i in sources
        ],
        molecule_embedding=[
            MoleculeEmbedding(
                embedding_model=m, embedding_model_version=v, embedding=list(vec)
            )
            for m, v, vec in embeddings
        ],
    )
    session.add(mol)
    session.commit()
    session.refresh(mol)
    return mol
