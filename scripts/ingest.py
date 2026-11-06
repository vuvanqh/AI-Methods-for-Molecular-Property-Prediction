"""Fetch ChEMBL small molecules, compute Morgan fingerprints and Uni-Mol embeddings, load them into Postgres.

Run from the repo root:  uv run python -m scripts.ingest
Safe to re-run: molecules, sources and embeddings that already exist are skipped.
"""

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.src.db.models import Molecule, MoleculeEmbedding, MoleculeSource
from backend.app.src.db.models.base import Base

from .embeddings import EMBEDDING_MODEL, unimol_embeddings
from .fetch_molecules import fetch_molecule_pages
from .fingerprints import morgan_fingerprint, standardize_smiles
from .source_data_models.chembl_source import ChEMBLSource

PROVIDER = "ChEMBL"


class IngestSettings(BaseSettings):
    # default matches docker-compose.yml; DATABASE_URL in .env overrides it
    # 127.0.0.1 not localhost: on Windows localhost tries IPv6 first and hangs against Docker's IPv4-only port
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/molecules"
    )
    max_molecules: int = 10_000
    page_size: int = 1000  # ChEMBL's maximum
    embedding_model_version: str = "unimolv1"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_page(session: Session, records: list[dict], embedding_version: str) -> int:
    """Upsert one page of ChEMBL records; returns how many embeddings were newly computed."""
    chembl_ids: dict[str, str] = {}
    for record in records:
        smiles = (record.get("molecule_structures") or {}).get("canonical_smiles")
        standardized = smiles and standardize_smiles(smiles)
        if standardized:
            chembl_ids.setdefault(standardized, record["molecule_chembl_id"])
    if not chembl_ids:
        return 0

    session.execute(
        insert(Molecule).on_conflict_do_nothing(index_elements=["standardized_smiles"]),
        [
            {"standardized_smiles": s, "morgan_fingerprint": morgan_fingerprint(s)}
            for s in chembl_ids
        ],
    )

    molecule_ids = dict(
        session.execute(
            select(Molecule.standardized_smiles, Molecule.id).where(
                Molecule.standardized_smiles.in_(chembl_ids)
            )
        ).all()
    )

    session.execute(
        insert(MoleculeSource).on_conflict_do_nothing(),
        [
            {
                "provider": PROVIDER,
                "molecule_id": molecule_ids[s],
                "source_identifier": chembl_id,
            }
            for s, chembl_id in chembl_ids.items()
        ],
    )

    # embeddings are the slow step, so only compute the ones this model version doesn't have yet
    already_embedded = set(
        session.scalars(
            select(MoleculeEmbedding.molecule_id).where(
                MoleculeEmbedding.molecule_id.in_(molecule_ids.values()),
                MoleculeEmbedding.embedding_model == EMBEDDING_MODEL,
                MoleculeEmbedding.embedding_model_version == embedding_version,
            )
        )
    )
    to_embed = [s for s, mid in molecule_ids.items() if mid not in already_embedded]
    if to_embed:
        vectors = unimol_embeddings(to_embed, embedding_version)
        session.execute(
            insert(MoleculeEmbedding).on_conflict_do_nothing(),
            [
                {
                    "molecule_id": molecule_ids[s],
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_model_version": embedding_version,
                    "embedding": vector,
                }
                # strict: a length mismatch would attach embeddings to the wrong molecules
                for s, vector in zip(to_embed, vectors, strict=True)
            ],
        )

    session.commit()
    return len(to_embed)


async def main() -> None:
    settings = IngestSettings()
    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # shortcut: create_all never alters existing tables; switch to Alembic migrations once the schema settles
    Base.metadata.create_all(engine)

    seen = 0
    with Session(engine) as session:
        async for page in fetch_molecule_pages(
            ChEMBLSource(), settings.max_molecules, settings.page_size
        ):
            embedded = load_page(session, page, settings.embedding_model_version)
            seen += len(page)
            print(
                f"{seen} ChEMBL records processed, {embedded} new embeddings in this page"
            )


if __name__ == "__main__":
    asyncio.run(main())
