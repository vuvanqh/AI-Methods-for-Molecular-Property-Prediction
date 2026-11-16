"""Fetch ChEMBL small molecules, compute Morgan fingerprints and Uni-Mol embeddings, load them into Postgres.

Run from the repo root, after creating the tables once with `cd backend && uv run --project .. alembic upgrade head`:
    uv run python -m scripts.ingestion.ingest
Safe to re-run: molecules, sources and embeddings that already exist are skipped.
"""

import asyncio
import logging
import time

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.src.db.models import Molecule, MoleculeEmbedding, MoleculeSource

from .embeddings import EMBEDDING_MODEL, unimol_embeddings
from .fetch_molecules import fetch_molecule_pages
from .fingerprints import morgan_fingerprint, standardize_smiles
from .source_data_models.chembl_source import ChEMBLSource

PROVIDER = "ChEMBL"

log = logging.getLogger("ingest")


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


def load_page(session: Session, records: list[dict], embedding_version: str) -> dict:
    """Upsert one page of ChEMBL records; returns counts and per-step timings for the log."""
    stats = {"unparseable": 0, "new_molecules": 0, "embedded": 0, "embed_s": 0.0}
    start = time.perf_counter()
    chembl_ids: dict[str, str] = {}
    for record in records:
        smiles = (record.get("molecule_structures") or {}).get("canonical_smiles")
        standardized = smiles and standardize_smiles(smiles)
        if standardized:
            chembl_ids.setdefault(standardized, record["molecule_chembl_id"])
        else:
            stats["unparseable"] += 1
    if not chembl_ids:
        return stats

    rows = [
        {"standardized_smiles": s, "morgan_fingerprint": morgan_fingerprint(s)}
        for s in chembl_ids
    ]
    stats["prepare_s"] = time.perf_counter() - start

    # RETURNING with DO NOTHING yields only the rows actually inserted, i.e. the new molecules
    stats["new_molecules"] = len(
        session.scalars(
            insert(Molecule)
            .on_conflict_do_nothing(index_elements=["standardized_smiles"])
            .returning(Molecule.id),
            rows,
        ).all()
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
        embed_start = time.perf_counter()
        vectors = unimol_embeddings(to_embed, embedding_version)
        stats["embed_s"] = time.perf_counter() - embed_start
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
    stats["embedded"] = len(to_embed)
    stats["total_s"] = time.perf_counter() - start
    return stats


async def main() -> None:
    settings = IngestSettings()
    engine = create_engine(settings.database_url)
    log.info(
        "ingesting up to %d ChEMBL molecules, %d per page, into %s (embeddings: %s/%s)",
        settings.max_molecules,
        settings.page_size,
        engine.url.render_as_string(hide_password=True),
        EMBEDDING_MODEL,
        settings.embedding_model_version,
    )
    totals = {"fetched": 0, "unparseable": 0, "new_molecules": 0, "embedded": 0}
    page_no, run_start = 0, time.perf_counter()  # page_no counts completed pages
    fetch_start = run_start
    try:
        with Session(engine) as session:
            async for page in fetch_molecule_pages(
                ChEMBLSource(), settings.max_molecules, settings.page_size
            ):
                fetch_s = time.perf_counter() - fetch_start
                stats = load_page(session, page, settings.embedding_model_version)
                for key in totals:
                    totals[key] += len(page) if key == "fetched" else stats[key]
                total_s = stats.get("total_s", 0.0)
                log.info(
                    "page %d (offset %d): %d fetched, %d unparseable, %d new molecules, %d embedded"
                    " | fetch %.1fs, prepare %.1fs, embed %.1fs, write %.1fs",
                    page_no + 1,
                    page_no * settings.page_size,
                    len(page),
                    stats["unparseable"],
                    stats["new_molecules"],
                    stats["embedded"],
                    fetch_s,
                    stats.get("prepare_s", 0.0),
                    stats["embed_s"],
                    total_s - stats.get("prepare_s", 0.0) - stats["embed_s"],
                )
                page_no += 1
                fetch_start = time.perf_counter()
    except Exception:
        # pages are committed one by one, so a re-run skips everything before this point
        log.exception(
            "failed on page %d (offset %d); earlier pages are saved, re-run to continue",
            page_no + 1,
            page_no * settings.page_size,
        )
        raise
    log.info(
        "done in %.1fs: %d fetched, %d unparseable, %d new molecules, %d embedded",
        time.perf_counter() - run_start,
        *totals.values(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # one long request line per page; the page log covers it
    asyncio.run(main())
