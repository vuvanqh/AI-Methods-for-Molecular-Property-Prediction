import asyncio
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session, sessionmaker

from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis
from src.db.session import get_db
from src.main import app, lifespan
from src.model import ErrorResponse, PredictionResponse
from src.rag.config import RAG_TOP_K
from src.rag.embeddings import unimol_embeddings as unimol_module
from tests.fakes import LLM_SUMMARY, TASKS, FakeExternalServices
from tests.integration.conftest import seed_molecule

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("isolated_rag_settings")]

# Bound before `mock_http` monkeypatches `httpx.AsyncClient` for outbound clients
_ASGIClient = httpx.AsyncClient


def _vec512(x: float, y: float) -> list[float]:
    return [x, y, *([0.0] * 510)]


@pytest.fixture
def fake_unimol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.get_repr.side_effect = lambda batch: [_vec512(1.0, 0.0) for _ in batch]
    monkeypatch.setattr(unimol_module, "UniMolRepr", MagicMock(return_value=fake_model))


@pytest.fixture
def live_db_client(
    external_services: FakeExternalServices,
    fake_unimol: None,
    db_dependency: Callable[[], Iterator[Session]],
    clean_db: None,
) -> Iterator[TestClient]:
    """Boot FastAPI wired to real PostgreSQL + pgvector, faking only external HTTP APIs and UniMol weights."""
    app.dependency_overrides[get_db] = db_dependency
    try:
        with TestClient(app) as client:
            try:
                yield client
            finally:
                timer = app.state.retriever.embedding_model._unload_timer
                if timer is not None:
                    timer.cancel()
    finally:
        app.dependency_overrides.pop(get_db, None)


def _seed_corpus(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as db:
        seed_molecule(
            db,
            "CCO",
            sources=[("ChEMBL", "CHEMBL545")],
            embeddings=[("unimol", "unimolv1", _vec512(0.6, 0.8))],
        )
        seed_molecule(
            db,
            "O=C(O)c1ccccc1O",
            sources=[("ChEMBL", "CHEMBL424"), ("UnknownProvider", "U1")],
            embeddings=[("unimol", "unimolv1", _vec512(0.95, 0.3122499))],
        )


def test_predict_endpoint_commits_completed_analysis_and_round_trips_jsonb(
    live_db_client: TestClient,
    external_services: FakeExternalServices,
    pg_session_factory: sessionmaker[Session],
) -> None:
    _seed_corpus(pg_session_factory)

    resp = live_db_client.post("/api/predict-properties", json={"smiles": "CN(=O)=O"})

    assert resp.status_code == 200
    http_body = PredictionResponse(**resp.json())
    assert set(http_body.properties_xai_data.predictions) == set(TASKS)
    assert len(http_body.similar_compounds) == 2
    salicylic = next(
        c for c in http_body.similar_compounds if c.smiles == "O=C(O)c1ccccc1O"
    )
    # UnknownProvider is filtered out by SOURCE_URL_TEMPLATES; ChEMBL ID resolved to URL
    assert salicylic.sources == [
        "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424"
    ]
    assert "KCNH2 (ChEMBL): IC50 = 14 nM" in external_services.last_llm_prompt

    # Query back from an independent session to prove the transaction committed
    with pg_session_factory() as db:
        rows = db.scalars(select(Analysis)).all()

    assert len(rows) == 1
    stored = rows[0]
    assert stored.standardized_input_smiles == "C[N+](=O)[O-]"
    assert stored.status == AnalysisStatus.COMPLETED
    assert stored.created_at is not None
    assert stored.config == {
        "input_smiles": "CN(=O)=O",
        "retrieval_type": "fusion",
        "top_k": RAG_TOP_K,
    }
    assert PredictionResponse.model_validate(stored.results) == http_body
    assert stored.results["llm_summary"] == LLM_SUMMARY


def test_predict_endpoint_commits_completed_analysis_when_llm_and_chembl_degrade(
    live_db_client: TestClient,
    external_services: FakeExternalServices,
    pg_session_factory: sessionmaker[Session],
) -> None:
    _seed_corpus(pg_session_factory)
    external_services.llm_status = 503
    external_services.chembl_status = 503

    resp = live_db_client.post("/api/predict-properties", json={"smiles": "CCCO"})

    assert resp.status_code == 200
    with pg_session_factory() as db:
        rows = db.scalars(select(Analysis)).all()

    assert len(rows) == 1
    assert rows[0].status == AnalysisStatus.COMPLETED
    assert PredictionResponse.model_validate(rows[0].results).llm_summary == ""


def test_predict_endpoint_commits_failed_analysis_with_sql_null_results_on_gnn_outage(
    live_db_client: TestClient,
    external_services: FakeExternalServices,
    pg_session_factory: sessionmaker[Session],
) -> None:
    external_services.gnn_status = 503

    resp = live_db_client.post("/api/predict-properties", json={"smiles": "OCC"})

    assert resp.status_code == 503
    assert ErrorResponse(**resp.json()).message == "GNN service unavailable."

    with pg_session_factory() as db:
        rows = db.scalars(select(Analysis).where(Analysis.results.is_(None))).all()

    assert len(rows) == 1
    assert rows[0].standardized_input_smiles == "CCO"
    assert rows[0].status == AnalysisStatus.FAILED
    assert rows[0].config["input_smiles"] == "OCC"


def test_predict_endpoint_rolls_back_aborted_retrieval_tx_and_logs_failed_analysis(
    live_db_client: TestClient, pg_session_factory: sessionmaker[Session]
) -> None:
    # Mismatched 4-d embedding under the active model causes pgvector DataError, aborting the DB tx
    with pg_session_factory() as db:
        seed_molecule(db, "CCCO", embeddings=[("unimol", "unimolv1", [0.1] * 4)])

    with pytest.raises(DataError):
        live_db_client.post("/api/predict-properties", json={"smiles": "CCO"})

    with pg_session_factory() as db:
        rows = db.scalars(select(Analysis).where(Analysis.results.is_(None))).all()

    assert len(rows) == 1
    assert rows[0].standardized_input_smiles == "CCO"
    assert rows[0].status == AnalysisStatus.FAILED


def test_predict_endpoint_writes_nothing_on_unbonded_smiles_after_db_retrieval(
    live_db_client: TestClient, pg_session_factory: sessionmaker[Session]
) -> None:
    # `[Na+]` passes describe_molecule and executes Postgres retrieval before GNN returns 422
    resp = live_db_client.post("/api/predict-properties", json={"smiles": "[Na+]"})

    assert resp.status_code == 422
    with pg_session_factory() as db:
        assert db.scalars(select(Analysis)).all() == []


def test_concurrent_predictions_exceeding_pool_size_persist_all_analyses(
    external_services: FakeExternalServices,
    fake_unimol: None,
    db_dependency: Callable[[], Iterator[Session]],
    pg_session_factory: sessionmaker[Session],
    clean_db: None,
) -> None:
    _seed_corpus(pg_session_factory)
    # 6 concurrent requests exceed SQLAlchemy's default pool_size=5
    smiles_batch = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CC(=O)O", "CCN"]

    async def _fire_all() -> list[httpx.Response]:
        app.dependency_overrides[get_db] = db_dependency
        try:
            async with lifespan(app):
                try:
                    transport = httpx.ASGITransport(app=app)
                    async with _ASGIClient(
                        transport=transport, base_url="http://testserver"
                    ) as ac:
                        return list(
                            await asyncio.gather(
                                *(
                                    ac.post(
                                        "/api/predict-properties", json={"smiles": s}
                                    )
                                    for s in smiles_batch
                                )
                            )
                        )
                finally:
                    timer = app.state.retriever.embedding_model._unload_timer
                    if timer is not None:
                        timer.cancel()
        finally:
            app.dependency_overrides.pop(get_db, None)

    responses = asyncio.run(_fire_all())

    assert [r.status_code for r in responses] == [200] * len(smiles_batch)
    with pg_session_factory() as db:
        rows = db.scalars(select(Analysis)).all()

    assert len(rows) == len(smiles_batch)
    assert len({row.id for row in rows}) == len(smiles_batch)
    assert {row.standardized_input_smiles for row in rows} == set(smiles_batch)
    assert all(
        row.status == AnalysisStatus.COMPLETED and row.results is not None
        for row in rows
    )
