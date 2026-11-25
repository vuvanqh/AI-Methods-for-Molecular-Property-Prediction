import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session, sessionmaker

from src.analysis_logging import log_analysis
from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis
from src.db.session import get_db
from src.main import app, lifespan
from src.model import PredictionResponse
from src.rag.config import RAG_TOP_K
from src.rag.embeddings import unimol_embeddings as unimol_module
from tests.fakes import LLM_SUMMARY, TASKS, FakeExternalServices
from tests.pg_fixtures import seed_molecule

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("isolated_rag_settings")]

# Bound before `mock_http` monkeypatches `httpx.AsyncClient` for outbound clients
_ASGIClient = httpx.AsyncClient


def _vec512(x: float, y: float) -> list[float]:
    return [x, y, *([0.0] * 510)]


def _cancel_unload_timer() -> None:
    timer = app.state.retriever.embedding_model._unload_timer
    if timer is not None:
        timer.cancel()


@contextmanager
def _live_app_client(db_dep: Callable[[], Iterator[Session]]) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = db_dep
    try:
        with TestClient(app) as client:
            try:
                yield client
            finally:
                _cancel_unload_timer()
    finally:
        app.dependency_overrides.pop(get_db, None)


@asynccontextmanager
async def _live_async_client(
    db_dep: Callable[[], Iterator[Session]],
) -> AsyncIterator[httpx.AsyncClient]:
    app.dependency_overrides[get_db] = db_dep
    try:
        async with lifespan(app):
            try:
                transport = httpx.ASGITransport(app=app)
                async with _ASGIClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    yield client
            finally:
                _cancel_unload_timer()
    finally:
        app.dependency_overrides.pop(get_db, None)


class _FakeUniMolRepr:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def get_repr(self, batch: list[str]) -> list[list[float]]:
        return [_vec512(1.0, 0.0) for _ in batch]


@pytest.fixture
def fake_unimol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unimol_module, "UniMolRepr", _FakeUniMolRepr)


@pytest.fixture
def live_db_client(
    external_services: FakeExternalServices,
    fake_unimol: None,
    db_dependency: Callable[[], Iterator[Session]],
    clean_db: None,
) -> Iterator[TestClient]:
    with _live_app_client(db_dependency) as client:
        yield client


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


def test_log_analysis_swallows_sqlalchemy_error_and_leaves_session_usable(
    pg_session_factory: sessionmaker[Session],
    clean_db: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pg_session_factory() as db:
        log_analysis(
            db,
            standardized_smiles="CCO",
            status=AnalysisStatus.COMPLETED,
            config=None,  # type: ignore[arg-type]
            response={"ok": True},
        )
        assert db.scalar(text("SELECT 1")) == 1

    assert "Failed to log analysis for CCO" in caplog.text
    with pg_session_factory() as verify_db:
        assert verify_db.scalars(select(Analysis)).all() == []


def test_predict_endpoint_commits_completed_analysis_and_round_trips_jsonb(
    live_db_client: TestClient, pg_session_factory: sessionmaker[Session]
) -> None:
    _seed_corpus(pg_session_factory)

    # Sodium acetate distinguishes rdMolStandardize.Cleanup ("CC(=O)[O-].[Na+]")
    # from plain RDKit canonicalization ("CC(=O)[O][Na]").
    resp = live_db_client.post(
        "/api/predict-properties", json={"smiles": "CC(=O)O[Na]"}
    )

    assert resp.status_code == 200
    http_body = PredictionResponse(**resp.json())
    assert set(http_body.properties_xai_data.predictions) == set(TASKS)
    assert {c.smiles for c in http_body.similar_compounds} == {"CCO", "O=C(O)c1ccccc1O"}
    salicylic = next(
        c for c in http_body.similar_compounds if c.smiles == "O=C(O)c1ccccc1O"
    )
    assert salicylic.sources == [
        "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424"
    ]

    with pg_session_factory() as db:
        stored = db.scalars(select(Analysis)).one()

    assert stored.standardized_input_smiles == "CC(=O)[O-].[Na+]"
    assert stored.status == AnalysisStatus.COMPLETED
    assert stored.created_at is not None
    assert stored.config == {
        "input_smiles": "CC(=O)O[Na]",
        "retrieval_type": "fusion",
        "top_k": RAG_TOP_K,
    }
    assert PredictionResponse.model_validate(stored.results) == http_body
    assert stored.results["llm_summary"] == LLM_SUMMARY


def test_predict_endpoint_still_returns_200_when_completed_analysis_write_fails_in_postgres(
    external_services: FakeExternalServices,
    fake_unimol: None,
    pg_session_factory: sessionmaker[Session],
    clean_db: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_corpus(pg_session_factory)

    def _read_only_db() -> Iterator[Session]:
        with pg_session_factory() as session:
            # Retrieval SELECTs succeed; INSERT into `analyses` fails with ReadOnlySqlTransaction
            session.execute(text("SET TRANSACTION READ ONLY"))
            yield session

    with _live_app_client(_read_only_db) as client:
        resp = client.post("/api/predict-properties", json={"smiles": "CCCO"})

    assert resp.status_code == 200
    body = PredictionResponse(**resp.json())
    assert set(body.properties_xai_data.predictions) == set(TASKS)
    assert len(body.similar_compounds) == 2
    assert body.llm_summary == LLM_SUMMARY
    assert "Failed to log analysis for CCCO" in caplog.text
    with pg_session_factory() as db:
        assert db.scalars(select(Analysis)).all() == []


def test_predict_endpoint_rolls_back_aborted_retrieval_tx_and_logs_failed_analysis(
    live_db_client: TestClient, pg_session_factory: sessionmaker[Session]
) -> None:
    with pg_session_factory() as db:
        seed_molecule(db, "CCCO", embeddings=[("unimol", "unimolv1", [0.1] * 4)])

    with pytest.raises(DataError):
        live_db_client.post("/api/predict-properties", json={"smiles": "CCO"})

    with pg_session_factory() as db:
        stored = db.scalars(select(Analysis).where(Analysis.results.is_(None))).one()
    assert stored.standardized_input_smiles == "CCO"
    assert stored.status == AnalysisStatus.FAILED


def test_concurrent_mixed_success_failure_and_invalid_requests_maintain_db_isolation(
    external_services: FakeExternalServices,
    mock_http: Callable[[Callable[[httpx.Request], httpx.Response]], None],
    fake_unimol: None,
    db_dependency: Callable[[], Iterator[Session]],
    pg_session_factory: sessionmaker[Session],
    clean_db: None,
) -> None:
    _seed_corpus(pg_session_factory)

    def _mixed_handler(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/gnn-xai/prediction_and_attribution"
            and json.loads(request.content).get("smiles") == "CCCCO"
        ):
            return httpx.Response(503, json={"message": "GNN overloaded"})
        return external_services.handle(request)

    mock_http(_mixed_handler)
    batch = ["CCO", "CCCO", "CCCCO", "invalid_smiles", "[Na+]"]

    async def _fire_mixed() -> list[httpx.Response]:
        async with _live_async_client(db_dependency) as client:
            return list(
                await asyncio.gather(
                    *(
                        client.post("/api/predict-properties", json={"smiles": s})
                        for s in batch
                    )
                )
            )

    responses = asyncio.run(_fire_mixed())

    assert [r.status_code for r in responses] == [200, 200, 503, 422, 422]
    with pg_session_factory() as db:
        by_smiles = {
            r.standardized_input_smiles: r for r in db.scalars(select(Analysis)).all()
        }
    assert set(by_smiles) == {"CCO", "CCCO", "CCCCO"}
    assert by_smiles["CCO"].status == AnalysisStatus.COMPLETED
    assert by_smiles["CCO"].results is not None
    assert by_smiles["CCCO"].status == AnalysisStatus.COMPLETED
    assert by_smiles["CCCO"].results is not None
    assert by_smiles["CCCCO"].status == AnalysisStatus.FAILED
    assert by_smiles["CCCCO"].results is None
