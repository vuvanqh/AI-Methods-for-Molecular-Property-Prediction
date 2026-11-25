from typing import Any
from unittest.mock import create_autospec

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from src import controller as controller_module
from src import main as main_module
from src.db.session import create_session_factory, get_db
from src.descriptors import describe_molecule
from src.model import ErrorResponse, PredictionResponse
from tests.fakes import LLM_SUMMARY, gnn_payload


def _setup_controller(
    monkeypatch: pytest.MonkeyPatch, *, side_effect: Exception | None = None
) -> tuple[TestClient, list[str]]:
    called_smiles: list[str] = []

    async def _stub_predict(smiles: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        called_smiles.append(smiles)
        if side_effect is not None:
            raise side_effect
        return {
            "molecule": describe_molecule(smiles),
            "properties_xai_data": gnn_payload(smiles),
            "similar_compounds": [],
            "llm_summary": LLM_SUMMARY,
        }

    monkeypatch.setattr(controller_module, "predict_properties", _stub_predict)
    app = FastAPI()
    app.state.retriever = app.state.llm_client = app.state.gnn_client = object()
    app.state.evidence_sources = []
    app.dependency_overrides[get_db] = object
    app.include_router(controller_module.app_router, prefix="/api")
    return TestClient(app), called_smiles


def test_predict_properties_endpoint_forwards_smiles_and_returns_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, called_smiles = _setup_controller(monkeypatch)
    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 200
    parsed = PredictionResponse(**resp.json())
    assert parsed.molecule.smiles == "CCO" and parsed.llm_summary == LLM_SUMMARY
    assert called_smiles == ["CCO"]


@pytest.mark.parametrize(
    ("service_error", "expected_status", "expected_message"),
    [
        (
            ValueError("internal RDKit detail"),
            422,
            "The molecule could not be processed.",
        ),
        (
            ConnectionError("http://internal-gnn timed out"),
            503,
            "GNN service unavailable.",
        ),
    ],
)
def test_predict_properties_endpoint_maps_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
    service_error: Exception,
    expected_status: int,
    expected_message: str,
) -> None:
    client, _ = _setup_controller(monkeypatch, side_effect=service_error)
    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == expected_status
    assert ErrorResponse(**resp.json()).message == expected_message


def test_predict_properties_endpoint_rejects_missing_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, called_smiles = _setup_controller(monkeypatch)
    resp = client.post("/api/predict-properties", json={})

    assert resp.status_code == 422
    assert called_smiles == []


def test_get_db_yields_session_and_closes_it_when_request_errors() -> None:
    mock_session = create_autospec(Session, instance=True)
    test_app = FastAPI()
    test_app.state.db_session_factory = lambda: mock_session
    request = Request({"type": "http", "app": test_app})

    gen = get_db(request)
    assert next(gen) is mock_session
    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("boom"))

    mock_session.close.assert_called_once()


@pytest.mark.parametrize("set_none", [False, True])
def test_get_db_raises_when_session_factory_is_uninitialized(set_none: bool) -> None:
    test_app = FastAPI()
    if set_none:
        test_app.state.db_session_factory = None
    request = Request({"type": "http", "app": test_app})

    with pytest.raises(RuntimeError, match="not initialized"):
        next(get_db(request))


def test_lifespan_creates_and_disposes_db_engine_when_database_url_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disposed: list[Engine] = []
    original_dispose = Engine.dispose

    def _track_dispose(self: Engine, *args: Any, **kwargs: Any) -> None:
        disposed.append(self)
        original_dispose(self, *args, **kwargs)

    monkeypatch.setattr(Engine, "dispose", _track_dispose)
    monkeypatch.setattr(main_module.settings, "database_url", "sqlite:///:memory:")

    test_app = FastAPI(lifespan=main_module.lifespan)
    with TestClient(test_app):
        engine = test_app.state.db_engine
        assert isinstance(engine, Engine)
        assert test_app.state.db_session_factory is not None
        assert test_app.state.db_session_factory.kw["bind"] is engine
        assert disposed == []

    assert disposed == [engine]


def test_lifespan_skips_db_engine_creation_when_database_url_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = create_autospec(create_session_factory)
    monkeypatch.setattr(main_module.settings, "database_url", "")
    monkeypatch.setattr(main_module, "create_session_factory", create_mock)

    test_app = FastAPI(lifespan=main_module.lifespan)
    with TestClient(test_app):
        assert test_app.state.db_engine is None
        assert test_app.state.db_session_factory is None

    create_mock.assert_not_called()
