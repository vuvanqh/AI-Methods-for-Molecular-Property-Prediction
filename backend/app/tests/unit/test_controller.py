from typing import Any
from unittest.mock import AsyncMock, create_autospec

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
from src.service import predict_properties
from tests.fakes import LLM_SUMMARY, gnn_payload


def _sample_prediction(smiles: str = "CCO") -> dict[str, Any]:
    return {
        "molecule": describe_molecule(smiles),
        "properties_xai_data": gnn_payload(smiles),
        "similar_compounds": [],
        "llm_summary": LLM_SUMMARY,
    }


def _setup_controller(
    monkeypatch: pytest.MonkeyPatch, side_effect: Exception | None = None
) -> tuple[TestClient, AsyncMock, FastAPI, Session]:
    predict_mock = create_autospec(predict_properties)
    if side_effect is not None:
        predict_mock.side_effect = side_effect
    else:
        predict_mock.return_value = _sample_prediction("CCO")
    monkeypatch.setattr(controller_module, "predict_properties", predict_mock)

    db_session = create_autospec(Session, instance=True)
    test_app = FastAPI()
    test_app.state.retriever = object()
    test_app.state.evidence_sources = [object()]
    test_app.state.llm_client = object()
    test_app.state.gnn_client = object()
    test_app.dependency_overrides[get_db] = lambda: db_session
    test_app.include_router(controller_module.app_router, prefix="/api")
    return TestClient(test_app), predict_mock, test_app, db_session


def test_controller_forwards_state_and_db_to_service_and_returns_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, predict_mock, test_app, db_session = _setup_controller(monkeypatch)

    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 200
    assert PredictionResponse(**resp.json()).molecule.smiles == "CCO"
    predict_mock.assert_awaited_once_with(
        "CCO",
        test_app.state.retriever,
        test_app.state.evidence_sources,
        test_app.state.llm_client,
        test_app.state.gnn_client,
        db_session,
    )


def test_controller_maps_value_error_to_422_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = _setup_controller(
        monkeypatch, side_effect=ValueError("internal RDKit parse detail")
    )

    resp = client.post("/api/predict-properties", json={"smiles": "bad"})

    assert resp.status_code == 422
    msg = ErrorResponse(**resp.json()).message
    assert msg == "The molecule could not be processed."
    assert "RDKit" not in msg


@pytest.mark.parametrize(
    "payload", [{}, {"wrong": "CCO"}, {"smiles": None}, {"smiles": 1}]
)
def test_controller_returns_422_on_malformed_request_body(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    client, predict_mock, _, _ = _setup_controller(monkeypatch)

    resp = client.post("/api/predict-properties", json=payload)

    assert resp.status_code == 422
    predict_mock.assert_not_awaited()


def test_controller_maps_connection_error_to_503_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = _setup_controller(
        monkeypatch, side_effect=ConnectionError("http://internal-gnn:8000 timed out")
    )

    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 503
    msg = ErrorResponse(**resp.json()).message
    assert msg == "GNN service unavailable."
    assert "internal-gnn" not in msg


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
