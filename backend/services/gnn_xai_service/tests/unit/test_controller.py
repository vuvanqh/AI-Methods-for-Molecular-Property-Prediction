"""Unit tests for `src.controller`: HTTP routing, error mapping, ASGI 499 disconnect, and recovery."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import TASKS
from src.controller import gnn_router
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.model import ErrorResponse, GNNXAIInferenceResponse
from tests.unit.conftest import make_valid_inference_response

ENDPOINT = "/gnn-xai/prediction_and_attribution"


class _StubInferenceService:
    """Deterministic `inference_service` test double with queued outcomes."""

    def __init__(
        self, outcomes: list[dict[str, Any] | Exception] | None = None
    ) -> None:
        self._outcomes = list(outcomes) if outcomes is not None else []
        self.calls: list[str] = []

    def predict_and_explain(self, smiles: str) -> dict[str, Any]:
        self.calls.append(smiles)
        outcome = (
            self._outcomes.pop(0) if self._outcomes else make_valid_inference_response()
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _create_test_app(service: _StubInferenceService) -> FastAPI:
    app = FastAPI()
    app.state.inference_service = service
    app.state.inference_lock = asyncio.Lock()
    app.include_router(gnn_router, prefix="/gnn-xai")
    return app


def test_returns_200_with_exact_payload_and_forwards_smiles() -> None:
    expected = make_valid_inference_response(num_heavy_atoms=3, num_explicit_h=2)
    stub = _StubInferenceService([expected])

    response = TestClient(_create_test_app(stub)).post(ENDPOINT, json={"smiles": "CCO"})

    assert response.status_code == 200
    validated = GNNXAIInferenceResponse(**response.json())
    assert validated == GNNXAIInferenceResponse(**expected)
    assert set(validated.predictions.keys()) == set(TASKS)
    assert stub.calls == ["CCO"]


def test_returns_422_on_malformed_schema_without_calling_service() -> None:
    stub = _StubInferenceService()

    response = TestClient(_create_test_app(stub)).post(ENDPOINT, json={})

    assert response.status_code == 422
    assert stub.calls == []


def test_rejects_get_with_405() -> None:
    stub = _StubInferenceService()

    response = TestClient(_create_test_app(stub)).get(ENDPOINT)

    assert response.status_code == 405
    assert stub.calls == []


@pytest.mark.parametrize(
    ("exc", "status", "expected_msg", "secret"),
    [
        (
            MoleculeValidationError("rdkit-secret"),
            422,
            "Invalid molecular input.",
            "rdkit-secret",
        ),
        (
            ModelInferenceError("shape-secret"),
            500,
            "Internal inference engine failure.",
            "shape-secret",
        ),
        (
            RuntimeError("cuda-secret"),
            500,
            "Internal inference engine failure.",
            "cuda-secret",
        ),
        (
            # ValueError is parent of MoleculeValidationError; must map to 500, not 422
            ValueError("value-secret"),
            500,
            "An unexpected error occurred during prediction and attribution.",
            "value-secret",
        ),
    ],
)
def test_maps_exceptions_without_leaking_details(
    exc: Exception, status: int, expected_msg: str, secret: str
) -> None:
    response = TestClient(_create_test_app(_StubInferenceService([exc]))).post(
        ENDPOINT, json={"smiles": "CCO"}
    )

    assert response.status_code == status
    assert ErrorResponse(**response.json()).message == expected_msg
    assert secret not in response.text


def test_returns_499_on_client_disconnect_via_asgi() -> None:
    stub = _StubInferenceService()
    test_app = _create_test_app(stub)

    async def _run() -> list[dict[str, Any]]:
        body = json.dumps({"smiles": "CCO"}).encode()
        body_delivered, disconnected = asyncio.Event(), asyncio.Event()
        body_sent = False
        sent: list[dict[str, Any]] = []
        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": ENDPOINT,
            "raw_path": ENDPOINT.encode(),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        async def _receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                body_delivered.set()
                return {"type": "http.request", "body": body, "more_body": False}
            if not disconnected.is_set():
                await disconnected.wait()
            return {"type": "http.disconnect"}

        async def _send(message: dict[str, Any]) -> None:
            sent.append(message)

        async with test_app.state.inference_lock:
            task = asyncio.create_task(test_app(scope, _receive, _send))
            await asyncio.wait_for(body_delivered.wait(), timeout=2.0)
            disconnected.set()

        await asyncio.wait_for(task, timeout=2.0)
        return sent

    sent = asyncio.run(_run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert start["status"] == 499
    assert body == b""
    assert stub.calls == []


def test_recovers_after_service_error() -> None:
    expected = make_valid_inference_response()
    stub = _StubInferenceService([ModelInferenceError("transient"), expected])
    test_app = _create_test_app(stub)

    async def _run() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r1 = await asyncio.wait_for(c.post(ENDPOINT, json={"smiles": "CCO"}), 2.0)
            r2 = await asyncio.wait_for(c.post(ENDPOINT, json={"smiles": "CC"}), 2.0)
            return r1, r2

    failed, recovered = asyncio.run(_run())
    assert failed.status_code == 500
    assert recovered.status_code == 200
    assert GNNXAIInferenceResponse(**recovered.json()) == GNNXAIInferenceResponse(
        **expected
    )
    assert stub.calls == ["CCO", "CC"]
