import asyncio
import time
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import TASKS
from src.controller import gnn_router, prediction_and_attribution_endpoint
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.model import ErrorResponse, GNNXAIInferenceResponse, InferenceRequest


def _sample_valid_response() -> dict[str, Any]:
    return {
        "predictions": {task: False for task in TASKS},
        "probabilities": {task: 0.25 for task in TASKS},
        "atomscores": {task: {0: 0.1, 1: -0.2, 2: None} for task in TASKS},
        "molblock": "stub\n  RDKit          3D\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n",
    }


class _StubInferenceService:
    """Stub `inference_service` collaborator for testing `controller.py` in isolation."""

    def __init__(self, exc: Exception | None = None, delay_s: float = 0.0) -> None:
        self._exc = exc
        self._delay_s = delay_s
        self.calls: list[str] = []
        self.active_count = 0
        self.max_concurrent_observed = 0

    def predict_and_explain(self, smiles: str) -> dict[str, Any]:
        self.calls.append(smiles)
        self.active_count += 1
        self.max_concurrent_observed = max(self.max_concurrent_observed, self.active_count)
        try:
            if self._delay_s > 0.0:
                time.sleep(self._delay_s)
            if self._exc is not None:
                raise self._exc
            return _sample_valid_response()
        finally:
            self.active_count -= 1


class _DisconnectedRequestStub:
    """Minimal ASGI request stub simulating a client that disconnected while queued."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def is_disconnected(self) -> bool:
        return True


def _create_test_app(service: _StubInferenceService) -> FastAPI:
    test_app = FastAPI()
    test_app.state.inference_service = service
    test_app.state.inference_lock = asyncio.Lock()
    test_app.include_router(gnn_router, prefix="/gnn-xai")
    return test_app


def test_controller_returns_200_and_passes_smiles_to_service() -> None:
    stub_service = _StubInferenceService()
    client = TestClient(_create_test_app(stub_service))

    response = client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"})

    assert response.status_code == 200
    validated = GNNXAIInferenceResponse(**response.json())
    assert "V2000" in validated.molblock
    assert stub_service.calls == ["CCO"]


def test_controller_returns_422_when_service_raises_molecule_validation_error() -> None:
    stub_service = _StubInferenceService(exc=MoleculeValidationError("The molecule must contain at least one bond."))
    client = TestClient(_create_test_app(stub_service))

    response = client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "C"})

    assert response.status_code == 422
    error = ErrorResponse(**response.json())
    assert error.message == "Invalid molecular input."


def test_controller_returns_499_without_calling_service_when_client_disconnected() -> None:
    stub_service = _StubInferenceService()
    test_app = _create_test_app(stub_service)
    disconnected_request = _DisconnectedRequestStub(test_app)

    response = asyncio.run(
        prediction_and_attribution_endpoint(
            disconnected_request,
            InferenceRequest(smiles="CCO"),
        )
    )

    assert response.status_code == 499
    assert response.body == b""
    assert stub_service.calls == []


@pytest.mark.parametrize(
    ("raised_exception", "expected_message", "sensitive_detail"),
    [
        (ModelInferenceError("internal-shape-secret"), "Internal inference engine failure.", "internal-shape-secret"),
        (RuntimeError("cuda-driver-secret"), "Internal inference engine failure.", "cuda-driver-secret"),
        (
            ZeroDivisionError("unexpected-math-secret"),
            "An unexpected error occurred during prediction and attribution.",
            "unexpected-math-secret",
        ),
    ],
)
def test_controller_returns_500_without_leaking_internal_exception_details(
    raised_exception: Exception,
    expected_message: str,
    sensitive_detail: str,
) -> None:
    stub_service = _StubInferenceService(exc=raised_exception)
    client = TestClient(_create_test_app(stub_service))

    response = client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"})

    assert response.status_code == 500
    error = ErrorResponse(**response.json())
    assert error.message == expected_message
    assert sensitive_detail not in error.message


def test_controller_serializes_concurrent_requests_to_one_active_inference() -> None:
    stub_service = _StubInferenceService(delay_s=0.02)
    test_app = _create_test_app(stub_service)

    async def _run_concurrent() -> None:
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            responses = await asyncio.gather(
                client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"}),
                client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CC(C)O"}),
                client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "c1ccccc1"}),
            )

        for resp in responses:
            assert resp.status_code == 200

    asyncio.run(_run_concurrent())
    assert len(stub_service.calls) == 3
    assert stub_service.max_concurrent_observed == 1
