import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import Response
from fastapi.responses import JSONResponse

from src.controller import prediction_and_attribution_endpoint
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.model import InferenceRequest


def _make_mock_request(
    service_mock: MagicMock, disconnected: bool = False
) -> MagicMock:
    req = MagicMock()
    req.app.state.inference_service = service_mock
    req.app.state.inference_lock = asyncio.Lock()
    req.is_disconnected = AsyncMock(return_value=disconnected)
    return req


def test_controller_happy_path_returns_service_result() -> None:
    expected = {
        "predictions": {},
        "probabilities": {},
        "atomscores": {},
        "molblock": "MOCK",
    }
    service_mock = MagicMock()
    service_mock.predict_and_explain.return_value = expected
    req = _make_mock_request(service_mock, disconnected=False)

    result = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles="CCO"))
    )

    assert result == expected
    service_mock.predict_and_explain.assert_called_once_with("CCO")


def test_controller_disconnected_client_returns_499_without_running_service() -> None:
    service_mock = MagicMock()
    req = _make_mock_request(service_mock, disconnected=True)

    response = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles="CCO"))
    )

    assert isinstance(response, Response)
    assert response.status_code == 499
    assert response.body == b""
    service_mock.predict_and_explain.assert_not_called()


def test_controller_molecule_validation_error_returns_422() -> None:
    service_mock = MagicMock()
    service_mock.predict_and_explain.side_effect = MoleculeValidationError(
        "SMILES must not be empty."
    )
    req = _make_mock_request(service_mock, disconnected=False)

    response = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles=""))
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 422
    assert json.loads(response.body) == {"message": "Invalid molecular input."}


def test_controller_model_inference_error_returns_500() -> None:
    service_mock = MagicMock()
    service_mock.predict_and_explain.side_effect = ModelInferenceError(
        "NaN or Inf detected in model output logits."
    )
    req = _make_mock_request(service_mock, disconnected=False)

    response = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles="CCO"))
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert json.loads(response.body) == {
        "message": "Internal inference engine failure."
    }


def test_controller_native_runtime_error_returns_500() -> None:
    service_mock = MagicMock()
    service_mock.predict_and_explain.side_effect = RuntimeError("CUDA out of memory")
    req = _make_mock_request(service_mock, disconnected=False)

    response = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles="CCO"))
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert json.loads(response.body) == {
        "message": "Internal inference engine failure."
    }


def test_controller_unexpected_exception_returns_generic_500() -> None:
    service_mock = MagicMock()
    service_mock.predict_and_explain.side_effect = ZeroDivisionError(
        "secret internal division by zero"
    )
    req = _make_mock_request(service_mock, disconnected=False)

    response = asyncio.run(
        prediction_and_attribution_endpoint(req, InferenceRequest(smiles="CCO"))
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert json.loads(response.body) == {
        "message": "An unexpected error occurred during prediction and attribution."
    }
