import asyncio
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import BINARY_PREDICTION_THRESHOLDS, TASKS
from src.main import app
from src.model import ErrorResponse, GNNXAIInferenceResponse


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Start the production FastAPI app once via `lifespan` for end-to-end HTTP tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("smiles", "expected_heavy_atoms", "expected_total_atoms"),
    [
        ("CCO", 3, 9),
        ("c1ccccc1", 6, 12),
        ("C[C@H](O)Cl", 4, 9),
        ("[2H]CCO", 4, 9),
        ("CC(=O)Oc1ccccc1C(=O)O", 13, 21),
    ],
)
def test_e2e_post_valid_smiles_returns_200_and_valid_schema(
    client: TestClient,
    smiles: str,
    expected_heavy_atoms: int,
    expected_total_atoms: int,
) -> None:
    response = client.post(
        "/gnn-xai/prediction_and_attribution",
        json={"smiles": smiles},
    )
    assert response.status_code == 200
    validated = GNNXAIInferenceResponse(**response.json())
    assert "V2000" in validated.molblock
    assert "M  END" in validated.molblock
    assert set(validated.predictions.keys()) == set(TASKS)
    assert set(validated.probabilities.keys()) == set(TASKS)
    assert set(validated.atomscores.keys()) == set(TASKS)

    for task in TASKS:
        prob = validated.probabilities[task]
        assert validated.predictions[task] is (
            prob > BINARY_PREDICTION_THRESHOLDS[task]
        )

        task_scores = validated.atomscores[task]
        assert len(task_scores) == expected_total_atoms
        for i in range(expected_heavy_atoms):
            assert isinstance(task_scores[i], float)
        for i in range(expected_heavy_atoms, expected_total_atoms):
            assert task_scores[i] is None


def test_e2e_equivalent_smiles_produce_identical_responses(client: TestClient) -> None:
    resp_a = client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"})
    resp_b = client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "OCC"})

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_a.json() == resp_b.json()


@pytest.mark.parametrize(
    "bad_smiles",
    [
        "",
        "   ",
        "\t\n",
        "invalid_smiles_string",
        "C(C)(C)(C)(C)C",
        "C",
        "O",
        "[Na+]",
        "[Na+].[Cl-]",
        "O.O",
        "[H]C([H])([H])[H]",
    ],
)
def test_e2e_post_invalid_empty_or_zero_bond_smiles_returns_422_error_response(
    client: TestClient,
    bad_smiles: str,
) -> None:
    response = client.post(
        "/gnn-xai/prediction_and_attribution",
        json={"smiles": bad_smiles},
    )
    assert response.status_code == 422
    error = ErrorResponse(**response.json())
    assert error.message == "Invalid molecular input."


@pytest.mark.parametrize(
    "malformed_payload",
    [
        {},
        {"wrong_key": "CCO"},
        {"smiles": None},
        {"smiles": 12345},
        {"smiles": ["CCO"]},
    ],
)
def test_e2e_post_schema_validation_errors_return_422(
    client: TestClient,
    malformed_payload: dict,
) -> None:
    response = client.post(
        "/gnn-xai/prediction_and_attribution",
        json=malformed_payload,
    )
    assert response.status_code == 422


def test_e2e_concurrent_requests_succeed_safely(client: TestClient) -> None:
    async def _run_concurrent_test() -> None:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            responses = await asyncio.gather(
                async_client.post(
                    "/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"}
                ),
                async_client.post(
                    "/gnn-xai/prediction_and_attribution", json={"smiles": "CC(C)O"}
                ),
                async_client.post(
                    "/gnn-xai/prediction_and_attribution", json={"smiles": "c1ccccc1"}
                ),
            )

        for resp in responses:
            assert resp.status_code == 200
            GNNXAIInferenceResponse(**resp.json())

    asyncio.run(_run_concurrent_test())


def test_e2e_openapi_schema_and_docs_endpoints(client: TestClient) -> None:
    openapi_resp = client.get(app.openapi_url)
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert "/gnn-xai/prediction_and_attribution" in schema["paths"]

    post_responses = schema["paths"]["/gnn-xai/prediction_and_attribution"]["post"][
        "responses"
    ]
    assert "200" in post_responses
    assert "422" in post_responses
    assert "500" in post_responses
    assert "499" not in post_responses

    assert client.get("/gnn-xai/docs").status_code == 200
    assert client.get("/gnn-xai/redoc").status_code == 200
