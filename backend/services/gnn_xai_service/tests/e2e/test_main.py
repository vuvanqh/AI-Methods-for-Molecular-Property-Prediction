import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import TASKS
from src.main import app, lifespan


@pytest.mark.random_weights
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
def test_e2e_post_valid_smiles_returns_200_and_manages_lifespan(
    smiles: str,
    expected_heavy_atoms: int,
    expected_total_atoms: int,
) -> None:
    """
    End-to-end test exercising FastAPI lifespan startup, real HTTP POST request through
    the ASGI router, full GNN + IG + MolBlock execution, and lifespan shutdown cleanup.
    """
    with TestClient(app) as client:
        assert hasattr(app.state, "inference_service")
        assert hasattr(app.state, "inference_lock")

        response = client.post(
            "/gnn-xai/prediction_and_attribution",
            json={"smiles": smiles},
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"predictions", "probabilities", "atomscores", "molblock"}
        assert "V2000" in data["molblock"]
        assert "M  END" in data["molblock"]
        assert set(data["predictions"].keys()) == set(TASKS)
        assert set(data["probabilities"].keys()) == set(TASKS)
        assert set(data["atomscores"].keys()) == set(TASKS)
        for task in TASKS:
            task_scores = data["atomscores"][task]
            assert len(task_scores) == expected_total_atoms
            for i in range(expected_heavy_atoms):
                assert isinstance(task_scores[str(i)], float)
            for i in range(expected_heavy_atoms, expected_total_atoms):
                assert task_scores[str(i)] is None

    # Lifespan exit cleans up app.state
    assert not hasattr(app.state, "inference_service")
    assert not hasattr(app.state, "inference_lock")


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    ("bad_smiles", "expected_msg_substring"),
    [
        ("", "SMILES must not be empty"),
        ("   ", "SMILES must not be empty"),
        ("\t\n", "SMILES must not be empty"),
        ("invalid_smiles_string", "RDKit could not parse the provided SMILES"),
        ("C(C)(C)(C)(C)C", "RDKit could not parse the provided SMILES"),
    ],
)
def test_e2e_post_invalid_or_empty_smiles_returns_422_error_response(
    bad_smiles: str,
    expected_msg_substring: str,
) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/gnn-xai/prediction_and_attribution",
            json={"smiles": bad_smiles},
        )
        assert response.status_code == 422
        body = response.json()
        assert "message" in body
        assert expected_msg_substring in body["message"]


@pytest.mark.fast_rdkit
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
    malformed_payload: dict,
) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/gnn-xai/prediction_and_attribution",
            json=malformed_payload,
        )
        assert response.status_code == 422


@pytest.mark.random_weights
def test_e2e_concurrent_requests_are_serialized_by_inference_lock() -> None:
    """
    Verifies that concurrent requests to POST /gnn-xai/prediction_and_attribution
    queue cleanly behind `app.state.inference_lock` so at most 1 inference runs at a time.
    """
    active_count = 0
    max_concurrent_observed = 0

    async def _run_concurrent_test() -> None:
        nonlocal active_count, max_concurrent_observed
        async with lifespan(app):
            original_predict = app.state.inference_service.predict_and_explain

            def tracked_predict(smiles: str) -> dict:
                nonlocal active_count, max_concurrent_observed
                active_count += 1
                max_concurrent_observed = max(max_concurrent_observed, active_count)
                try:
                    time.sleep(0.02)
                    return original_predict(smiles)
                finally:
                    active_count -= 1

            app.state.inference_service.predict_and_explain = tracked_predict

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                responses = await asyncio.gather(
                    client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CCO"}),
                    client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "CC(C)O"}),
                    client.post("/gnn-xai/prediction_and_attribution", json={"smiles": "c1ccccc1"}),
                )

            for resp in responses:
                assert resp.status_code == 200

            assert max_concurrent_observed == 1

    asyncio.run(_run_concurrent_test())


@pytest.mark.fast_rdkit
def test_e2e_openapi_schema_and_docs_endpoints() -> None:
    with TestClient(app) as client:
        openapi_resp = client.get(app.openapi_url)
        assert openapi_resp.status_code == 200
        schema = openapi_resp.json()
        assert "/gnn-xai/prediction_and_attribution" in schema["paths"]

        post_responses = schema["paths"]["/gnn-xai/prediction_and_attribution"]["post"]["responses"]
        assert "200" in post_responses
        assert "422" in post_responses
        assert "500" in post_responses
        assert "499" not in post_responses

        assert client.get("/gnn-xai/docs").status_code == 200
        assert client.get("/gnn-xai/redoc").status_code == 200
