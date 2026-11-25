import asyncio
import math
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from src.config import BINARY_PREDICTION_THRESHOLDS, TASKS, TASKS_NUMBER
from src.main import app
from src.model import ErrorResponse, GNNXAIInferenceResponse

ENDPOINT = "/gnn-xai/prediction_and_attribution"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Boot the production FastAPI app and Chemprop model once via `lifespan`."""
    with TestClient(app) as test_client:
        yield test_client


def _assert_valid_inference_response(
    raw_payload: dict[str, Any],
    smiles: str,
    *,
    expect_3d: bool = True,
    expect_nonzero_ig: bool = False,
) -> GNNXAIInferenceResponse:
    """Assert HTTP schema, MolBlock geometry, raw JSON string keys, and aligned scores."""
    validated = GNNXAIInferenceResponse(**raw_payload)

    parsed = Chem.MolFromSmiles(smiles)
    assert parsed is not None
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(parsed, canonical=True))
    assert canonical is not None
    n_heavy = canonical.GetNumAtoms()
    n_total = Chem.AddHs(canonical).GetNumAtoms()

    assert "V2000" in validated.molblock and "M  END" in validated.molblock
    mol = Chem.MolFromMolBlock(validated.molblock, removeHs=False)
    assert mol is not None and mol.GetNumAtoms() == n_total
    assert mol.GetNumConformers() == 1

    conf = mol.GetConformer()
    z_coords = [conf.GetAtomPosition(i).z for i in range(n_total)]
    assert conf.Is3D() is expect_3d
    if expect_3d:
        assert any(abs(z) > 1e-6 for z in z_coords)
    else:
        assert "2D" in validated.molblock and all(z == 0.0 for z in z_coords)

    assert [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_heavy)] == [
        atom.GetAtomicNum() for atom in canonical.GetAtoms()
    ]
    assert all(
        mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in range(n_heavy, n_total)
    )

    expected_tasks = set(TASKS)
    assert len(validated.predictions) == TASKS_NUMBER
    assert set(validated.predictions) == expected_tasks
    assert set(validated.probabilities) == expected_tasks
    assert set(validated.atomscores) == expected_tasks

    expected_raw_keys = [str(i) for i in range(n_total)]
    for task in TASKS:
        prob = validated.probabilities[task]
        pred = validated.predictions[task]
        assert math.isfinite(prob) and 0.0 <= prob <= 1.0
        assert isinstance(pred, bool) and pred is (
            prob > BINARY_PREDICTION_THRESHOLDS[task]
        )

        assert list(raw_payload["atomscores"][task].keys()) == expected_raw_keys
        scores = validated.atomscores[task]
        heavy = [scores[i] for i in range(n_heavy)]
        assert all(isinstance(s, float) and math.isfinite(s) for s in heavy)
        if expect_nonzero_ig:
            assert any(abs(s) > 1e-6 for s in heavy if s is not None)
        assert all(scores[i] is None for i in range(n_heavy, n_total))

    return validated


def test_inference_happy_path_for_drug_like_molecule(client: TestClient) -> None:
    """Aspirin returns 200 with 3D MolBlock, all 18 tasks, and non-zero IG scores."""
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    resp = client.post(ENDPOINT, json={"smiles": aspirin})
    assert resp.status_code == 200
    _assert_valid_inference_response(
        resp.json(), aspirin, expect_3d=True, expect_nonzero_ig=True
    )


def test_inference_degrades_gracefully_to_2d_molblock_for_strained_cage(
    client: TestClient,
) -> None:
    """A strained cage whose 3D embedding fails degrades cleanly to a 2D MolBlock."""
    strained_cage = "C1=CC2=CC=C1C=C2"
    resp = client.post(ENDPOINT, json={"smiles": strained_cage})
    assert resp.status_code == 200
    validated = _assert_valid_inference_response(
        resp.json(), strained_cage, expect_3d=False
    )
    assert len(validated.atomscores[TASKS[0]]) == 14


def test_service_rejects_malformed_and_invalid_inputs_and_recovers_cleanly(
    client: TestClient,
) -> None:
    """Malformed JSON, invalid SMILES, and zero-bond ions return 422 without wedging."""
    malformed = client.post(
        ENDPOINT, content=b"{invalid", headers={"Content-Type": "application/json"}
    )
    assert malformed.status_code == 422 and malformed.json().get("detail")

    for invalid_smiles in ("not_a_smiles", "[Na+]"):
        invalid = client.post(ENDPOINT, json={"smiles": invalid_smiles})
        assert invalid.status_code == 422
        assert ErrorResponse(**invalid.json()).message == "Invalid molecular input."

    recovered = client.post(ENDPOINT, json={"smiles": "CCO"})
    assert recovered.status_code == 200
    _assert_valid_inference_response(recovered.json(), "CCO")


def test_concurrent_http_requests_succeed_without_cross_contamination(
    client: TestClient,
) -> None:
    """Concurrent ASGI requests on the lifespan event loop match their own molecules."""
    molecules = ("CCO", "c1ccccc1")

    async def _send_concurrent() -> list[httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            posts = [ac.post(ENDPOINT, json={"smiles": s}) for s in molecules]
            return list(await asyncio.gather(*posts))

    assert client.portal is not None
    responses = client.portal.call(_send_concurrent)
    assert len(responses) == 2 and all(r.status_code == 200 for r in responses)

    validated = [
        _assert_valid_inference_response(r.json(), s)
        for s, r in zip(molecules, responses, strict=True)
    ]
    assert validated[0].molblock != validated[1].molblock
    assert validated[0].probabilities != validated[1].probabilities


def test_docker_healthcheck_docs_and_openapi_contract(client: TestClient) -> None:
    """Docker healthcheck `/gnn-xai/docs` and `/openapi.json` advertise the contract."""
    docs_resp = client.get("/gnn-xai/docs")
    openapi_resp = client.get("/openapi.json")

    assert docs_resp.status_code == 200
    assert "text/html" in docs_resp.headers["content-type"]

    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    post_responses = schema["paths"][ENDPOINT]["post"]["responses"]
    assert {"200", "422", "500"}.issubset(post_responses)
    schemas = schema["components"]["schemas"]
    assert {"InferenceRequest", "GNNXAIInferenceResponse", "ErrorResponse"}.issubset(
        schemas
    )
