from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.db.session import get_db
from src.main import app
from src.model import ErrorResponse, PredictionResponse
from src.rag.embeddings import unimol_embeddings as unimol_module
from tests.fakes import LLM_SUMMARY, TASKS, FakeExternalServices, fake_db


def _cancel_embedding_timer() -> None:
    timer = app.state.retriever.embedding_model._unload_timer
    if timer is not None:
        timer.cancel()


@pytest.fixture
def client(
    external_services: FakeExternalServices, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Boot the FastAPI app via `lifespan` with external HTTP, DB, and UniMol weights faked."""
    fake_unimol_model = MagicMock()
    fake_unimol_model.get_repr.side_effect = lambda batch: [[0.1] * 512 for _ in batch]
    monkeypatch.setattr(
        unimol_module, "UniMolRepr", MagicMock(return_value=fake_unimol_model)
    )

    app.dependency_overrides[get_db] = fake_db
    try:
        with TestClient(app) as test_client:
            try:
                yield test_client
            finally:
                _cancel_embedding_timer()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_e2e_lifespan_closes_http_clients_and_unloads_embeddings_on_shutdown(
    external_services: FakeExternalServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_unimol_model = MagicMock()
    fake_unimol_model.get_repr.side_effect = lambda batch: [[0.1] * 512 for _ in batch]
    monkeypatch.setattr(
        unimol_module, "UniMolRepr", MagicMock(return_value=fake_unimol_model)
    )

    app.dependency_overrides[get_db] = fake_db
    try:
        with TestClient(app) as test_client:
            assert (
                test_client.post(
                    "/api/predict-properties", json={"smiles": "CCO"}
                ).status_code
                == 200
            )
            assert app.state.retriever.embedding_model.model is not None
            _cancel_embedding_timer()

        assert app.state.gnn_client.http.is_closed
        assert app.state.llm_client.http.is_closed
        assert app.state.evidence_sources[0].client.http.is_closed
        assert app.state.retriever.embedding_model.model is None
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_e2e_valid_smiles_returns_200_and_expected_response(
    client: TestClient, external_services: FakeExternalServices
) -> None:
    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 200
    parsed = PredictionResponse(**resp.json())
    assert parsed.molecule.smiles == "CCO"
    assert parsed.molecule.atom_count == 9
    assert set(parsed.properties_xai_data.predictions) == set(TASKS)
    assert "V2000" in parsed.properties_xai_data.molblock
    assert len(parsed.similar_compounds) == 1
    assert parsed.similar_compounds[0].smiles == "O=C(O)c1ccccc1O"
    assert parsed.similar_compounds[0].sources == [
        "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424"
    ]
    assert parsed.llm_summary == LLM_SUMMARY
    assert "KCNH2 (ChEMBL): IC50 = 14 nM" in external_services.last_llm_prompt
    assert "hERG paper (2001)" in external_services.last_llm_prompt


def test_e2e_equivalent_smiles_return_same_response(client: TestClient) -> None:
    resp_a = client.post("/api/predict-properties", json={"smiles": "CCO"})
    resp_b = client.post("/api/predict-properties", json={"smiles": "OCC"})

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_a.json() == resp_b.json()


def test_e2e_missing_smiles_field_returns_422(client: TestClient) -> None:
    assert client.post("/api/predict-properties", json={}).status_code == 422


@pytest.mark.parametrize("bad_smiles", ["", "invalid", "[Na+]"])
def test_e2e_unparseable_or_unbonded_smiles_returns_422_error_response(
    client: TestClient, bad_smiles: str
) -> None:
    resp = client.post("/api/predict-properties", json={"smiles": bad_smiles})

    assert resp.status_code == 422
    assert (
        ErrorResponse(**resp.json()).message == "The molecule could not be processed."
    )


def test_e2e_gnn_outage_returns_503(
    client: TestClient, external_services: FakeExternalServices
) -> None:
    external_services.gnn_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 503
    assert ErrorResponse(**resp.json()).message == "GNN service unavailable."


def test_e2e_llm_outage_degrades_to_empty_summary(
    client: TestClient, external_services: FakeExternalServices
) -> None:
    external_services.llm_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 200
    assert PredictionResponse(**resp.json()).llm_summary == ""


def test_e2e_chembl_outage_still_returns_200(
    client: TestClient, external_services: FakeExternalServices
) -> None:
    external_services.chembl_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": "CCO"})

    assert resp.status_code == 200
    assert PredictionResponse(**resp.json()).llm_summary == LLM_SUMMARY
    assert "ChEMBL could not be reached" in external_services.last_llm_prompt
