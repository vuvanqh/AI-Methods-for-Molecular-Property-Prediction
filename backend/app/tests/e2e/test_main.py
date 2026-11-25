from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis
from src.main import app
from src.model import ErrorResponse, PredictionResponse
from tests.e2e.conftest import (
    ETHANOL_DOC_ABSTRACT,
    ETHANOL_SMILES,
    PROPANOL_CHEMBL_ID,
    PROPANOL_SMILES,
    SALICYLIC_CHEMBL_ID,
    SALICYLIC_SMILES,
    E2ETransportController,
    boot_e2e_client,
    seed_corpus,
)
from tests.fakes import LLM_SUMMARY, TASKS, FakeExternalServices

pytestmark = pytest.mark.integration

PROPANOL_CHEMBL_URL = (
    f"https://www.ebi.ac.uk/chembl/explore/compound/{PROPANOL_CHEMBL_ID}"
)
SALICYLIC_CHEMBL_URL = (
    f"https://www.ebi.ac.uk/chembl/explore/compound/{SALICYLIC_CHEMBL_ID}"
)


def _assert_ethanol_prediction_intact(parsed: PredictionResponse) -> None:
    """Assert descriptors, GNN-XAI payload, and real-Postgres ranked neighbours for ethanol."""
    assert parsed.molecule.smiles == ETHANOL_SMILES
    assert parsed.molecule.formula == "C2H6O"
    assert parsed.molecule.molar_mass == pytest.approx(46.07, abs=0.05)
    assert parsed.molecule.atom_count == 9
    assert parsed.molecule.bond_count == 8

    xai = parsed.properties_xai_data
    assert set(xai.predictions) == set(TASKS)
    assert xai.predictions == {"CYP3A4": False, "hERG_Karim": True}
    assert xai.probabilities == {
        "CYP3A4": pytest.approx(0.2),
        "hERG_Karim": pytest.approx(0.85),
    }
    assert set(xai.atomscores) == set(TASKS)
    for task in TASKS:
        scores = xai.atomscores[task]
        assert len(scores) == parsed.molecule.atom_count
        assert sum(score is not None for score in scores.values()) == 3
        assert sum(score is None for score in scores.values()) == 6
    assert "V2000" in xai.molblock

    assert [c.smiles for c in parsed.similar_compounds] == [
        PROPANOL_SMILES,
        SALICYLIC_SMILES,
    ]
    propanol, salicylic = parsed.similar_compounds
    assert propanol.molecular_formula == "C3H8O"
    assert salicylic.molecular_formula == "C7H6O3"
    assert 0.0 < salicylic.similarity_score < propanol.similarity_score < 1.0
    assert propanol.sources == [PROPANOL_CHEMBL_URL]
    assert salicylic.sources == [SALICYLIC_CHEMBL_URL]


def _assert_single_analysis(
    db_session: Session,
    *,
    status: AnalysisStatus,
    results: dict | None,
) -> Analysis:
    """Assert exactly one ethanol `Analysis` row was persisted with expected status and payload."""
    analyses = db_session.scalars(select(Analysis)).all()
    assert len(analyses) == 1
    row = analyses[0]
    assert row.standardized_input_smiles == ETHANOL_SMILES
    assert row.status == status
    assert row.results == results
    if results is None:
        assert db_session.scalar(select(Analysis.results.is_(None))) is True
    return row


def test_openapi_and_documentation_endpoints_expose_prediction_contract(
    client: TestClient,
) -> None:
    """OpenAPI schema, Swagger UI, and ReDoc advertise POST /api/predict-properties."""
    openapi_resp = client.get("/openapi.json")
    docs_resp = client.get("/api/docs")
    redoc_resp = client.get("/api/redoc")

    assert openapi_resp.status_code == 200
    assert docs_resp.status_code == 200
    assert "text/html" in docs_resp.headers["content-type"]
    assert redoc_resp.status_code == 200
    assert "text/html" in redoc_resp.headers["content-type"]

    post_op = openapi_resp.json()["paths"]["/api/predict-properties"]["post"]
    assert {"200", "422", "503"}.issubset(post_op["responses"])


def test_lifespan_closes_http_clients_and_unloads_embeddings_on_shutdown(
    db_dependency: Callable[[], Iterator[Session]],
    db_session: Session,
    e2e_transport: E2ETransportController,
    fake_unimol: type[object],
) -> None:
    """App lifespan opens HTTP pools and lazy embeddings during traffic and releases them on shutdown."""
    seed_corpus(db_session)

    with boot_e2e_client(db_dependency) as test_client:
        resp = test_client.post(
            "/api/predict-properties", json={"smiles": ETHANOL_SMILES}
        )

        assert resp.status_code == 200
        assert not app.state.gnn_client.http.is_closed
        assert not app.state.llm_client.http.is_closed
        assert all(
            not source.client.http.is_closed for source in app.state.evidence_sources
        )
        assert isinstance(app.state.retriever.embedding_model.model, fake_unimol)

    assert app.state.gnn_client.http.is_closed
    assert app.state.llm_client.http.is_closed
    assert all(source.client.http.is_closed for source in app.state.evidence_sources)
    assert app.state.retriever.embedding_model.model is None


def test_predict_properties_happy_path_runs_full_pipeline_and_persists_completed_analysis(
    client: TestClient,
    db_session: Session,
    external_services: FakeExternalServices,
) -> None:
    """Full pipeline executes real Postgres hybrid retrieval, ChEMBL evidence, LLM prompt, and DB logging."""
    seed_corpus(db_session)

    resp = client.post("/api/predict-properties", json={"smiles": ETHANOL_SMILES})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    parsed = PredictionResponse.model_validate(body)
    _assert_ethanol_prediction_intact(parsed)
    assert parsed.llm_summary == LLM_SUMMARY

    prompt = external_services.last_llm_prompt
    assert ETHANOL_SMILES in prompt
    assert "C2H6O" in prompt
    assert "hERG_Karim" in prompt
    assert "0.85" in prompt
    assert "O2" in prompt
    assert "CYP3A4" in prompt
    assert "500 nM" in prompt
    assert "Ethanol CYP3A4 study" in prompt
    assert ETHANOL_DOC_ABSTRACT in prompt
    assert SALICYLIC_SMILES in prompt
    assert "14 nM" in prompt
    assert "hERG paper" in prompt

    row = _assert_single_analysis(
        db_session, status=AnalysisStatus.COMPLETED, results=body
    )
    assert row.config["input_smiles"] == ETHANOL_SMILES
    assert row.config["retrieval_type"] == "fusion"


def test_invalid_smiles_fails_fast_and_unbonded_ion_stops_before_llm_with_no_db_writes(
    client: TestClient,
    db_session: Session,
    e2e_transport: E2ETransportController,
) -> None:
    """Invalid SMILES rejects before any HTTP call; unbonded ions stop at GNN 422 without LLM or DB writes."""
    invalid_resp = client.post(
        "/api/predict-properties", json={"smiles": "not_a_smiles"}
    )

    assert invalid_resp.status_code == 422
    assert (
        ErrorResponse.model_validate(invalid_resp.json()).message
        == "The molecule could not be processed."
    )
    assert e2e_transport.requests == []
    assert db_session.scalars(select(Analysis)).all() == []

    ion_resp = client.post("/api/predict-properties", json={"smiles": "[Na+]"})

    assert ion_resp.status_code == 422
    assert (
        ErrorResponse.model_validate(ion_resp.json()).message
        == "The molecule could not be processed."
    )
    assert ("POST", "/gnn-xai/prediction_and_attribution") in e2e_transport.requests
    assert ("POST", "/api/v1/chat/completions") not in e2e_transport.requests
    assert db_session.scalars(select(Analysis)).all() == []


def test_gnn_outage_returns_503_and_persists_failed_analysis(
    client: TestClient,
    db_session: Session,
    external_services: FakeExternalServices,
    e2e_transport: E2ETransportController,
) -> None:
    """GNN 503 returns ErrorResponse 503, skips LLM, and logs a FAILED analysis with SQL NULL results."""
    seed_corpus(db_session)
    external_services.gnn_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": ETHANOL_SMILES})

    assert resp.status_code == 503
    assert (
        ErrorResponse.model_validate(resp.json()).message == "GNN service unavailable."
    )
    assert ("POST", "/api/v1/chat/completions") not in e2e_transport.requests

    _assert_single_analysis(db_session, status=AnalysisStatus.FAILED, results=None)


def test_llm_outage_degrades_gracefully_with_full_predictions_and_completed_analysis(
    client: TestClient,
    db_session: Session,
    external_services: FakeExternalServices,
) -> None:
    """LLM 503 still returns 200 with full predictions, empty summary, and COMPLETED DB log."""
    seed_corpus(db_session)
    external_services.llm_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": ETHANOL_SMILES})

    assert resp.status_code == 200
    body = resp.json()
    parsed = PredictionResponse.model_validate(body)
    _assert_ethanol_prediction_intact(parsed)
    assert parsed.llm_summary == ""

    _assert_single_analysis(db_session, status=AnalysisStatus.COMPLETED, results=body)


def test_chembl_outage_degrades_gracefully_and_flags_unavailability_in_prompt(
    client: TestClient,
    db_session: Session,
    external_services: FakeExternalServices,
) -> None:
    """ChEMBL 503 preserves predictions, links, summary, and flags unreachable source instead of absent data."""
    seed_corpus(db_session)
    external_services.chembl_status = 503

    resp = client.post("/api/predict-properties", json={"smiles": ETHANOL_SMILES})

    assert resp.status_code == 200
    body = resp.json()
    parsed = PredictionResponse.model_validate(body)
    _assert_ethanol_prediction_intact(parsed)
    assert parsed.llm_summary == LLM_SUMMARY

    prompt = external_services.last_llm_prompt
    assert "ChEMBL could not be reached" in prompt
    assert "no measured data found" not in prompt.lower()
    assert "14 nM" not in prompt
    assert "500 nM" not in prompt
    assert "hERG paper" not in prompt
    assert "Ethanol CYP3A4 study" not in prompt

    _assert_single_analysis(db_session, status=AnalysisStatus.COMPLETED, results=body)
