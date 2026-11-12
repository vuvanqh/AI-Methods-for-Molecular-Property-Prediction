import json
import uuid
from collections.abc import Sequence
from typing import Any
from unittest.mock import create_autospec

import httpx
from rdkit import Chem
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from src.db.models import Molecule, MoleculeSource
from src.rag.morgan_fingerprint_helper import (
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
    get_canonical_smiles,
)
from src.rag.result_types.molecular_evidence import MolecularEvidence

# Only 2 representative tasks are needed here: the orchestrator treats task names
# as dynamic dict keys rather than hardcoding all 18 GNN endpoints, so tests stay
# decoupled if GNN tasks are added or renamed.
TASKS = ("CYP3A4", "hERG_Karim")
LLM_SUMMARY = "LLM summary placeholder"

SAMPLE_EVIDENCE = MolecularEvidence(
    provider="ChEMBL",
    molecule_identifier="CHEMBL424",
    target_identifier="CHEMBL240",
    target_name="KCNH2",
    measurement="IC50",
    relation="=",
    value=14.0,
    units="nM",
    log_potency=7.85,
    outcome=None,
    assay_description="hERG patch clamp",
    document_title="hERG paper",
    document_abstract=None,
    document_year=2001,
    document_url="https://doi.org/10.1/x",
)


def gnn_payload(smiles: str) -> dict[str, Any]:
    """Build a valid GNN-XAI response payload, or raise ValueError on bad/unbonded SMILES."""
    canonical = get_canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if mol.GetNumBonds() == 0:
        raise ValueError("Invalid or unbonded SMILES")
    heavy = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol)
    scores = {
        str(i): (0.1 * (i + 1) if i < heavy else None)
        for i in range(mol_h.GetNumAtoms())
    }
    return {
        "predictions": {"CYP3A4": False, "hERG_Karim": True},
        "probabilities": {"CYP3A4": 0.2, "hERG_Karim": 0.85},
        "atomscores": {t: dict(scores) for t in TASKS},
        "molblock": Chem.MolToMolBlock(mol_h),
    }


def corpus_mol(smiles: str, sources: Sequence[tuple[str, str]] = ()) -> Molecule:
    """Create an in-memory SQLAlchemy `Molecule` ORM row with `MoleculeSource` links."""
    canonical = get_canonical_smiles(smiles)
    mol_id = uuid.uuid5(uuid.NAMESPACE_OID, canonical)
    return Molecule(
        id=mol_id,
        standardized_smiles=canonical,
        morgan_fingerprint=fingerprint_to_bitstring(
            generate_morgan_fingerprint(canonical)
        ),
        molecule_sources=[
            MoleculeSource(provider=p, molecule_id=mol_id, source_identifier=i)
            for p, i in sources
        ],
    )


def fake_db() -> Session:
    """Zero-arg `get_db` override that compiles Postgres SQL and filters excluded SMILES."""
    corpus = [
        (corpus_mol("CCO", [("ChEMBL", "CHEMBL545")]), 1.0),
        (
            corpus_mol("O=C(O)c1ccccc1O", [("ChEMBL", "CHEMBL424"), ("Unknown", "X1")]),
            0.85,
        ),
    ]
    db = create_autospec(Session, instance=True)

    def _execute(stmt):
        compiled = stmt.compile(dialect=postgresql.dialect())
        excluded = {
            val
            for key, val in compiled.params.items()
            if key.startswith("standardized_smiles") and isinstance(val, str)
        }
        rows = [
            (mol, score)
            for mol, score in corpus
            if mol.standardized_smiles not in excluded
        ]
        result = db.execute.return_value
        result.all.return_value = rows
        return result

    db.execute.side_effect = _execute
    return db


class FakeExternalServices:
    """In-memory fake for external HTTP APIs (GNN service, OpenRouter LLM, ChEMBL)
    so E2E tests run offline without real network calls."""

    def __init__(self) -> None:
        self.gnn_status = 200
        self.llm_status = 200
        self.chembl_status = 200
        self.last_llm_prompt = ""

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        # 1. GNN-XAI microservice
        if request.method == "POST" and path == "/gnn-xai/prediction_and_attribution":
            if self.gnn_status != 200:
                return httpx.Response(self.gnn_status, json={"message": "GNN error"})
            try:
                body = json.loads(request.content)
                if not isinstance(body, dict) or not isinstance(
                    body.get("smiles"), str
                ):
                    return httpx.Response(422, json={"message": "Missing smiles"})
                return httpx.Response(200, json=gnn_payload(body["smiles"]))
            except ValueError as exc:
                return httpx.Response(422, json={"message": str(exc)})

        # 2. OpenRouter LLM API
        if request.method == "POST" and path == "/api/v1/chat/completions":
            if self.llm_status != 200:
                return httpx.Response(self.llm_status, json={"error": "LLM error"})
            payload = json.loads(request.content)
            self.last_llm_prompt = payload["messages"][-1]["content"]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": LLM_SUMMARY}}]}
            )

        # 3. ChEMBL REST API (each resource returns only its own collection key)
        if request.method == "GET" and path.startswith("/chembl/api/data/"):
            if self.chembl_status != 200:
                return httpx.Response(
                    self.chembl_status, json={"message": "ChEMBL error"}
                )
            if path == "/chembl/api/data/molecule.json":
                return httpx.Response(
                    200, json={"molecules": [], "page_meta": {"next": None}}
                )
            if path == "/chembl/api/data/activity.json":
                mol_id = request.url.params.get("molecule_chembl_id")
                activities = (
                    [
                        {
                            "molecule_chembl_id": "CHEMBL424",
                            "target_chembl_id": "CHEMBL240",
                            "target_pref_name": "KCNH2",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "14.0",
                            "standard_units": "nM",
                            "pchembl_value": "7.85",
                            "activity_comment": None,
                            "assay_description": "hERG patch clamp",
                            "document_chembl_id": "CHEMBL_DOC1",
                        }
                    ]
                    if mol_id == "CHEMBL424"
                    else []
                )
                return httpx.Response(
                    200, json={"activities": activities, "page_meta": {"next": None}}
                )
            if path == "/chembl/api/data/document.json":
                requested_ids = request.url.params.get(
                    "document_chembl_id__in", ""
                ).split(",")
                documents = (
                    [
                        {
                            "document_chembl_id": "CHEMBL_DOC1",
                            "title": "hERG paper",
                            "abstract": None,
                            "journal": "J. Med. Chem.",
                            "year": 2001,
                            "doi": "10.1/x",
                        }
                    ]
                    if "CHEMBL_DOC1" in requested_ids
                    else []
                )
                return httpx.Response(
                    200, json={"documents": documents, "page_meta": {"next": None}}
                )

        return httpx.Response(
            404, json={"message": f"Unexpected request: {request.method} {request.url}"}
        )
