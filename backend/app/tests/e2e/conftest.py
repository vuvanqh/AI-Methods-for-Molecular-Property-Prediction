import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sqlalchemy.orm import Session

from src.db.models import Molecule
from src.db.session import get_db
from src.descriptors import standardize_smiles
from src.main import app
from src.rag.embeddings import unimol_embeddings as unimol_module
from src.rag.embeddings.unimol_embeddings import UniMolEmbeddings
from tests.fakes import FakeExternalServices
from tests.pg_fixtures import seed_molecule

EMBEDDING_DIM = 512

ETHANOL_SMILES = "CCO"
PROPANOL_SMILES = "CCCO"
SALICYLIC_SMILES = "O=C(O)c1ccccc1O"

ETHANOL_CHEMBL_ID = "CHEMBL545"
PROPANOL_CHEMBL_ID = "CHEMBL14687"
SALICYLIC_CHEMBL_ID = "CHEMBL424"
ETHANOL_DOC_ID = "CHEMBL_DOC_ETHANOL"
ETHANOL_DOC_ABSTRACT = (
    "Microsomal assays measured ethanol inhibition against human CYP3A4."
)

DEFAULT_CORPUS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (ETHANOL_SMILES, (("ChEMBL", ETHANOL_CHEMBL_ID),)),
    (PROPANOL_SMILES, (("ChEMBL", PROPANOL_CHEMBL_ID), ("Unknown", "X1"))),
    (SALICYLIC_SMILES, (("ChEMBL", SALICYLIC_CHEMBL_ID), ("Unknown", "X2"))),
)

HttpMocker = Callable[[Callable[[httpx.Request], httpx.Response]], None]
RequestOverride = Callable[[httpx.Request], httpx.Response | None]


def make_unit_embedding(smiles: str) -> list[float]:
    """Return a deterministic, chemically meaningful 512-dim unit vector for a SMILES."""
    mol = Chem.MolFromSmiles(standardize_smiles(smiles))
    fp = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, includeChirality=True, fpSize=EMBEDDING_DIM
    ).GetFingerprint(mol)
    bits = [float(fp.GetBit(i)) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(b * b for b in bits))
    if norm == 0.0:
        bits[0] = 1.0
        return bits
    return [b / norm for b in bits]


class FakeUniMolRepr:
    """Deterministic stand-in for `UniMolRepr` avoiding heavy model weight downloads."""

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs

    def get_repr(self, smiles_batch: list[str]) -> list[list[float]]:
        return [make_unit_embedding(smiles) for smiles in smiles_batch]


class E2ETransportController:
    """Request recorder, fault-injection hook, and queried-molecule ChEMBL handler."""

    def __init__(self, services: FakeExternalServices) -> None:
        self.services = services
        self.override: RequestOverride | None = None
        self.requests: list[tuple[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.override is not None:
            response = self.override(request)
            if response is not None:
                return response
        queried_chembl = self._handle_queried_chembl(request)
        if queried_chembl is not None:
            return queried_chembl
        return self.services.handle(request)

    def _handle_queried_chembl(self, request: httpx.Request) -> httpx.Response | None:
        if request.method != "GET" or self.services.chembl_status != 200:
            return None
        path = request.url.path
        if (
            path == "/chembl/api/data/molecule.json"
            and request.url.params.get(
                "molecule_structures__canonical_smiles__flexmatch"
            )
            == ETHANOL_SMILES
        ):
            return httpx.Response(
                200,
                json={
                    "molecules": [{"molecule_chembl_id": ETHANOL_CHEMBL_ID}],
                    "page_meta": {"next": None},
                },
            )
        if (
            path == "/chembl/api/data/activity.json"
            and request.url.params.get("molecule_chembl_id") == ETHANOL_CHEMBL_ID
        ):
            return httpx.Response(
                200,
                json={
                    "activities": [
                        {
                            "molecule_chembl_id": ETHANOL_CHEMBL_ID,
                            "target_chembl_id": "CHEMBL340",
                            "target_pref_name": "CYP3A4",
                            "standard_type": "IC50",
                            "standard_relation": "=",
                            "standard_value": "500.0",
                            "standard_units": "nM",
                            "pchembl_value": "6.30",
                            "activity_comment": None,
                            "assay_description": "CYP3A4 inhibition assay",
                            "document_chembl_id": ETHANOL_DOC_ID,
                        }
                    ],
                    "page_meta": {"next": None},
                },
            )
        if path == "/chembl/api/data/document.json":
            requested_ids = request.url.params.get("document_chembl_id__in", "").split(
                ","
            )
            if ETHANOL_DOC_ID in requested_ids:
                return httpx.Response(
                    200,
                    json={
                        "documents": [
                            {
                                "document_chembl_id": ETHANOL_DOC_ID,
                                "title": "Ethanol CYP3A4 study",
                                "abstract": ETHANOL_DOC_ABSTRACT,
                                "journal": "Drug Metab. Dispos.",
                                "year": 2018,
                                "doi": "10.1/ethanol",
                            }
                        ],
                        "page_meta": {"next": None},
                    },
                )
        return None


def cancel_embedding_timer() -> None:
    """Cancel any lingering idle-unload timer left behind after UniMol shutdown."""
    retriever = getattr(app.state, "retriever", None)
    embedding_model = getattr(retriever, "embedding_model", None)
    timer = getattr(embedding_model, "_unload_timer", None)
    if timer is not None:
        timer.cancel()


def seed_corpus(
    session: Session,
    entries: Sequence[tuple[str, Sequence[tuple[str, str]]]] = DEFAULT_CORPUS,
) -> list[Molecule]:
    """Insert standardized molecules with Morgan fingerprints, sources, and 512-d unit embeddings."""
    return [
        seed_molecule(
            session,
            smiles,
            sources=sources,
            embeddings=[
                (
                    UniMolEmbeddings.name,
                    UniMolEmbeddings.version,
                    make_unit_embedding(smiles),
                )
            ],
        )
        for smiles, sources in entries
    ]


@pytest.fixture
def fake_unimol(
    monkeypatch: pytest.MonkeyPatch,
    isolated_rag_settings: None,
) -> type[FakeUniMolRepr]:
    """Swap heavy UniMol weights for `FakeUniMolRepr` on top of pinned RAG settings."""
    monkeypatch.setattr(unimol_module, "UniMolRepr", FakeUniMolRepr)
    return FakeUniMolRepr


@pytest.fixture
def e2e_transport(
    external_services: FakeExternalServices,
    mock_http: HttpMocker,
) -> E2ETransportController:
    """Route all outbound HTTP through `E2ETransportController`."""
    controller = E2ETransportController(external_services)
    mock_http(controller.handle)
    return controller


@contextmanager
def boot_e2e_client(
    db_dep: Callable[[], Iterator[Session]],
) -> Iterator[TestClient]:
    """Boot the real FastAPI `app` via `lifespan` wired to real PostgreSQL sessions."""
    app.dependency_overrides[get_db] = db_dep
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        cancel_embedding_timer()
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client(
    db_dependency: Callable[[], Iterator[Session]],
    clean_db: None,
    e2e_transport: E2ETransportController,
    fake_unimol: type[FakeUniMolRepr],
) -> Iterator[TestClient]:
    """Full-stack FastAPI `TestClient` backed by real PostgreSQL and faked external HTTP/UniMol."""
    with boot_e2e_client(db_dependency) as test_client:
        yield test_client
