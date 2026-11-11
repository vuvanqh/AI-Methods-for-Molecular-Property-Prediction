import asyncio
from functools import partial

import httpx

from src.clients import ChEMBLClient
from src.clients import client as client_module
from src.rag.augmentation import ChemblRetrieval


def activity(document: str | None = "DOC1", **fields) -> dict:
    return {
        "molecule_chembl_id": "CHEMBL1",
        "target_chembl_id": "CHEMBL240",
        "target_pref_name": "KCNH2",
        "target_organism": "Homo sapiens",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "14.0",
        "standard_units": "nM",
        "pchembl_value": "7.85",
        "activity_comment": None,
        "assay_chembl_id": "ASSAY1",
        "assay_description": "hERG patch clamp",
        "assay_type": "B",
        "document_chembl_id": document,
        **fields,
    }


class FakeClient:
    def __init__(self, activities=(), documents=(), smiles_hit="CHEMBL1"):
        self.activities = list(activities)
        self.documents = list(documents)
        self.smiles_hit = smiles_hit
        self.calls = []

    async def find_molecule_id(self, smiles):
        self.calls.append(("find", smiles))
        return self.smiles_hit

    async def get_activities(self, molecule_id):
        self.calls.append(("activities", molecule_id))
        return self.activities

    async def get_documents(self, document_ids):
        self.calls.append(("documents", document_ids))
        return self.documents


def test_activity_is_joined_with_document_and_converted():
    client = FakeClient(
        [activity()],
        [
            {
                "document_chembl_id": "DOC1",
                "title": "hERG paper",
                "abstract": "",
                "year": 2001,
                "doi": "10.1/x",
            }
        ],
    )

    [evidence] = asyncio.run(
        ChemblRetrieval(client).retrieve_molecular_evidence("CHEMBL1")
    )

    assert evidence.provider == "ChEMBL"
    assert evidence.target_name == "KCNH2"
    assert evidence.value == 14.0
    assert evidence.log_potency == 7.85
    assert evidence.document_title == "hERG paper"
    assert evidence.document_abstract is None
    assert evidence.document_url == "https://doi.org/10.1/x"


def test_activity_without_document_has_no_url():
    client = FakeClient([activity(document=None)])

    [evidence] = asyncio.run(
        ChemblRetrieval(client).retrieve_molecular_evidence("CHEMBL1")
    )

    assert evidence.document_url is None
    assert client.calls[-1] == ("documents", [])


def test_activities_survive_a_failed_document_lookup():
    client = FakeClient([activity()])
    client.get_documents = _raise_connection_error

    [evidence] = asyncio.run(
        ChemblRetrieval(client).retrieve_molecular_evidence("CHEMBL1")
    )

    assert evidence.value == 14.0
    assert evidence.document_title is None


async def _raise_connection_error(document_ids):
    raise ConnectionError("ChEMBL API unavailable")


def test_chembl_id_skips_smiles_lookup():
    client = FakeClient()

    asyncio.run(ChemblRetrieval(client).retrieve_molecular_evidence("CHEMBL25"))

    assert client.calls[0] == ("activities", "CHEMBL25")


def test_smiles_is_resolved_to_chembl_id():
    client = FakeClient(smiles_hit="CHEMBL25")

    asyncio.run(
        ChemblRetrieval(client).retrieve_molecular_evidence("CC(=O)Oc1ccccc1C(=O)O")
    )

    assert client.calls[:2] == [
        ("find", "CC(=O)Oc1ccccc1C(=O)O"),
        ("activities", "CHEMBL25"),
    ]


def test_smiles_unknown_to_chembl_returns_nothing():
    client = FakeClient(smiles_hit=None)

    assert asyncio.run(ChemblRetrieval(client).retrieve_molecular_evidence("CCO")) == []
    assert len(client.calls) == 1


def fake_api(monkeypatch, handler):
    monkeypatch.setattr(
        client_module.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    return ChEMBLClient("https://www.ebi.ac.uk/chembl/api/data")


def test_client_follows_every_page(monkeypatch):
    pages = {
        "0": {
            "activities": [{"id": 1}],
            "page_meta": {"next": "/chembl/api/data/activity.json?offset=1"},
        },
        "1": {"activities": [{"id": 2}], "page_meta": {"next": None}},
    }
    client = fake_api(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=pages[request.url.params.get("offset", "0")]
        ),
    )

    assert asyncio.run(client.get_activities("CHEMBL1")) == [{"id": 1}, {"id": 2}]


def test_client_splits_many_documents_into_batches(monkeypatch):
    requested = []

    def handler(request):
        ids = request.url.params["document_chembl_id__in"].split(",")
        requested.append(len(ids))
        return httpx.Response(200, json={"documents": ids, "page_meta": {"next": None}})

    client = fake_api(monkeypatch, handler)

    documents = asyncio.run(client.get_documents([f"DOC{i}" for i in range(120)]))

    assert requested == [50, 50, 20]
    assert len(documents) == 120
