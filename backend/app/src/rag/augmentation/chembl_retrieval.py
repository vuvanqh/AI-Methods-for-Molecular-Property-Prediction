import re
from ...clients.chembl_client import ChEMBLClient
from ..config import CHEMBL_DOCUMENT_URL
from ..result_types.molecular_evidence import MolecularEvidence
from .client_retrieval_base import ClientStrategyBase


def _to_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _document_url(document: dict) -> str | None:
    if document.get("doi"):
        return f"https://doi.org/{document['doi']}"
    if document.get("document_chembl_id"):
        return CHEMBL_DOCUMENT_URL.format(document["document_chembl_id"])
    return None


class ChemblRetrieval(ClientStrategyBase):
    provider = "ChEMBL"
    CHEMBL_ID = re.compile(r"CHEMBL\d+")

    def __init__(self, client: ChEMBLClient):
        self.client = client

    async def retrieve_molecular_evidence(self, query: str) -> list[MolecularEvidence]:
        """
        All ChEMBL activities of the molecule given as a ChEMBL id or SMILES,
        joined with the publication each one comes from.
        """
        if self.CHEMBL_ID.fullmatch(query):
            molecule_id = query
        else:
            molecule_id = await self.client.find_molecule_id(query)
        if molecule_id is None:
            return []

        activities = await self.client.get_activities(molecule_id)

        # sorting just for debugging; the order is not important to the caller
        document_ids = sorted(
            {
                activity["document_chembl_id"]
                for activity in activities
                if activity["document_chembl_id"]
            }
        )

        try:
            documents = {
                document["document_chembl_id"]: document
                for document in await self.client.get_documents(document_ids)
            }
        except ConnectionError:
            # the measurements are the evidence => losing their citations must not lose them too
            documents = {}

        evidence = []
        for activity in activities:
            document = documents.get(activity["document_chembl_id"], {})
            evidence.append(
                MolecularEvidence(
                    provider=self.provider,
                    molecule_identifier=activity["molecule_chembl_id"],
                    target_identifier=activity["target_chembl_id"],
                    target_name=activity["target_pref_name"],
                    measurement=activity["standard_type"],
                    relation=activity["standard_relation"],
                    value=_to_float(activity["standard_value"]),
                    units=activity["standard_units"],
                    log_potency=_to_float(activity["pchembl_value"]),
                    outcome=activity["activity_comment"],
                    assay_description=activity["assay_description"],
                    document_title=document.get("title"),
                    # old publications come back with "" instead of null
                    document_abstract=document.get("abstract") or None,
                    document_year=document.get("year"),
                    document_url=_document_url(document),
                )
            )
        return evidence
