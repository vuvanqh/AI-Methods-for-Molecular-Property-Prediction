import httpx

from ..core.config import settings
from .client import Client

ACTIVITY_FIELDS = (
    "molecule_chembl_id,target_chembl_id,target_pref_name,"
    "standard_type,standard_relation,standard_value,standard_units,pchembl_value,"
    "activity_comment,assay_description,document_chembl_id"
)
DOCUMENT_FIELDS = "document_chembl_id,title,abstract,journal,year,doi"
# the API silently caps larger limits at 1000
PAGE_LIMIT = 1000
# chembl allows only a few hundred document ids per request, so we batch them to avoid 400 errors (url length limit)
DOCUMENT_BATCH_SIZE = 50


class ChEMBLClient(Client):
    def __init__(self, base_url: str = settings.chembl_api_url):
        # trailing slash so url.join("molecule.json") appends instead of replacing the last path segment
        super().__init__(base_url.rstrip("/") + "/", settings.chembl_timeout_seconds)

    # Why not just query for molecule details and act on 404? Because the API returns 404 for a molecule with no activities, which is not the same as "not found",
    # and we need the felxmatch as chembl might use different toolkit for smiles canonization than we did so it could introduce some discrepancies
    # NOTE: we only use this for the molecule the user submitted as we filter it out in the sql statement, not for the retrieved molecules.
    async def find_molecule_id(self, smiles: str) -> str | None:
        """
        ChEMBL id of the molecule with this structure, or None when ChEMBL does not have it.
        """
        molecules = await self._get_all(
            "molecule",
            "molecules",
            {
                "molecule_structures__canonical_smiles__flexmatch": smiles,
                "only": "molecule_chembl_id",
            },
        )
        return molecules[0]["molecule_chembl_id"] if molecules else None

    async def get_activities(self, molecule_id: str) -> list[dict]:
        """
        Every measured activity of the molecule on human targets, without records
        ChEMBL flags as invalid or duplicated.
        """
        return await self._get_all(
            "activity",
            "activities",
            {
                "molecule_chembl_id": molecule_id,
                "target_organism": "Homo sapiens",
                "data_validity_comment__isnull": "true",
                "potential_duplicate": 0,
                "only": ACTIVITY_FIELDS,
            },
        )

    async def get_documents(self, document_ids: list[str]) -> list[dict]:
        documents = []
        for start in range(0, len(document_ids), DOCUMENT_BATCH_SIZE):
            batch = document_ids[start : start + DOCUMENT_BATCH_SIZE]
            documents += await self._get_all(
                "document",
                "documents",
                {"document_chembl_id__in": ",".join(batch), "only": DOCUMENT_FIELDS},
            )
        return documents

    async def _get_all(self, resource: str, key: str, params: dict) -> list[dict]:
        """
        Collect every page of a list endpoint by following page_meta.next.
        """
        url = self.url.join(f"{resource}.json")
        query: dict | None = {**params, "limit": PAGE_LIMIT}
        rows = []
        try:
            while url is not None:
                response = await self.http.get(url, params=query)
                if response.is_error:
                    raise ConnectionError(
                        f"ChEMBL returned {response.status_code} for {response.url}"
                    )
                page = response.json()
                rows.extend(page[key])
                next_page = page["page_meta"]["next"]
                url = self.url.join(next_page) if next_page else None
                # None keeps the query string of next; {} would replace it with nothing
                query = None
        except httpx.RequestError as exc:
            raise ConnectionError("ChEMBL API unavailable") from exc
        return rows
