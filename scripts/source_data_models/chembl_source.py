from .data_source import DataSource


class ChEMBLSource(DataSource):
    def __init__(self):
        super().__init__("https://www.ebi.ac.uk/chembl/api/data")

    async def fetch_molecules(
        self,
        limit: int = 1000,
        offset: int = 0,
        params: dict | None = None,
    ):
        return await self.fetch_data(
            "molecule.json",
            params={
                "limit": limit,
                "offset": offset,
                "molecule_type": "Small molecule",
                "molecule_structures__isnull": "false",
                "only": "molecule_chembl_id,molecule_structures",
                **(params or {}),
            },
        )

    async def fetch_molecule(self, molecule_id: str, params: dict | None = None):
        return await self.fetch_data(f"molecule/{molecule_id}.json", params=params)

    async def fetch_target(self, target_id: str):
        return await self.fetch_data(f"target/{target_id}.json")
