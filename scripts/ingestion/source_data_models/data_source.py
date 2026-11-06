import ssl

import httpx


class DataSource:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def fetch_data(self, url: str, params: dict | None = None):
        async with httpx.AsyncClient(
            timeout=60, verify=ssl.create_default_context()
        ) as client:
            response = await client.get(f"{self.base_url}/{url}", params=params)
            response.raise_for_status()
            return response.json()
