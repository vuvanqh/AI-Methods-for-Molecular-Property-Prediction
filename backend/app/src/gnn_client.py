import httpx
from .core.config import settings


class GNNClient:
    def __init__(self, base_url: str = settings.gnn_service_url):
        self.url = f"{base_url.rstrip('/')}/prediction_and_attribution"

    async def get_prediction_and_attribution(self, smiles: str) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, json={"smiles": smiles})
        except httpx.RequestError as exc:
            raise ConnectionError("GNN service unavailable") from exc

        if response.status_code == 422:
            raise ValueError("The molecule could not be processed.")
        response.raise_for_status()

        return response.json()


gnn_client = GNNClient()
