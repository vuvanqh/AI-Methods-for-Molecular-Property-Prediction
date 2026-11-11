import httpx
from ..core.config import settings
from .client import Client


class GNNClient(Client):
    def __init__(self, base_url: str = settings.gnn_service_url):
        super().__init__(
            f"{base_url.rstrip('/')}/prediction_and_attribution",
            timeout=settings.gnn_timeout_seconds,
        )

    async def get_prediction_and_attribution(self, smiles: str) -> dict:
        try:
            response = await self.http.post(self.url, json={"smiles": smiles})
        except httpx.RequestError as exc:
            raise ConnectionError("GNN service unavailable") from exc

        if response.status_code == 422:
            raise ValueError("The molecule could not be processed.")
        if response.is_error:
            # any other upstream failure (wrong URL, crash) is reported as "unavailable" (503), not a 500 traceback
            raise ConnectionError(
                f"GNN service returned {response.status_code} for {self.url}"
            )

        return response.json()
