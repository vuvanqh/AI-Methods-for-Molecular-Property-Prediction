import ssl

import httpx


class Client:
    def __init__(self, base_url: str, timeout: float):
        self.url = httpx.URL(base_url)
        # one connection pool for the app's lifetime, so later calls skip the TCP/TLS handshake
        # trying to mimic the singleton behaviour liike in asp.net core, but httpx doesn't have a built-in singleton client, so we just keep one instance around
        self.http = httpx.AsyncClient(
            timeout=timeout,
            verify=ssl.create_default_context(),
        )

    async def aclose(self):
        await self.http.aclose()
