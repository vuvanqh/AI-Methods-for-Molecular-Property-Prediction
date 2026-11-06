from collections.abc import AsyncIterator
from typing import Any

from .source_data_models.chembl_source import ChEMBLSource


async def fetch_molecule_pages(
    source: ChEMBLSource,
    max_molecules: int = 10_000,
    page_size: int = 1000,
) -> AsyncIterator[list[dict[str, Any]]]:
    for offset in range(0, max_molecules, page_size):
        page = await source.fetch_molecules(
            limit=min(page_size, max_molecules - offset),
            offset=offset,
        )

        yield page["molecules"]

        if page["page_meta"]["next"] is None:
            return


if __name__ == "__main__":
    import asyncio

    class FakeSource:
        """3 molecules on the server, served 2 per page."""

        def __init__(self):
            self.calls = []

        async def fetch_molecules(self, limit, offset):
            self.calls.append((limit, offset))
            rows = [{"id": i} for i in range(3)][offset : offset + limit]
            more = offset + limit < 3
            return {"molecules": rows, "page_meta": {"next": "url" if more else None}}

    async def collect(source, **kwargs):
        return [page async for page in fetch_molecule_pages(source, **kwargs)]

    src = FakeSource()
    assert asyncio.run(collect(src, max_molecules=10, page_size=2)) == [
        [{"id": 0}, {"id": 1}],
        [{"id": 2}],
    ]
    assert src.calls == [
        (2, 0),
        (2, 2),
    ]  # stopped when ChEMBL said there is no next page

    src = FakeSource()
    assert asyncio.run(collect(src, max_molecules=3, page_size=2)) == [
        [{"id": 0}, {"id": 1}],
        [{"id": 2}],
    ]
    assert src.calls == [(2, 0), (1, 2)]  # last page shrunk to respect max_molecules
    print("fetch_molecules self-check passed")
