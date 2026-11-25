import asyncio
from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from src.rag.augmentation.client_retrieval_base import ClientStrategyBase
from src.rag.result_types.molecular_evidence import MolecularEvidence
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from src.rag.retrieval.retrieval_base import Retrieval
from tests.fakes import SAMPLE_EVIDENCE


def run_client[T, R](client: T, action: Callable[[T], Any]) -> R:
    async def _execute():
        async with aclosing(client):
            return await action(client)

    return asyncio.run(_execute())


def make_evidence(**overrides: Any) -> MolecularEvidence:
    return replace(SAMPLE_EVIDENCE, **overrides)


def make_molecule(
    mol_id: str = "mol-1",
    smiles: str = "CCO",
    sources: Sequence[tuple[str, str]] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=mol_id,
        standardized_smiles=smiles,
        molecule_sources=[
            SimpleNamespace(provider=p, source_identifier=i) for p, i in sources
        ],
    )


def make_neighbour(
    smiles: str = "CCN",
    sources: Sequence[tuple[str, str]] = (),
    score: float = 0.75,
    mol_id: str = "neighbour-1",
) -> RetrievedMolecule:
    return RetrievedMolecule(make_molecule(mol_id, smiles, sources), score)


class FakeEvidenceSource(ClientStrategyBase):
    def __init__(
        self,
        provider: str = "ChEMBL",
        *,
        fail_on: set[str] | None = None,
        error: Exception | None = None,
        evidence_by_query: dict[str, list[MolecularEvidence]] | None = None,
    ) -> None:
        self.provider = provider
        self.fail_on = fail_on or set()
        self.error = error
        self.evidence_by_query = evidence_by_query
        self.queries: list[str] = []

    async def retrieve_molecular_evidence(self, query: str) -> list[MolecularEvidence]:
        self.queries.append(query)
        if self.error is not None and (not self.fail_on or query in self.fail_on):
            raise self.error
        if query in self.fail_on:
            raise ConnectionError(f"{self.provider} unavailable for {query}")
        if self.evidence_by_query is not None:
            return list(self.evidence_by_query.get(query, []))
        return [make_evidence(provider=self.provider, molecule_identifier=query)]


class FakeRetrieval(Retrieval):
    def __init__(self, results: list[RetrievedMolecule]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, smiles: str, top_k: int) -> list[RetrievedMolecule]:
        self.calls.append((smiles, top_k))
        return self.results[:top_k] if top_k > 0 else []
