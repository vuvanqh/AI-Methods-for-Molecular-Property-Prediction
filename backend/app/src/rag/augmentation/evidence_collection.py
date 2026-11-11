import asyncio
import logging
from collections.abc import Coroutine

from ...db.models import Molecule
from ..result_types.molecular_evidence import MolecularEvidence
from ..result_types.retrieved_molecules import RetrievedMolecule
from .client_retrieval_base import ClientStrategyBase

logger = logging.getLogger(__name__)

EvidenceCall = Coroutine[None, None, list[MolecularEvidence]]


async def _retrieve(
    strategy: ClientStrategyBase, query: str, unavailable: set[str]
) -> list[MolecularEvidence]:
    try:
        return await strategy.retrieve_molecular_evidence(query)
    except ConnectionError as exc:
        # evidence only enriches the summary, so a source being down must not fail the prediction
        logger.warning("%s evidence skipped for %s: %s", strategy.provider, query, exc)
        # the prompt needs this to say the data is unknown rather than absent
        unavailable.add(strategy.provider)
        return []


async def _flatten(calls: list[EvidenceCall]) -> list[MolecularEvidence]:
    return [evidence for result in await asyncio.gather(*calls) for evidence in result]


def _calls_for_stored(
    strategies: list[ClientStrategyBase], molecule: Molecule, unavailable: set[str]
) -> list[EvidenceCall]:
    """
    One call per provider the molecule was ingested from, each asked with that
    provider's own identifier. A provider no strategy handles is skipped.
    """
    return [
        _retrieve(strategy, source.source_identifier, unavailable)
        for source in molecule.molecule_sources
        for strategy in strategies
        if strategy.provider == source.provider
    ]


async def gather_evidence(
    strategies: list[ClientStrategyBase],
    smiles: str,
    neighbours: list[RetrievedMolecule],
) -> tuple[list[list[MolecularEvidence]], set[str]]:
    """
    Evidence for the queried molecule first, then for each neighbour in retrieval
    order. The queried molecule is not stored, so every source is asked by structure;
    a neighbour is asked only of the providers it was actually ingested from.

    Also returns the providers that could not be reached at all, so the prompt can
    distinguish a source being down from a molecule genuinely having no data.
    """
    unavailable: set[str] = set()
    evidence = await asyncio.gather(
        _flatten([_retrieve(strategy, smiles, unavailable) for strategy in strategies]),
        *(
            _flatten(_calls_for_stored(strategies, neighbour.molecule, unavailable))
            for neighbour in neighbours
        ),
    )
    return list(evidence), unavailable
