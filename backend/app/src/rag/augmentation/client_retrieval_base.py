from abc import ABC, abstractmethod

from ..result_types.molecular_evidence import MolecularEvidence


class ClientStrategyBase(ABC):
    # matches MoleculeSource.provider, so callers can pass this source's own identifier
    provider: str

    @abstractmethod
    async def retrieve_molecular_evidence(self, query: str) -> list[MolecularEvidence]:
        '''
        Fetch experimental evidence for one molecule from an external source.
        The query is a SMILES or the source's own identifier for the molecule.
        '''
        raise NotImplementedError("Subclasses must implement this method.")
