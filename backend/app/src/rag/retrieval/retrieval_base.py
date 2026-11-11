from abc import ABC, abstractmethod

from ..result_types.retrieved_molecules import RetrievedMolecule


class Retrieval(ABC):
    @abstractmethod
    def retrieve(self, smiles: str, top_k: int) -> list[RetrievedMolecule]:
        """
        Abstract method to retrieve similar molecules from the database.
        """
        raise NotImplementedError("Subclasses must implement this method.")
