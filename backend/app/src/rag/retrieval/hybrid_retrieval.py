from ..config import retrievalSettings
from ..result_types.retrieved_molecules import RetrievedMolecule
from .retrieval_base import Retrieval


class HybridRetrieval(Retrieval):
    def __init__(
        self,
        fingerprint: Retrieval,
        semantic: Retrieval,
        rrf_k: int = 60,
    ):
        self.fingerprint = fingerprint
        self.semantic = semantic
        self.rrf_k = rrf_k

    def retrieve(self, smiles: str, top_k: int) -> list[RetrievedMolecule]:

        candidate_k = top_k * 3

        fp_results = self.fingerprint.retrieve(smiles, candidate_k)
        sem_results = self.semantic.retrieve(smiles, candidate_k)

        scores = {}
        molecules = {}

        for results, weight in (
            (fp_results, retrievalSettings.fingerprint_weight),
            (sem_results, retrievalSettings.semantic_weight),
        ):
            for rank, result in enumerate(results, start=1):
                molecules[result.molecule.id] = result.molecule
                scores[result.molecule.id] = scores.get(
                    result.molecule.id, 0.0
                ) + weight / (self.rrf_k + rank)

        ranked_ids = sorted(
            scores,
            key=lambda mol_id: (-scores[mol_id], str(mol_id)),
        )

        return [
            RetrievedMolecule(molecules[mol_id], scores[mol_id])
            for mol_id in ranked_ids[:top_k]
        ]
