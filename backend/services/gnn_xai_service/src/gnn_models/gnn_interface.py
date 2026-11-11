from abc import ABC, abstractmethod

import torch
from rdkit import Chem


class GNNPredictor(ABC):
    """Abstract interface for molecular Graph Neural Network predictors."""

    @abstractmethod
    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        """Predict molecular properties (`probabilities`, `predictions`, `atom_embeddings`) for one RDKit molecule."""

    @abstractmethod
    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute differentiable task logits `[1, n_tasks]` from single-molecule `atom_embeddings` `[num_atoms, hidden_dim]`."""
