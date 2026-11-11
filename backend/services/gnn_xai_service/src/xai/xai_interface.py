from abc import ABC, abstractmethod

import torch


class XAIExplainer(ABC):
    """Abstract interface for atom-level XAI attribution explainers."""

    @abstractmethod
    def explain(
        self,
        atom_embeddings: torch.Tensor,
        num_targets: int,
    ) -> list[list[float]]:
        """Calculate atom-level attribution scores `[target_index][atom_index]` for all model targets."""
