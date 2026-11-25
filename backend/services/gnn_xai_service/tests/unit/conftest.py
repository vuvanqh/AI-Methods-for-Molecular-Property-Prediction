"""Shared lightweight stubs and response builders for gnn_xai_service unit tests."""

from typing import Any

import torch
from rdkit import Chem

from src.config import TASKS, TASKS_NUMBER
from src.gnn_models.gnn_interface import GNNPredictor
from src.xai.xai_interface import XAIExplainer


def make_valid_inference_response(
    *, num_heavy_atoms: int = 2, num_explicit_h: int = 1
) -> dict[str, Any]:
    """Build a schema-valid inference response dict with distinct per-task values."""
    return {
        "predictions": {task: (idx % 2 == 0) for idx, task in enumerate(TASKS)},
        "probabilities": {
            task: round((idx + 1) / (TASKS_NUMBER + 1), 6)
            for idx, task in enumerate(TASKS)
        },
        "atomscores": {
            task: {
                **{
                    i: round(0.1 * (i + 1) - 0.05 * task_idx, 6)
                    for i in range(num_heavy_atoms)
                },
                **{num_heavy_atoms + h: None for h in range(num_explicit_h)},
            }
            for task_idx, task in enumerate(TASKS)
        },
        "molblock": (
            "stub\n  RDKit          3D\n\n"
            "  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
        ),
    }


class StubGNNPredictor(GNNPredictor):
    """Deterministic `GNNPredictor` stub/spy for isolating service orchestration tests."""

    def __init__(
        self,
        *,
        custom_output: dict[str, torch.Tensor] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._custom_output = custom_output
        self._exc = exc
        self.predict_call_count = 0
        self.received_mols: list[Chem.Mol] = []
        self.received_smiles: list[str] = []

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        self.predict_call_count += 1
        self.received_mols.append(mol)
        self.received_smiles.append(Chem.MolToSmiles(mol, canonical=False))

        if self._exc is not None:
            raise self._exc
        if self._custom_output is not None:
            return self._custom_output

        atom_embeddings = torch.tensor(
            [[float(a.GetAtomicNum()), float(a.GetDegree())] for a in mol.GetAtoms()],
            dtype=torch.float32,
        )
        probabilities = torch.linspace(
            0.05, 0.95, steps=TASKS_NUMBER, dtype=torch.float32
        )
        return {
            "probabilities": probabilities,
            "predictions": (probabilities > 0.5).int(),
            "atom_embeddings": atom_embeddings,
        }

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, TASKS_NUMBER), dtype=torch.float32)


class StubXAIExplainer(XAIExplainer):
    """Deterministic `XAIExplainer` stub/spy for isolating service orchestration tests."""

    def __init__(
        self,
        *,
        custom_scores: list[list[float]] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._custom_scores = custom_scores
        self._exc = exc
        self.explain_call_count = 0
        self.received_args: list[tuple[torch.Tensor, int]] = []

    def explain(
        self, atom_embeddings: torch.Tensor, num_targets: int
    ) -> list[list[float]]:
        self.explain_call_count += 1
        self.received_args.append((atom_embeddings, num_targets))

        if self._exc is not None:
            raise self._exc
        if self._custom_scores is not None:
            return self._custom_scores

        atom_sums = atom_embeddings.sum(dim=-1).tolist()
        return [
            [float(val) + 0.01 * target_idx for val in atom_sums]
            for target_idx in range(num_targets)
        ]
