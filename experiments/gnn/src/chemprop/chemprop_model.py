"""Chemprop MPNN model identical to `gnn_xai_service/src/gnn_models/chemprop.py`."""

from __future__ import annotations

import torch
import torch.nn as nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import BinaryClassificationFFN, BondMessagePassing, NormAggregation


class ChempropModel(nn.Module):
    """
    Wraps Chemprop `MPNN(BondMessagePassing, NormAggregation, BinaryClassificationFFN)`
    and returns raw logits `[batch_size, n_tasks]` for `BCEWithLogitsLoss`.
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        mp = BondMessagePassing()
        agg = NormAggregation()
        ffn = BinaryClassificationFFN(n_tasks)
        self.model = MPNN(mp, agg, ffn)

    def get_encoder_parameters(self) -> list[nn.Parameter]:
        """
        Parameters of the message-passing body + batchnorm.
        Isolation needed because different lr might be applied to encoder and head.
        """
        return list(self.model.message_passing.parameters()) + list(self.model.bn.parameters())

    def get_head_parameters(self) -> list[nn.Parameter]:
        """
        Parameters of the classification FFN head.
        Isolation needed because different lr might be applied to encoder and head.
        """
        return list(self.model.predictor.parameters())

    def forward(self, bmg: BatchMolGraph) -> torch.Tensor:
        atom_embeddings = self.model.message_passing(bmg)
        mol_embeddings = self.model.agg(atom_embeddings, bmg.batch)
        mol_embeddings = self.model.bn(mol_embeddings)
        logits = self.model.predictor.ffn(mol_embeddings)
        return logits
