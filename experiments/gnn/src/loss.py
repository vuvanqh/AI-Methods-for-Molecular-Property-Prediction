"""Masked, task-weighted, and class-weighted BCE loss for multi-task training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiTaskLoss(nn.Module):
    """
    Binary cross-entropy with logits with mask:
      1. Slices out only valid (non-`NaN`) entries before computing BCE.
      2. Applies within-task positive class weights.
      3. Optionally applies across-task sample-count weights.
    """

    def __init__(
        self,
        tasks: list[str],
        pos_weights: dict[str, float],
        task_weights: dict[str, float],
        use_task_weights: bool = True,
    ) -> None:
        super().__init__()
        pw = [pos_weights[t] for t in tasks]
        tw = [task_weights[t] if use_task_weights else 1.0 for t in tasks]

        self.register_buffer("pos_weight_vec", torch.tensor(pw, dtype=torch.float32))
        self.register_buffer("task_weight_vec", torch.tensor(tw, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)

        pos_w = self.pos_weight_vec.expand_as(targets)
        task_w = self.task_weight_vec.expand_as(targets)

        bce = F.binary_cross_entropy_with_logits(
            logits[mask],
            targets[mask],
            pos_weight=pos_w[mask],
            reduction="none", #not to avg loss before applying task weights
        )
        return (bce * task_w[mask]).mean()
