"""Pretrained CheMeleon MPNN wrapper mirroring `ChempropModel`."""

from __future__ import annotations

import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import (
    BinaryClassificationFFN,
    BondMessagePassing,
    MeanAggregation,
    NormAggregation,
)

CHEMELEON_ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
DEFAULT_CHEMELEON_PATH = Path.home() / ".chemprop" / "chemeleon_mp.pt"
DEFAULT_CHEMELEON_HYPERPARAMS: dict[str, object] = {
    "d_v": 72,
    "d_e": 14,
    "d_h": 2048,
    "bias": False,
    "depth": 6,
    "dropout": 0.0,
    "activation": "relu",
    "undirected": False,
    "d_vd": None,
    "V_d_transform": None,
    "graph_transform": None,
}


def ensure_chemeleon_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Ensure `~/.chemprop/chemeleon_mp.pt` exists, downloading it atomically if missing."""
    mp_path = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path is not None
        else DEFAULT_CHEMELEON_PATH
    )
    if mp_path.exists() and mp_path.stat().st_size > 0:
        return mp_path

    mp_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = mp_path.with_name(f"{mp_path.name}.part")

    print(f"Downloading pretrained CheMeleon weights from {CHEMELEON_ZENODO_URL} -> {mp_path} ...")
    try:
        req = urllib.request.Request(
            CHEMELEON_ZENODO_URL,
            headers={"User-Agent": "chemprop-chemeleon-loader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=300) as response, open(part_path, "wb") as out_f:
            shutil.copyfileobj(response, out_f)
            out_f.flush()
            os.fsync(out_f.fileno())

        if not part_path.exists() or part_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded checkpoint at {part_path} is empty.")

        os.replace(part_path, mp_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    return mp_path


def _linear_block(in_dim: int, out_dim: int, dropout: float, norm_act: str) -> nn.Sequential:
    if norm_act == "relu":
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    if norm_act == "ln_silu":
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
    raise ValueError(f"Unsupported head_norm_act '{norm_act}'. Choose 'relu' or 'ln_silu'.")


class StandardNormActMLP(nn.Module):
    """Multi-layer FFN head with LayerNorm + SiLU blocks (`mol_embeddings -> logits`)."""

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        norm_act: str = "ln_silu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hdim in hidden_dims:
            layers.append(_linear_block(prev, hdim, dropout=dropout, norm_act=norm_act))
            prev = hdim
        layers.append(nn.Linear(prev, n_tasks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskGroupBranchedMLP(nn.Module):
    """
    Shared trunk (`2048 -> trunk_dim`) splitting into three specialized task-group branches
    matching `ALL_TASKS = TOX_21_TASKS (12) + CYP450_TASKS (5) + HERG_TASKS (1)`.
    """

    def __init__(
        self,
        input_dim: int,
        n_tasks: int = 18,
        trunk_dim: int = 1024,
        branch_dim: int = 512,
        dropout: float = 0.3,
        branch_dropouts: Sequence[float] | None = None,
        norm_act: str = "relu",
        group_sizes: tuple[int, ...] = (12, 5, 1),
    ) -> None:
        super().__init__()
        if sum(group_sizes) != n_tasks:
            raise ValueError(
                f"TaskGroupBranchedMLP group_sizes {group_sizes} sum to {sum(group_sizes)}, "
                f"expected n_tasks={n_tasks}."
            )
        if branch_dropouts is not None and len(branch_dropouts) != len(group_sizes):
            raise ValueError(
                f"branch_dropouts length {len(branch_dropouts)} must match group_sizes {group_sizes}."
            )
        self.trunk = _linear_block(input_dim, trunk_dim, dropout=dropout, norm_act=norm_act)
        herg_dim = max(branch_dim // 2, 128)
        branch_dims = (branch_dim, branch_dim, herg_dim)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        list(
                            _linear_block(
                                trunk_dim,
                                b_dim,
                                dropout=(
                                    float(branch_dropouts[idx])
                                    if branch_dropouts is not None
                                    else dropout
                                ),
                                norm_act=norm_act,
                            ).children()
                        )
                        + [nn.Linear(b_dim, out_tasks)]
                    )
                )
                for idx, (out_tasks, b_dim) in enumerate(zip(group_sizes, branch_dims))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([branch(h) for branch in self.branches], dim=-1)


class CheMeleonModel(nn.Module):
    """
    Wraps pretrained CheMeleon `BondMessagePassing` (2048-dim) inside Chemprop `MPNN`
    and returns raw logits `[batch_size, n_tasks]` for `BCEWithLogitsLoss` and Captum IG XAI.
    """

    def __init__(
        self,
        n_tasks: int,
        aggregation: str = "mean",
        ffn_hidden_dim: int | Sequence[int] = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        head_type: str = "standard",
        head_norm_act: str = "relu",
        branch_dropouts: Sequence[float] | None = None,
        checkpoint_path: str | Path | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            mp_path = ensure_chemeleon_checkpoint(checkpoint_path)
            chemeleon_mp = torch.load(mp_path, map_location="cpu", weights_only=True)
            mp = BondMessagePassing(**chemeleon_mp["hyper_parameters"])
            mp.load_state_dict(chemeleon_mp["state_dict"])
        else:
            mp = BondMessagePassing(**DEFAULT_CHEMELEON_HYPERPARAMS)

        if aggregation == "mean":
            agg = MeanAggregation()
        elif aggregation == "norm":
            agg = NormAggregation()
        else:
            raise ValueError(f"Unsupported aggregation '{aggregation}'. Choose 'mean' or 'norm'.")

        if isinstance(ffn_hidden_dim, Sequence) and not isinstance(ffn_hidden_dim, (str, bytes)):
            hidden_dim_arg: int | Sequence[int] = (
                () if ffn_num_layers == 0 else tuple(int(d) for d in ffn_hidden_dim)
            )
            n_layers_arg = len(hidden_dim_arg)
        else:
            hidden_dim_arg = int(ffn_hidden_dim)
            n_layers_arg = int(ffn_num_layers)

        ffn = BinaryClassificationFFN(
            n_tasks=n_tasks,
            input_dim=mp.output_dim,
            hidden_dim=hidden_dim_arg,
            n_layers=n_layers_arg,
            dropout=dropout,
        )
        if head_type == "branched":
            if isinstance(hidden_dim_arg, tuple) and len(hidden_dim_arg) >= 2:
                trunk_dim, branch_dim = int(hidden_dim_arg[0]), int(hidden_dim_arg[1])
            elif isinstance(hidden_dim_arg, tuple) and len(hidden_dim_arg) == 1:
                trunk_dim = int(hidden_dim_arg[0])
                branch_dim = max(trunk_dim // 2, 128)
            else:
                trunk_dim = int(hidden_dim_arg) if isinstance(hidden_dim_arg, int) else 1024
                branch_dim = max(trunk_dim // 2, 128)
            ffn.ffn = TaskGroupBranchedMLP(
                input_dim=mp.output_dim,
                n_tasks=n_tasks,
                trunk_dim=trunk_dim,
                branch_dim=branch_dim,
                dropout=dropout,
                branch_dropouts=branch_dropouts,
                norm_act=head_norm_act,
            )
        elif head_type == "standard":
            if head_norm_act != "relu":
                hdims = (
                    tuple(int(d) for d in hidden_dim_arg)
                    if isinstance(hidden_dim_arg, tuple)
                    else (int(hidden_dim_arg),) * n_layers_arg
                )
                ffn.ffn = StandardNormActMLP(
                    input_dim=mp.output_dim,
                    n_tasks=n_tasks,
                    hidden_dims=hdims,
                    dropout=dropout,
                    norm_act=head_norm_act,
                )
        else:
            raise ValueError(f"Unsupported head_type '{head_type}'. Choose 'standard' or 'branched'.")

        self.model = MPNN(mp, agg, ffn, batch_norm=batch_norm)

    def get_encoder_parameters(self) -> list[nn.Parameter]:
        """Parameters of the message-passing body + batchnorm for differential learning rates."""
        return list(self.model.message_passing.parameters()) + list(self.model.bn.parameters())

    def get_head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the classification FFN head for differential learning rates."""
        return list(self.model.predictor.parameters())

    def forward(self, bmg: BatchMolGraph) -> torch.Tensor:
        atom_embeddings = self.model.message_passing(bmg)
        mol_embeddings = self.model.agg(atom_embeddings, bmg.batch)
        mol_embeddings = self.model.bn(mol_embeddings)
        logits = self.model.predictor.ffn(mol_embeddings)
        return logits

