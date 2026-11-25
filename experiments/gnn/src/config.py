"""Training settings for GNN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GNNExperimentsSettings:
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "gnn_multitask"
    seed: int = 42

    model_name: str = "chemprop"  # "chemprop" | "chemeleon" | "chemeleon_scratch"
    aggregation: str = "mean"
    head_type: str = "standard"  # "standard" | "branched"
    head_norm_act: str = "relu"  # "relu" | "ln_silu"
    ffn_hidden_dim: int | tuple[int, ...] = 300
    ffn_num_layers: int = 1
    dropout: float = 0.0
    branch_dropouts: tuple[float, ...] | None = None
    batch_norm: bool = False
    cv_fold: int | None = None  # None = fixed 70/10/20 split; 0..4 = 5-fold scaffold CV

    # Loss & fine-tuning settings
    use_task_weights: bool = True
    task_weight_mode: str = "inv"  # "inv" (1/N_t, default) | "sqrt" (1/sqrt(N_t))
    pos_weight_cap: float | None = None  # None = uncapped (default); e.g. 10.0 or 15.0
    focal_gamma: float = 0.0  # 0.0 = standard weighted BCE (default); >0 = focal modulation (1-p_t)^gamma
    label_smoothing: float = 0.0
    pos_label_smoothing: float = 0.0
    freeze_encoder_epochs: int = 0
    l2sp_weight: float = 0.0
    ema_decay: float = 0.0

    # training settings
    epochs: int = 40
    batch_size: int = 64
    optimizer: str = "adamw"
    lr_backbone: float = 3e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "none"  # "none" | "plateau"
    grad_clip_norm: float | None = None
    patience: int = 10

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        if isinstance(self.ffn_hidden_dim, (tuple, list)):
            raw["ffn_hidden_dim"] = (
                "x".join(str(d) for d in self.ffn_hidden_dim) if self.ffn_hidden_dim else "0"
            )
        if isinstance(self.branch_dropouts, (tuple, list)):
            raw["branch_dropouts"] = (
                "x".join(f"{float(d):g}" for d in self.branch_dropouts)
                if self.branch_dropouts
                else "none"
            )
        return {k: ("none" if v is None else v) for k, v in raw.items()}


settings = GNNExperimentsSettings()
