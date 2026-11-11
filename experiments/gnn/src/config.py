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

    model_name: str = "chemprop"

    # Loss settings
    use_task_weights: bool = True

    # training settings
    epochs: int = 40
    batch_size: int = 64
    optimizer: str = "adamw"
    lr_backbone: float = 3e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


settings = GNNExperimentsSettings()
