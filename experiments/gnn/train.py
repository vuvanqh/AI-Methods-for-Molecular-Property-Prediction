"""Multi-task Chemprop / CheMeleon GNN training script with local MLflow logging."""

from __future__ import annotations

# PyTDC (imported via `gnn_data.preprocess`) still imports `pkg_resources`, removed in setuptools>=82.
try:
    import pkg_resources  # type: ignore[import-untyped]  # noqa: F401
except ModuleNotFoundError:
    import importlib.metadata as _importlib_metadata
    import importlib.resources as _importlib_resources
    import sys as _sys
    import types as _types

    _pkg_resources = _types.ModuleType("pkg_resources")
    _pkg_resources.get_distribution = lambda name: _types.SimpleNamespace(  # type: ignore[attr-defined]
        version=_importlib_metadata.version(name)
    )
    _pkg_resources.resource_filename = lambda pkg, resource: str(  # type: ignore[attr-defined]
        _importlib_resources.files(pkg).joinpath(resource)
    )
    _sys.modules["pkg_resources"] = _pkg_resources

import argparse
import copy
import math
from pathlib import Path
import random
import tempfile
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gnn_data.preprocess import (
    ALL_TASKS,
    CYP450_TASKS,
    HERG_TASKS,
    TOX_21_TASKS,
    preprocess_cv_datasets,
    preprocess_datasets,
)
from src.chemprop import CheMeleonModel, ChempropModel, build_chemprop_loader
from src.config import DEVICE, GNNExperimentsSettings, settings
from src.loss import MaskedMultiTaskLoss
from src.metrics import compute_metrics, save_predictions_csv

ModelType = ChempropModel | CheMeleonModel


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_eval_batch_size(n_samples: int, batch_size: int) -> int:
    """
    Pick an eval batch size that never leaves a trailing 1-molecule batch.
    We cannot use `drop_last=True` on val/test (all molecules must be scored),
    and a size-1 batch breaks BatchNorm1d (encountered on CV Fold 1: 6,465 % 64 == 1).
    """
    bs = batch_size
    while n_samples > 1 and n_samples % bs == 1:
        bs += 1
    return bs


def build_loaders(
    cfg: GNNExperimentsSettings,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build Chemprop/CheMeleon train, validation, and test DataLoaders."""
    if cfg.model_name not in {"chemprop", "chemeleon", "chemeleon_scratch"}:
        raise ValueError(
            f"Unsupported model_name '{cfg.model_name}'. "
            "Choose 'chemprop', 'chemeleon', or 'chemeleon_scratch'."
        )
    val_bs = _safe_eval_batch_size(len(valid_df), cfg.batch_size)
    test_bs = _safe_eval_batch_size(len(test_df), cfg.batch_size)
    return (
        build_chemprop_loader(train_df, ALL_TASKS, batch_size=cfg.batch_size, shuffle=True),
        build_chemprop_loader(valid_df, ALL_TASKS, batch_size=val_bs, shuffle=False),
        build_chemprop_loader(test_df, ALL_TASKS, batch_size=test_bs, shuffle=False),
    )


def build_model(cfg: GNNExperimentsSettings, n_tasks: int) -> ModelType:
    """Instantiate configured multi-task Chemprop or CheMeleon model."""
    if cfg.model_name == "chemprop":
        return ChempropModel(n_tasks=n_tasks)
    if cfg.model_name in {"chemeleon", "chemeleon_scratch"}:
        return CheMeleonModel(
            n_tasks=n_tasks,
            aggregation=cfg.aggregation,
            ffn_hidden_dim=cfg.ffn_hidden_dim,
            ffn_num_layers=cfg.ffn_num_layers,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
            head_type=cfg.head_type,
            head_norm_act=cfg.head_norm_act,
            branch_dropouts=cfg.branch_dropouts,
            pretrained=(cfg.model_name == "chemeleon"),
        )
    raise ValueError(
        f"Unsupported model_name '{cfg.model_name}'. "
        "Choose 'chemprop', 'chemeleon', or 'chemeleon_scratch'."
    )


def _set_frozen_encoder_eval(model: ModelType) -> None:
    model.model.message_passing.eval()
    model.model.bn.eval()


class _MaskedFocalMultiTaskLoss(nn.Module):
    """Focal-modulated masked weighted BCE (`(1 - p_t)^gamma * weighted_bce`) for ablation studies."""

    def __init__(
        self,
        tasks: list[str],
        pos_weights: dict[str, float],
        task_weights: dict[str, float],
        use_task_weights: bool = True,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        pw = [pos_weights[t] for t in tasks]
        tw = [task_weights[t] if use_task_weights else 1.0 for t in tasks]
        self.register_buffer("pos_weight_vec", torch.tensor(pw, dtype=torch.float32))
        self.register_buffer("task_weight_vec", torch.tensor(tw, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        pos_w = self.pos_weight_vec.expand_as(targets)[mask]
        task_w = self.task_weight_vec.expand_as(targets)[mask]
        z = logits[mask]
        y = targets[mask]
        bce = nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w, reduction="none")
        probs = torch.sigmoid(z)
        pt = torch.where(y >= 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
        focal = (1.0 - pt).pow(self.gamma)
        return (focal * bce * task_w).mean()


def _build_criterion(cfg: GNNExperimentsSettings, meta: dict) -> nn.Module:
    pos_w = dict(meta["pos_weights"])
    if cfg.pos_weight_cap is not None:
        cap = float(cfg.pos_weight_cap)
        pos_w = {k: min(float(v), cap) for k, v in pos_w.items()}

    task_w = dict(meta["task_weights"])
    if cfg.task_weight_mode == "sqrt":
        raw = {k: math.sqrt(float(v)) for k, v in task_w.items()}
        mean_w = sum(raw.values()) / max(len(raw), 1)
        task_w = {k: v / mean_w for k, v in raw.items()}
    elif cfg.task_weight_mode != "inv":
        raise ValueError(f"Unsupported task_weight_mode '{cfg.task_weight_mode}'. Choose 'inv' or 'sqrt'.")

    if cfg.focal_gamma > 0.0:
        return _MaskedFocalMultiTaskLoss(
            tasks=ALL_TASKS,
            pos_weights=pos_w,
            task_weights=task_w,
            use_task_weights=cfg.use_task_weights,
            gamma=cfg.focal_gamma,
        )
    return MaskedMultiTaskLoss(
        tasks=ALL_TASKS,
        pos_weights=pos_w,
        task_weights=task_w,
        use_task_weights=cfg.use_task_weights,
    )


def build_optimizer(model: ModelType, cfg: GNNExperimentsSettings) -> torch.optim.Optimizer:
    """Create optimizer with separate learning rates for encoder body and head."""
    param_groups = [
        {"params": model.get_encoder_parameters(), "lr": cfg.lr_backbone, "initial_lr": cfg.lr_backbone},
        {"params": model.get_head_parameters(), "lr": cfg.lr_head, "initial_lr": cfg.lr_head},
    ]

    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Choose 'adamw', 'adam', or 'sgd'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: GNNExperimentsSettings,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create optional LR scheduler ('none' or 'plateau') across parameter groups."""
    if cfg.lr_scheduler == "none":
        return None
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
        )
    raise ValueError(f"Unsupported lr_scheduler '{cfg.lr_scheduler}'. Choose 'none' or 'plateau'.")


def train_one_epoch(
    model: ModelType,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
    label_smoothing: float = 0.0,
    pos_label_smoothing: float = 0.0,
    freeze_encoder: bool = False,
    l2sp_weight: float = 0.0,
    pretrained_enc_params: list[torch.Tensor] | None = None,
    ema_decay: float = 0.0,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> float:
    """Run one training epoch and return mean batch training loss."""
    model.train()
    if freeze_encoder:
        _set_frozen_encoder_eval(model)
    core = model.model
    losses: list[float] = []
    enc_params = (
        model.get_encoder_parameters()
        if (l2sp_weight > 0.0 and pretrained_enc_params is not None)
        else []
    )

    for batch in loader:
        batch.bmg.to(device)
        targets = batch.Y.to(device)
        if label_smoothing > 0.0:
            targets = torch.where(
                torch.isnan(targets),
                targets,
                targets * (1.0 - label_smoothing) + 0.5 * label_smoothing,
            )
        elif pos_label_smoothing > 0.0:
            targets = torch.where(
                targets > 0.5,
                targets * (1.0 - pos_label_smoothing),
                targets,
            )

        optimizer.zero_grad()
        logits = model(batch.bmg)
        bce_loss = criterion(logits, targets)
        if l2sp_weight > 0.0 and pretrained_enc_params is not None and not freeze_encoder:
            l2sp_pen = torch.zeros((), device=device)
            for p, p0 in zip(enc_params, pretrained_enc_params):
                if p.requires_grad:
                    l2sp_pen = l2sp_pen + torch.sum((p - p0) ** 2)
            total_loss = bce_loss + 0.5 * l2sp_weight * l2sp_pen
        else:
            total_loss = bce_loss
        total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        if ema_state is not None and ema_decay > 0.0:
            with torch.no_grad():
                for k, v in core.state_dict().items():
                    if v.is_floating_point():
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(v)

        losses.append(float(bce_loss.item()))

    return float(np.mean(losses))


def evaluate(
    model: ModelType,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run evaluation pass and return `(mean_loss, y_true, y_prob)`."""
    model.eval()
    losses: list[float] = []
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            targets = batch.Y.to(device)
            logits = model(batch.bmg)
            loss = criterion(logits, targets)

            losses.append(float(loss.item()))
            y_true_list.append(batch.Y.detach().cpu().numpy())
            y_prob_list.append(logits.sigmoid().cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(y_true_list, axis=0),
        np.concatenate(y_prob_list, axis=0),
    )


def log_run_artifacts(
    run_name: str,
    best_epoch: int,
    best_weights: dict[str, torch.Tensor],
    valid_smiles: pd.Series,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    test_smiles: pd.Series,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
    test_df_out: pd.DataFrame,
) -> None:
    """Write predictions, best checkpoint, and test metrics table to MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        preds_path = save_predictions_csv(
            valid_smiles=valid_smiles,
            y_true_val=y_true_val,
            y_prob_val=y_prob_val,
            test_smiles=test_smiles,
            y_true_test=y_true_test,
            y_prob_test=y_prob_test,
            output_path=tmp_dir / f"{run_name}_predictions.csv",
            tasks=ALL_TASKS,
        )
        mlflow.log_artifact(str(preds_path))

        ckpt_path = tmp_dir / f"{run_name}_best_ep{best_epoch}.pt"
        torch.save(best_weights, ckpt_path)
        mlflow.log_artifact(str(ckpt_path))

        metrics_csv = tmp_dir / "test_metrics.csv"
        test_df_out.to_csv(metrics_csv, index_label="task")
        mlflow.log_artifact(str(metrics_csv))


def _group_mean(per_task: dict[str, dict[str, float | str]], tasks: list[str], key: str) -> float:
    return float(np.nanmean([float(per_task[t][key]) for t in tasks]))


def train(
    run_name: str | None = None,
    cfg: GNNExperimentsSettings | None = None,
) -> dict[str, object]:
    """Full training, validation early stopping, test evaluation, and MLflow logging."""
    cfg = cfg if cfg is not None else settings
    t0 = time.perf_counter()
    seed_everything(cfg.seed)

    # 1. Configure local MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # 2. Load splits & build model-specific DataLoaders
    if cfg.cv_fold is None:
        train_df, valid_df, test_df, meta = preprocess_datasets(save_to_disk=True)
    else:
        folds, test_df, _ = preprocess_cv_datasets(n_folds=5, save_to_disk=True)
        if not (0 <= cfg.cv_fold < len(folds)):
            raise ValueError(f"cv_fold must be in 0..{len(folds) - 1}, got {cfg.cv_fold}.")
        fold_dict = folds[cfg.cv_fold]
        train_df = fold_dict["train_df"]
        valid_df = fold_dict["valid_df"]
        meta = {
            "pos_weights": fold_dict["pos_weights"],
            "task_weights": fold_dict["task_weights"],
        }

    seed_everything(cfg.seed)
    train_loader, valid_loader, test_loader = build_loaders(cfg, train_df, valid_df, test_df)

    # 3. Build model, loss, optimizer, and optional LR scheduler
    model = build_model(cfg, n_tasks=len(ALL_TASKS)).to(DEVICE)
    core = model.model
    pretrained_enc_params = (
        [p.detach().clone() for p in model.get_encoder_parameters()]
        if cfg.l2sp_weight > 0.0
        else None
    )
    ema_state = (
        {k: v.detach().clone() for k, v in core.state_dict().items()}
        if cfg.ema_decay > 0.0
        else None
    )
    criterion = _build_criterion(cfg, meta).to(DEVICE)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 4. Training loop with early stopping & per-task best epoch tracking
    run_name = run_name or (
        f"{cfg.model_name}_{cfg.optimizer}_bs{cfg.batch_size}_lr{cfg.lr_backbone:g}"
    )

    best_val_obj = -float("inf")
    best_epoch = 1
    epochs_ran = 0
    best_weights = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    best_task_epoch = {t: (1, -float("inf")) for t in ALL_TASKS}
    no_improve = 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "mlflow.user": "",
                "model_name": cfg.model_name,
                "optimizer": cfg.optimizer,
            }
        )
        mlflow.log_params({**cfg.to_dict(), "device": str(DEVICE)})

        for epoch in range(1, cfg.epochs + 1):
            epochs_ran = epoch
            freeze_enc = cfg.freeze_encoder_epochs > 0 and epoch <= cfg.freeze_encoder_epochs
            for p in model.get_encoder_parameters():
                p.requires_grad_(not freeze_enc)

            cur_lr_bb = 0.0 if freeze_enc else float(optimizer.param_groups[0]["lr"])
            cur_lr_hd = float(optimizer.param_groups[1]["lr"])

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
                grad_clip_norm=cfg.grad_clip_norm,
                label_smoothing=cfg.label_smoothing,
                pos_label_smoothing=cfg.pos_label_smoothing,
                freeze_encoder=freeze_enc,
                l2sp_weight=cfg.l2sp_weight,
                pretrained_enc_params=pretrained_enc_params,
                ema_decay=cfg.ema_decay,
                ema_state=ema_state,
            )

            live_state: dict[str, torch.Tensor] | None = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
                core.load_state_dict(ema_state)

            val_loss, y_true_val, y_prob_val = evaluate(model, valid_loader, criterion, DEVICE)
            val_obj, val_tasks = compute_metrics(y_true_val, y_prob_val, ALL_TASKS)

            if scheduler is not None and not freeze_enc:
                scheduler.step(val_obj)

            step_log = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_objective": val_obj,
                "lr_backbone_cur": cur_lr_bb,
                "lr_head_cur": cur_lr_hd,
            }
            for t, m in val_tasks.items():
                score = float(m["primary_score"])
                if not np.isnan(score):
                    step_log[f"val_{m['primary_metric']}/{t}"] = score
                    if score > best_task_epoch[t][1]:
                        best_task_epoch[t] = (epoch, score)

            mlflow.log_metrics(step_log, step=epoch)
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_objective={val_obj:.4f}"
            )

            if val_obj > best_val_obj:
                best_val_obj = val_obj
                best_epoch = epoch
                best_weights = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    if live_state is not None:
                        core.load_state_dict(live_state)
                    print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                    break

            if live_state is not None:
                core.load_state_dict(live_state)

        # 5. Restore best checkpoint & evaluate on validation and test sets
        core.load_state_dict(best_weights)
        _, y_true_val, y_prob_val = evaluate(model, valid_loader, criterion, DEVICE)
        _, y_true_test, y_prob_test = evaluate(model, test_loader, criterion, DEVICE)

        val_obj_best, val_tasks_best = compute_metrics(y_true_val, y_prob_val, ALL_TASKS)
        test_obj, test_tasks = compute_metrics(y_true_test, y_prob_test, ALL_TASKS)
        duration_sec = round(time.perf_counter() - t0, 2)

        val_tox21_pr_auc = _group_mean(val_tasks_best, TOX_21_TASKS, "pr_auc")
        val_cyp450_pr_auc = _group_mean(val_tasks_best, CYP450_TASKS, "pr_auc")
        val_herg_roc_auc = _group_mean(val_tasks_best, HERG_TASKS, "roc_auc")

        test_tox21_pr_auc = _group_mean(test_tasks, TOX_21_TASKS, "pr_auc")
        test_cyp450_pr_auc = _group_mean(test_tasks, CYP450_TASKS, "pr_auc")
        test_herg_roc_auc = _group_mean(test_tasks, HERG_TASKS, "roc_auc")
        test_mean_pr_auc = _group_mean(test_tasks, ALL_TASKS, "pr_auc")
        test_mean_roc_auc = _group_mean(test_tasks, ALL_TASKS, "roc_auc")

        test_log: dict[str, float] = {
            "best_epoch": float(best_epoch),
            "epochs_ran": float(epochs_ran),
            "best_val_objective": float(val_obj_best),
            "test_objective": float(test_obj),
            "duration_sec": float(duration_sec),
            "val_tox21_pr_auc": val_tox21_pr_auc,
            "val_cyp450_pr_auc": val_cyp450_pr_auc,
            "val_herg_roc_auc": val_herg_roc_auc,
            "test_tox21_pr_auc": test_tox21_pr_auc,
            "test_cyp450_pr_auc": test_cyp450_pr_auc,
            "test_herg_roc_auc": test_herg_roc_auc,
            "test_mean_pr_auc": test_mean_pr_auc,
            "test_mean_roc_auc": test_mean_roc_auc,
        }
        for t, m in test_tasks.items():
            test_log[f"best_epoch/{t}"] = float(best_task_epoch[t][0])
            test_log[f"test_pr_auc/{t}"] = float(m["pr_auc"])
            test_log[f"test_roc_auc/{t}"] = float(m["roc_auc"])
        mlflow.log_metrics(test_log)

        test_df_out = pd.DataFrame.from_dict(test_tasks, orient="index")
        test_df_out["best_val_epoch"] = [best_task_epoch[t][0] for t in ALL_TASKS]

        # 6. Log artifacts to MLflow
        log_run_artifacts(
            run_name=run_name,
            best_epoch=best_epoch,
            best_weights=best_weights,
            valid_smiles=valid_df["smiles"],
            y_true_val=y_true_val,
            y_prob_val=y_prob_val,
            test_smiles=test_df["smiles"],
            y_true_test=y_true_test,
            y_prob_test=y_prob_test,
            test_df_out=test_df_out,
        )

        print(
            f"\nDone! Run: {run_name} ({run.info.run_id}) | Best Epoch: {best_epoch}/{epochs_ran} | "
            f"Val Objective: {val_obj_best:.4f} | Test Objective: {test_obj:.4f} | Time: {duration_sec:.1f}s"
        )
        print(test_df_out.round(4).to_string())

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model_name": cfg.model_name,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "val_objective": float(val_obj_best),
            "test_objective": float(test_obj),
            "val_tox21_pr_auc": val_tox21_pr_auc,
            "val_cyp450_pr_auc": val_cyp450_pr_auc,
            "val_herg_roc_auc": val_herg_roc_auc,
            "test_tox21_pr_auc": test_tox21_pr_auc,
            "test_cyp450_pr_auc": test_cyp450_pr_auc,
            "test_herg_roc_auc": test_herg_roc_auc,
            "test_mean_pr_auc": test_mean_pr_auc,
            "test_mean_roc_auc": test_mean_roc_auc,
            "duration_sec": duration_sec,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task GNN with MLflow.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom MLflow run name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["chemprop", "chemeleon", "chemeleon_scratch"],
        default=None,
        help="Optional model override.",
    )
    args = parser.parse_args()

    cfg = copy.deepcopy(settings)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.model is not None:
        cfg.model_name = args.model

    train(run_name=args.run_name, cfg=cfg)


if __name__ == "__main__":
    main()
