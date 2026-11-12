"""Multi-task GNN training script with local MLflow logging."""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn_data.preprocess import ALL_TASKS, preprocess_datasets
from src.chemprop import ChempropModel, build_chemprop_loader
from src.config import DEVICE, GNNExperimentsSettings, settings
from src.loss import MaskedMultiTaskLoss
from src.metrics import compute_metrics, save_predictions_csv


def build_optimizer(model: ChempropModel, cfg: GNNExperimentsSettings) -> torch.optim.Optimizer:
    """Create optimizer with separate learning rates for encoder body and head."""
    param_groups = [
        {"params": model.get_encoder_parameters(), "lr": cfg.lr_backbone},
        {"params": model.get_head_parameters(), "lr": cfg.lr_head},
    ]

    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Choose 'adamw', 'adam', or 'sgd'.")


def train_one_epoch(
    model: ChempropModel,
    loader: DataLoader,
    criterion: MaskedMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean batch training loss."""
    model.train()
    losses: list[float] = []

    for batch in loader:
        batch.bmg.to(device)
        targets = batch.Y.to(device)

        optimizer.zero_grad()
        logits = model(batch.bmg)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def evaluate(
    model: ChempropModel,
    loader: DataLoader,
    criterion: MaskedMultiTaskLoss,
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
            y_true_list.append(batch.Y.numpy())
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


def train(run_name: str | None = None) -> None:
    """Full training, validation early stopping, test evaluation, and MLflow logging."""
    torch.manual_seed(settings.seed)

    # 1. Configure local MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    # 2. Load splits & build Chemprop DataLoaders
    train_df, valid_df, test_df, meta = preprocess_datasets(save_to_disk=True)

    train_loader = build_chemprop_loader(train_df, ALL_TASKS, batch_size=settings.batch_size, shuffle=True)
    valid_loader = build_chemprop_loader(valid_df, ALL_TASKS, batch_size=settings.batch_size, shuffle=False)
    test_loader = build_chemprop_loader(test_df, ALL_TASKS, batch_size=settings.batch_size, shuffle=False)

    # 3. Build model, loss, and optimizer
    model = ChempropModel(n_tasks=len(ALL_TASKS)).to(DEVICE)
    criterion = MaskedMultiTaskLoss(
        tasks=ALL_TASKS,
        pos_weights=meta["pos_weights"],
        task_weights=meta["task_weights"],
        use_task_weights=settings.use_task_weights,
    ).to(DEVICE)
    optimizer = build_optimizer(model, settings)

    # 4. Training loop with early stopping & per-task best epoch tracking
    run_name = run_name or (
        f"{settings.model_name}_{settings.optimizer}_bs{settings.batch_size}_lr{settings.lr_backbone:g}"
    )

    best_val_obj = -float("inf")
    best_epoch = 1
    best_weights = copy.deepcopy(model.model.state_dict())
    best_task_epoch = {t: (1, -float("inf")) for t in ALL_TASKS}
    no_improve = 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"mlflow.user": "","model_name": settings.model_name, "optimizer": settings.optimizer})
        mlflow.log_params({**settings.to_dict(), "device": str(DEVICE)})

        for epoch in range(1, settings.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, y_true_val, y_prob_val = evaluate(model, valid_loader, criterion, DEVICE)
            val_obj, val_tasks = compute_metrics(y_true_val, y_prob_val, ALL_TASKS)

            step_log = {"train_loss": train_loss, "val_loss": val_loss, "val_objective": val_obj}
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
                best_weights = copy.deepcopy(model.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= settings.patience:
                    print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                    break

        # 5. Restore best checkpoint & evaluate on validation and test sets
        model.model.load_state_dict(best_weights)
        _, y_true_val, y_prob_val = evaluate(model, valid_loader, criterion, DEVICE)
        _, y_true_test, y_prob_test = evaluate(model, test_loader, criterion, DEVICE)

        test_obj, test_tasks = compute_metrics(y_true_test, y_prob_test, ALL_TASKS)

        test_log: dict[str, float] = {"best_epoch": best_epoch, "test_objective": test_obj}
        for t, m in test_tasks.items():
            test_log[f"best_epoch/{t}"] = best_task_epoch[t][0]
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

        print(f"\nDone! Run: {run_name} ({run.info.run_id}) | Best Epoch: {best_epoch} | Test Objective: {test_obj:.4f}")
        print(test_df_out.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task GNN with MLflow.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom MLflow run name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    args = parser.parse_args()

    if args.epochs is not None:
        settings.epochs = args.epochs

    train(run_name=args.run_name)


if __name__ == "__main__":
    main()
