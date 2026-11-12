"""Leaderboard evaluation metrics (ROC-AUC and PR-AUC) and prediction CSV builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gnn_data.preprocess import ALL_TASKS, CYP450_TASKS, HERG_TASKS, TOX_21_TASKS

PRIMARY_METRIC: dict[str, str] = {
    **{t: "pr_auc" for t in TOX_21_TASKS},
    **{t: "pr_auc" for t in CYP450_TASKS},
    **{t: "roc_auc" for t in HERG_TASKS},
}


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tasks: list[str] = ALL_TASKS,
) -> tuple[float, dict[str, dict[str, float | str]]]:
    """Compute ROC-AUC, PR-AUC, and designated leaderboard primary score per endpoint."""
    per_task: dict[str, dict[str, float | str]] = {}
    primary_scores: list[float] = []

    for i, task in enumerate(tasks):
        valid = ~np.isnan(y_true[:, i])
        yt = y_true[valid, i].astype(int)
        yp = y_prob[valid, i]

        # if both 0 and 1 labels are in the split
        if len(yt) > 0 and len(np.unique(yt)) == 2:
            roc = float(roc_auc_score(yt, yp))
            pr = float(average_precision_score(yt, yp))
        else:
            roc, pr = float("nan"), float("nan")

        metric_name = PRIMARY_METRIC[task]
        primary = roc if metric_name == "roc_auc" else pr
        primary_scores.append(primary)

        per_task[task] = {
            "primary_metric": metric_name,
            "primary_score": primary,
            "roc_auc": roc,
            "pr_auc": pr,
        }

    mean_objective = float(np.nanmean(primary_scores))
    return mean_objective, per_task


def save_predictions_csv(
    valid_smiles: pd.Series,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    test_smiles: pd.Series,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
    output_path: Path,
    tasks: list[str] = ALL_TASKS,
) -> Path:
    """Gather validation and test ground-truth labels + predicted probabilities into one CSV file."""

    def _pack(smiles: pd.Series, split: str, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        cols: dict[str, object] = {"smiles": smiles.tolist(), "split": split}
        for i, task in enumerate(tasks):
            cols[f"y_true_{task}"] = y_true[:, i].astype(np.float32)
            cols[f"y_prob_{task}"] = y_prob[:, i].astype(np.float32)
        return pd.DataFrame(cols)

    df = pd.concat(
        [
            _pack(valid_smiles, "valid", y_true_val, y_prob_val),
            _pack(test_smiles, "test", y_true_test, y_prob_test),
        ],
        ignore_index=True,
    )
    df.to_csv(output_path, index=False)
    return output_path
