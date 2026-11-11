"""
Data Preprocessing Pipeline.

Provides two functions that return splits and training weights and optionally save them to `data/processed/`:
  - `preprocess_datasets()`: Fixed 70% train / 10% valid / 20% test balanced scaffold split.
  - `preprocess_cv_datasets()`: 5-fold cross-validation scaffold splits (80% train+validation) + 20% held-out test set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tdc.single_pred import ADME, Tox
from tdc.utils import retrieve_label_name_list


TOX_21_TASKS: list[str] = retrieve_label_name_list("Tox21")

CYP450_TASKS: list[str] = [
    "CYP2C19_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP1A2_Veith",
    "CYP2C9_Veith",
]
HERG_TASKS: list[str] = ["hERG_Karim"]

ALL_TASKS: list[str] = TOX_21_TASKS + CYP450_TASKS + HERG_TASKS


def fetch_raw_tdc_datasets(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Download (if missing in `raw_dir`) or load cached TDC datasets for all 18 binary endpoints.
    Returns a dictionary mapping each endpoint name to its `["Drug", "Y"]` DataFrame.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dfs: dict[str, pd.DataFrame] = {}

    label_list = retrieve_label_name_list("Tox21")

    for label in label_list:
        data = Tox(name="Tox21", label_name=label, path=str(raw_dir))
        raw_dfs[label] = data.get_data()[["Drug", "Y"]].astype({"Y": int})

    for task in CYP450_TASKS:
        data = ADME(name=task, path=str(raw_dir))
        raw_dfs[task] = data.get_data()[["Drug", "Y"]].astype({"Y": int})

    for task in HERG_TASKS:
        data = Tox(name=task, path=str(raw_dir))
        raw_dfs[task] = data.get_data()[["Drug", "Y"]].astype({"Y": int})

    return raw_dfs


def build_clean_merged_dataframe(
    raw_dir: Path,
    filter_zero_bonds: bool = True,
) -> pd.DataFrame:
    """
    Canonicalize SMILES across all 18 endpoints, optionally filter zero-bond molecules,
    and merge into a single table with columns `['smiles'] + ALL_TASKS + ['scaffold_smiles']`.
    """
    raw_dfs = fetch_raw_tdc_datasets(raw_dir)

    all_raw_smiles = sorted({smile for df in raw_dfs.values() for smile in df["Drug"]})

    raw_to_canon: dict[str, str] = {}
    canon_to_scaffold: dict[str, str] = {}

    for s in all_raw_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        if filter_zero_bonds and mol.GetNumBonds() == 0:
            continue
        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        raw_to_canon[s] = canon_smiles
        if canon_smiles not in canon_to_scaffold:
            canon_to_scaffold[canon_smiles] = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )

    task_series: dict[str, pd.Series] = {}
    for task in ALL_TASKS:
        df = raw_dfs[task]
        df_val = df[df["Drug"].isin(raw_to_canon)].copy()
        df_val["smiles"] = df_val["Drug"].map(raw_to_canon)
        task_series[task] = (
            df_val.drop_duplicates("smiles").set_index("smiles")["Y"].astype(int)
        )

    merged_df = pd.DataFrame(task_series)[ALL_TASKS].sort_index()
    merged_df.index.name = "smiles"
    merged_df = merged_df.reset_index()
    merged_df["scaffold_smiles"] = merged_df["smiles"].map(canon_to_scaffold)
    return merged_df


def assign_buckets_to_smiles(
    df: pd.DataFrame,
    bucket_fractions: list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Deterministically partition molecules in `df` into `len(bucket_fractions)` scaffold-disjoint
    buckets, greedily balancing both negative (`0`) and positive (`1`) labels across all 18 tasks.
    Returns an integer array of bucket indices (`0 .. n_buckets - 1`) of length `len(df)`.
    """
    fracs = np.asarray(bucket_fractions, dtype=float)
    fracs = fracs / fracs.sum()
    n_buckets = len(fracs)

    scaf_groups: dict[str, list[int]] = defaultdict(list)
    for idx, scaf in enumerate(df["scaffold_smiles"].values):
        scaf_groups[scaf].append(idx)

    neg_indicator_matrix = (df[ALL_TASKS].values == 0).astype(float)
    pos_indicator_matrix = (df[ALL_TASKS].values == 1).astype(float)
    label_indicator_matrix = np.hstack([neg_indicator_matrix, pos_indicator_matrix])

    bucket_target_counts = np.outer(fracs, label_indicator_matrix.sum(axis=0))

    scaf_items = sorted(scaf_groups.items(), key=lambda x: (-len(x[1]), x[0]))

    bucket_current_counts = np.zeros((n_buckets, 36), dtype=float)
    bucket_ids = np.empty(len(df), dtype=int)

    for _scaf, idxs in scaf_items:
        scaffold_label_counts = label_indicator_matrix[idxs].sum(axis=0)
        active_cols = scaffold_label_counts > 0

        best_b, best_cost = 0, float("inf")

        for b in range(n_buckets):
            fill_ratio = (
                bucket_current_counts[b] + scaffold_label_counts
            ) / bucket_target_counts[b]
            cost = float(np.mean(fill_ratio[active_cols]))
            if cost < best_cost:
                best_cost, best_b = cost, b

        bucket_current_counts[best_b] += scaffold_label_counts
        bucket_ids[idxs] = best_b

    return bucket_ids


def compute_train_weights(
    train_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute within-task positive class weights (`pos_weights = N_neg / N_pos`) and across-task
    sample-size weights (`task_weights`, inverse training sample count divided by mean of all inverses).
    """
    pos_weights: dict[str, float] = {}
    train_counts: dict[str, int] = {}

    for task in ALL_TASKS:
        column_cleaned_train = train_df[task].dropna()
        num_neg_label = int((column_cleaned_train == 0).sum())
        num_pos_label = int((column_cleaned_train == 1).sum())
        pos_weights[task] = round(num_neg_label / max(num_pos_label, 1), 4)
        train_counts[task] = len(column_cleaned_train)

    inv_counts = {task: 1.0 / max(counts, 1) for task, counts in train_counts.items()}
    mean_inv = sum(inv_counts.values()) / len(ALL_TASKS)
    task_weights = {
        task: round(inv_count / mean_inv, 4) for task, inv_count in inv_counts.items()
    }
    return pos_weights, task_weights


def compute_weights_and_metadata(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Compute per-property sample counts, positive rates, and loss weights for a train/valid/test split."""
    pos_weights, task_weights = compute_train_weights(train_df)
    properties_meta: dict[str, dict] = {}

    for task in ALL_TASKS:
        column_cleaned_train = train_df[task].dropna()
        column_cleaned_val = valid_df[task].dropna()
        column_cleaned_test = test_df[task].dropna()
        num_total = (
            len(column_cleaned_train)
            + len(column_cleaned_val)
            + len(column_cleaned_test)
        )
        num_pos_total = (
            int((column_cleaned_train == 1).sum())
            + int((column_cleaned_val == 1).sum())
            + int((column_cleaned_test == 1).sum())
        )

        properties_meta[task] = {
            "total_count": num_total,
            "train_count": len(column_cleaned_train),
            "valid_count": len(column_cleaned_val),
            "test_count": len(column_cleaned_test),
            "overall_pos_rate": round(num_pos_total / num_total, 4),
            "train_pos_rate": round(float(column_cleaned_train.mean()), 4),
            "valid_pos_rate": round(float(column_cleaned_val.mean()), 4),
            "test_pos_rate": round(float(column_cleaned_test.mean()), 4),
            "pos_weight": pos_weights[task],
            "task_weight": task_weights[task],
        }

    return {
        "tasks": ALL_TASKS,
        "num_tasks": len(ALL_TASKS),
        "split_counts": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
            "total": len(train_df) + len(valid_df) + len(test_df),
        },
        "pos_weights": pos_weights,
        "task_weights": task_weights,
        "properties": properties_meta,
    }


def preprocess_datasets(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    filter_zero_bonds: bool = True,
    frac: tuple[float, float, float] = (0.70, 0.10, 0.20),
    save_to_disk: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the deterministic fixed-split preprocessing pipeline (`70% train / 10% valid / 20% test`).

    Returns `(train_df, valid_df, test_df, metadata)`:
      - `train_df`, `valid_df`, `test_df` (`pd.DataFrame`): Tables with columns `['smiles'] + ALL_TASKS`
        (where each task column contains `0.0`, `1.0`, or `NaN`).
      - `metadata` (`dict`):
          - `"tasks"` (`list[str]`): List of the 18 endpoint names (`ALL_TASKS`).
          - `"num_tasks"` (`int`): `18`.
          - `"split_counts"` (`dict`): Molecule row counts `{"train", "valid", "test", "total"}`.
          - `"pos_weights"` (`dict[str, float]`): Within-task `N_neg / N_pos` weights on `train_df`.
          - `"task_weights"` (`dict[str, float]`): Across-task inverse sample-count weights on `train_df`.
          - `"properties"` (`dict[str, dict]`): Per-endpoint stats (`total_count`, `train_count`,
            `valid_count`, `test_count`, `overall_pos_rate`, `train_pos_rate`, `valid_pos_rate`,
            `test_pos_rate`, `pos_weight`, `task_weight`).
    """
    base_dir = Path(__file__).resolve().parent
    raw_path = Path(raw_dir) if raw_dir is not None else base_dir / "data" / "raw"
    out_path = (
        Path(output_dir) if output_dir is not None else base_dir / "data" / "processed"
    )

    merged_df = build_clean_merged_dataframe(
        raw_dir=raw_path,
        filter_zero_bonds=filter_zero_bonds,
    )

    bucket_ids = assign_buckets_to_smiles(merged_df, bucket_fractions=frac)
    model_cols = ["smiles"] + ALL_TASKS

    train_df = merged_df[bucket_ids == 0][model_cols].reset_index(drop=True)
    valid_df = merged_df[bucket_ids == 1][model_cols].reset_index(drop=True)
    test_df = merged_df[bucket_ids == 2][model_cols].reset_index(drop=True)

    metadata = compute_weights_and_metadata(train_df, valid_df, test_df)

    if save_to_disk:
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_path / "train.csv", index=False)
        valid_df.to_csv(out_path / "valid.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)
        with open(out_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return train_df, valid_df, test_df, metadata


def preprocess_cv_datasets(
    n_folds: int = 5,
    test_frac: float = 0.20,
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    filter_zero_bonds: bool = True,
    save_to_disk: bool = True,
) -> tuple[list[dict], pd.DataFrame, dict]:
    """
    Run the deterministic K-Fold cross-validation preprocessing pipeline.
    Holds out the identical 20% `test_df` as `preprocess_datasets()` and partitions the
    remaining 80% development set into `n_folds` balanced scaffold-disjoint folds.

    Returns `(folds, test_df, cv_metadata)`:
      - `folds` (`list[dict]`): List of `n_folds` dicts, each with keys:
          - `"fold"` (`int`): Fold index `0 .. n_folds - 1`.
          - `"train_df"` (`pd.DataFrame`): Training split (`~64%` of total, columns `['smiles'] + ALL_TASKS`).
          - `"valid_df"` (`pd.DataFrame`): Validation split (`~16%` of total, columns `['smiles'] + ALL_TASKS`).
          - `"pos_weights"` (`dict[str, float]`): Within-task `N_neg / N_pos` weights on fold's `train_df`.
          - `"task_weights"` (`dict[str, float]`): Across-task inverse sample-count weights on fold's `train_df`.
      - `test_df` (`pd.DataFrame`): Held-out 20% test split (columns `['smiles'] + ALL_TASKS`).
      - `cv_metadata` (`dict`):
          - `"tasks"` (`list[str]`): List of the 18 endpoint names (`ALL_TASKS`).
          - `"num_tasks"` (`int`): `18`.
          - `"n_folds"` (`int`): Number of cross-validation folds (`5`).
          - `"split_counts"` (`dict`): Molecule row counts `{"total", "dev", "test", "fold_valid"}`.
          - `"properties"` (`dict[str, dict]`): Per-endpoint stats (`total_count`, `dev_count`,
            `test_count`, `overall_pos_rate`, `test_pos_rate`, `fold_valid_counts`, `fold_valid_pos_rates`).
    """
    base_dir = Path(__file__).resolve().parent
    raw_path = Path(raw_dir) if raw_dir is not None else base_dir / "data" / "raw"
    out_path = (
        Path(output_dir) if output_dir is not None else base_dir / "data" / "processed"
    )

    merged_df = build_clean_merged_dataframe(
        raw_dir=raw_path,
        filter_zero_bonds=filter_zero_bonds,
    )

    fixed_buckets = assign_buckets_to_smiles(
        merged_df,
        bucket_fractions=(round(1.0 - test_frac - 0.10, 6), 0.10, test_frac),
    )
    model_cols = ["smiles"] + ALL_TASKS
    test_df = merged_df[fixed_buckets == 2][model_cols].reset_index(drop=True)

    full_for_folds_df = merged_df[fixed_buckets != 2].reset_index(drop=True)
    fold_ids = assign_buckets_to_smiles(
        full_for_folds_df,
        bucket_fractions=[1.0 / n_folds] * n_folds,
    )

    folds: list[dict] = []
    fold_val_counts: dict[int, int] = {}

    for k in range(n_folds):
        f_train_df = full_for_folds_df[fold_ids != k][model_cols].reset_index(drop=True)
        f_valid_df = full_for_folds_df[fold_ids == k][model_cols].reset_index(drop=True)
        pos_weight_k, task_weight_k = compute_train_weights(f_train_df)

        folds.append(
            {
                "fold": k,
                "train_df": f_train_df,
                "valid_df": f_valid_df,
                "pos_weights": pos_weight_k,
                "task_weights": task_weight_k,
            }
        )
        fold_val_counts[k] = len(f_valid_df)

    properties_meta: dict[str, dict] = {}
    for task in ALL_TASKS:
        col_total = merged_df[task].dropna()
        col_dev = full_for_folds_df[task].dropna()
        col_test = test_df[task].dropna()
        fold_cols = [f["valid_df"][task].dropna() for f in folds]

        properties_meta[task] = {
            "total_count": len(col_total),
            "dev_count": len(col_dev),
            "test_count": len(col_test),
            "overall_pos_rate": round(float(col_total.mean()), 4),
            "test_pos_rate": round(float(col_test.mean()), 4),
            "fold_valid_counts": [len(c) for c in fold_cols],
            "fold_valid_pos_rates": [round(float(c.mean()), 4) for c in fold_cols],
        }

    cv_metadata = {
        "tasks": ALL_TASKS,
        "num_tasks": len(ALL_TASKS),
        "n_folds": n_folds,
        "split_counts": {
            "total": len(merged_df),
            "dev": len(full_for_folds_df),
            "test": len(test_df),
            "fold_valid": fold_val_counts,
        },
        "properties": properties_meta,
    }

    if save_to_disk:
        cv_dir = out_path / "cv_folds"
        cv_dir.mkdir(parents=True, exist_ok=True)
        for f_dict in folds:
            k = f_dict["fold"]
            f_dict["train_df"].to_csv(cv_dir / f"fold_{k}_train.csv", index=False)
            f_dict["valid_df"].to_csv(cv_dir / f"fold_{k}_valid.csv", index=False)
        with open(cv_dir / "cv_metadata.json", "w") as f:
            json.dump(cv_metadata, f, indent=2)

    return folds, test_df, cv_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Tox21, CYP450, and hERG_Karim into fixed and 5-fold CV scaffold splits."
    )
    parser.add_argument(
        "--keep-zero-bonds",
        action="store_true",
        help="Keep molecules with zero bonds.",
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Perform cross-validation.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds.")

    args = parser.parse_args()

    if args.cross_validation:
        folds, cv_test_df, cv_meta = preprocess_cv_datasets(
            n_folds=args.n_folds,
            filter_zero_bonds=not args.keep_zero_bonds,
            save_to_disk=True,
        )
        print(
            f"{args.n_folds}-Fold CV: total={cv_meta['split_counts']['total']:,} "
            f"(dev={cv_meta['split_counts']['dev']:,}, val_folds={cv_meta['split_counts']['fold_valid']}, "
            f"test={cv_meta['split_counts']['test']:,})"
        )
    else:
        train_df, valid_df, test_df, meta = preprocess_datasets(
            filter_zero_bonds=not args.keep_zero_bonds,
            save_to_disk=True,
        )
        print(
            f"Fixed Split: total={meta['split_counts']['total']:,} "
            f"(train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,})"
        )


if __name__ == "__main__":
    main()
