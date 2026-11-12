"""Chemprop-specific dataset and DataLoader builder.

References:
- Official training notebook: https://github.com/chemprop/chemprop/blob/main/examples/training.ipynb
- Datapoints (`MoleculeDatapoint`): https://chemprop.readthedocs.io/en/latest/tutorial/python/data/datapoints.html
- Datasets (`MoleculeDataset`): https://chemprop.readthedocs.io/en/latest/tutorial/python/data/datasets.html
- Dataloaders (`build_dataloader`): https://chemprop.readthedocs.io/en/latest/tutorial/python/data/dataloaders.html
"""

from __future__ import annotations

import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from torch.utils.data import DataLoader

from gnn_data.preprocess import ALL_TASKS


def build_chemprop_loader(
    df: pd.DataFrame,
    tasks: list[str] = ALL_TASKS,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Convert a SMILES + targets DataFrame into a Chemprop DataLoader."""
    smis = df.loc[:, "smiles"].values
    ys = df.loc[:, tasks].values

    datapoints = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dataset = MoleculeDataset(datapoints, featurizer)
    return build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)

