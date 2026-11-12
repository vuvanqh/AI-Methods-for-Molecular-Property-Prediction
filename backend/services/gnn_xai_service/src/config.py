"""Service configuration constants: target endpoints (Tox21, CYP450, hERG), per-endpoint thresholds, device, and IG steps."""

import torch

# Prediction target groups

TOX_21 = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
CYP450 = [
    "CYP2C19",
    "CYP2D6",
    "CYP3A4",
    "CYP1A2",
    "CYP2C9",
]
HERG = [
    "hERG_Karim",
]

TASKS = TOX_21 + CYP450 + HERG
TASKS_NUMBER = len(TASKS)

# Per-endpoint binary classification thresholds mapped by task name
BINARY_PREDICTION_THRESHOLDS: dict[str, float] = {
    # Tox21
    "NR-AR": 0.5,
    "NR-AR-LBD": 0.5,
    "NR-AhR": 0.5,
    "NR-Aromatase": 0.5,
    "NR-ER": 0.5,
    "NR-ER-LBD": 0.5,
    "NR-PPAR-gamma": 0.5,
    "SR-ARE": 0.5,
    "SR-ATAD5": 0.5,
    "SR-HSE": 0.5,
    "SR-MMP": 0.5,
    "SR-p53": 0.5,
    # CYP450
    "CYP2C19": 0.5,
    "CYP2D6": 0.5,
    "CYP3A4": 0.5,
    "CYP1A2": 0.5,
    "CYP2C9": 0.5,
    # hERG
    "hERG_Karim": 0.5,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IG_STEPS = 100
