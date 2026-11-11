"""Service configuration constants: target endpoints (Tox21, CYP450, hERG), device, threshold, and IG steps."""

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
HERG = ["hERG_Karim"]

TASKS = TOX_21 + CYP450 + HERG
TASKS_NUMBER = len(TASKS)

BINARY_PREDICTION_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IG_STEPS = 100
