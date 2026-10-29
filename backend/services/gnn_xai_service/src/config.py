#defione all of the prediction properties

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
    "hERG_Karim"
]

PREDICTIONS = TOX_21 + CYP450 + HERG
TASKS_NUMBER = len(PREDICTIONS)