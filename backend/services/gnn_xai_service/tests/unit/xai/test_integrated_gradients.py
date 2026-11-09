from unittest.mock import MagicMock, patch
import pytest
import torch
from rdkit import Chem

from src.config import TASKS_NUMBER
from src.exceptions import ModelInferenceError
from src.gnn_models.chemprop import Chemprop
from src.xai.integrated_gradients import IntegratedGradientsExplainer


# =====================================================================
# 1. IntegratedGradientsExplainer.explain — Happy Paths
# =====================================================================


@pytest.mark.random_weights
@pytest.mark.parametrize(
    "smiles",
    ["CCO", "CC(C)O", "c1ccccc1", "[2H]CCO"],
)
def test_ig_scores_count_shape_and_finiteness_match_chemprop_graph_atoms(
    chemprop_predictor: Chemprop,
    smiles: str,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    explainer = IntegratedGradientsExplainer(predictor=chemprop_predictor, steps=20)
    pred = chemprop_predictor.predict(mol)

    expected_num_atoms = mol.GetNumAtoms()
    raw_scores = explainer.explain(
        atom_embeddings=pred["atom_embeddings"],
        mol_graph_index=pred["mol_graph_index"],
        num_targets=TASKS_NUMBER,
    )

    assert isinstance(raw_scores, list) and len(raw_scores) == TASKS_NUMBER
    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    assert scores_tensor.shape == (TASKS_NUMBER, expected_num_atoms)
    assert torch.isfinite(scores_tensor).all()


@pytest.mark.random_weights
@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
    ],
)
def test_ig_scores_match_when_atoms_are_mapped_across_equivalent_smiles(
    chemprop_predictor: Chemprop,
    smiles_a: str,
    smiles_b: str,
) -> None:
    """
    Verifies that `IntegratedGradientsExplainer.explain` assigns identical attribution
    scores to corresponding physical atoms regardless of atom-index permutation.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    assert mol_a is not None and mol_b is not None

    match = list(mol_b.GetSubstructMatch(mol_a, useChirality=True))
    explainer = IntegratedGradientsExplainer(predictor=chemprop_predictor, steps=25)

    res_a = chemprop_predictor.predict(mol_a)
    res_b = chemprop_predictor.predict(mol_b)

    scores_a = torch.tensor(
        explainer.explain(res_a["atom_embeddings"], res_a["mol_graph_index"], num_targets=TASKS_NUMBER),
        dtype=torch.float32,
    )
    scores_b = torch.tensor(
        explainer.explain(res_b["atom_embeddings"], res_b["mol_graph_index"], num_targets=TASKS_NUMBER),
        dtype=torch.float32,
    )

    torch.testing.assert_close(scores_a, scores_b[:, match], rtol=1e-5, atol=1e-6)


# =====================================================================
# 2. IntegratedGradientsExplainer.explain — Unhappy Paths (Guardrails)
# =====================================================================


@pytest.mark.fast_rdkit
def test_ig_explainer_input_validation_guardrails() -> None:
    dummy_predictor = MagicMock()
    explainer = IntegratedGradientsExplainer(predictor=dummy_predictor, steps=5)

    # 1. None or non-Tensor atom_embeddings
    with pytest.raises(ModelInferenceError, match="atom_embeddings must be a valid torch.Tensor"):
        explainer.explain(None, torch.zeros(3, dtype=torch.long), num_targets=2)

    with pytest.raises(ModelInferenceError, match="atom_embeddings must be a valid torch.Tensor"):
        explainer.explain([[1.0, 2.0]], torch.zeros(1, dtype=torch.long), num_targets=2)

    # 2. Non-2D atom_embeddings
    with pytest.raises(ModelInferenceError, match="atom_embeddings must be 2D"):
        explainer.explain(torch.ones(3), torch.zeros(3, dtype=torch.long), num_targets=2)

    with pytest.raises(ModelInferenceError, match="atom_embeddings must be 2D"):
        explainer.explain(torch.ones((1, 3, 8)), torch.zeros(3, dtype=torch.long), num_targets=2)

    # 3. 0 atoms
    with pytest.raises(ModelInferenceError, match="Cannot explain molecule with 0 atom embeddings"):
        explainer.explain(torch.ones((0, 8)), torch.zeros(0, dtype=torch.long), num_targets=2)

    # 4. mol_graph_index None or length mismatch
    with pytest.raises(ModelInferenceError, match="mol_graph_index shape mismatch"):
        explainer.explain(torch.ones((3, 8)), None, num_targets=2)

    with pytest.raises(ModelInferenceError, match="mol_graph_index shape mismatch"):
        explainer.explain(torch.ones((3, 8)), torch.zeros(5, dtype=torch.long), num_targets=2)

    # 5. Non-positive num_targets
    with pytest.raises(ModelInferenceError, match="num_targets must be positive"):
        explainer.explain(torch.ones((3, 8)), torch.zeros(3, dtype=torch.long), num_targets=0)


@pytest.mark.fast_rdkit
def test_ig_explainer_output_shape_and_nan_guardrails() -> None:
    dummy_predictor = MagicMock()
    explainer = IntegratedGradientsExplainer(predictor=dummy_predictor, steps=5)

    # 1. Captum returns wrong attribution tensor shape
    with patch("src.xai.integrated_gradients.IntegratedGradients") as mock_ig_cls:
        mock_ig_cls.return_value.attribute.return_value = torch.ones((1, 5, 8))  # 5 atoms instead of 3
        with pytest.raises(ModelInferenceError, match="Captum attribution shape"):
            explainer.explain(torch.ones((3, 8)), torch.zeros(3, dtype=torch.long), num_targets=1)

    # 2. Captum returns NaN attribution
    with patch("src.xai.integrated_gradients.IntegratedGradients") as mock_ig_cls:
        nan_attr = torch.ones((1, 3, 8))
        nan_attr[0, 1, 2] = float("nan")
        mock_ig_cls.return_value.attribute.return_value = nan_attr
        with pytest.raises(ModelInferenceError, match="NaN or Inf detected in attribution scores"):
            explainer.explain(torch.ones((3, 8)), torch.zeros(3, dtype=torch.long), num_targets=1)
