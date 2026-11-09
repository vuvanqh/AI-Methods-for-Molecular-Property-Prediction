from unittest.mock import patch
import pytest
import torch
from rdkit import Chem

from src.config import BINARY_PREDICTION_THRESHOLD, TASKS_NUMBER
from src.exceptions import ModelInferenceError
from src.gnn_models.chemprop import Chemprop


# =====================================================================
# 1. Chemprop.predict — Happy Paths & Invariance
# =====================================================================


@pytest.mark.random_weights
def test_chemprop_predict_returns_expected_keys_shapes_and_thresholded_predictions(
    chemprop_predictor: Chemprop,
) -> None:
    mol = Chem.MolFromSmiles("CC(C)O")
    assert mol is not None

    result = chemprop_predictor.predict(mol)

    assert set(result.keys()) == {"probabilities", "predictions", "atom_embeddings", "mol_graph_index"}
    probs = result["probabilities"]
    preds = result["predictions"]
    emb = result["atom_embeddings"]
    batch_idx = result["mol_graph_index"]

    assert probs.shape == (TASKS_NUMBER,)
    assert preds.shape == (TASKS_NUMBER,)
    assert emb.ndim == 2 and emb.shape[0] == 4
    assert batch_idx.shape == (4,)
    assert torch.equal(preds, (probs > BINARY_PREDICTION_THRESHOLD).int())


@pytest.mark.random_weights
@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccccc1OC(C)=O"),
        ("F/C=C/F", r"F\C=C\F"),
        ("C[C@H](O)Cl", "Cl[C@H](C)O"),
    ],
)
def test_chemprop_prediction_invariance_across_equivalent_smiles(
    chemprop_predictor: Chemprop,
    smiles_a: str,
    smiles_b: str,
) -> None:
    """
    Verifies that `Chemprop.predict` produces identical 18-endpoint probability vectors
    across equivalent SMILES representations when evaluated on the same model instance.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    assert mol_a is not None and mol_b is not None

    with torch.no_grad():
        result_a = chemprop_predictor.predict(mol_a)
        result_b = chemprop_predictor.predict(mol_b)

    torch.testing.assert_close(
        result_a["probabilities"],
        result_b["probabilities"],
        rtol=1e-5,
        atol=1e-6,
    )


# =====================================================================
# 2. Chemprop — Unhappy Paths (Guardrail Assertions)
# =====================================================================


@pytest.mark.random_weights
def test_chemprop_predict_rejects_none_and_zero_atom_molecules(
    chemprop_predictor: Chemprop,
) -> None:
    with pytest.raises(ModelInferenceError, match="Cannot run Chemprop prediction on None molecule"):
        chemprop_predictor.predict(None)

    with pytest.raises(ModelInferenceError, match="Molecule contains 0 atoms"):
        chemprop_predictor.predict(Chem.Mol())


@pytest.mark.random_weights
def test_chemprop_build_batch_mol_graph_raises_when_featurizer_returns_none(
    chemprop_predictor: Chemprop,
) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    with patch.object(chemprop_predictor, "featurizer", return_value=None):
        with pytest.raises(ModelInferenceError, match="Failed to featurize molecule into MolGraph"):
            chemprop_predictor.build_batch_mol_graph(mol)


@pytest.mark.random_weights
def test_chemprop_predict_embedding_and_logit_invariant_checks(
    chemprop_predictor: Chemprop,
) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    # 1. 1D embeddings tensor
    with patch.object(chemprop_predictor, "get_embeddings", return_value=torch.ones(3)):
        with pytest.raises(ModelInferenceError, match="Expected 2D atom embeddings"):
            chemprop_predictor.predict(mol)

    # 2. Embeddings row count != num_atoms
    with patch.object(chemprop_predictor, "get_embeddings", return_value=torch.ones((5, 300))):
        with pytest.raises(ModelInferenceError, match="Atom count mismatch in embeddings"):
            chemprop_predictor.predict(mol)

    # 3. NaN in embeddings
    nan_emb = torch.ones((3, 300))
    nan_emb[1, 0] = float("nan")
    with patch.object(chemprop_predictor, "get_embeddings", return_value=nan_emb):
        with pytest.raises(ModelInferenceError, match="NaN or Inf values detected in atom embeddings"):
            chemprop_predictor.predict(mol)

    # 4. Wrong logits shape
    with patch.object(chemprop_predictor, "forward_from_embeddings", return_value=torch.zeros((2, TASKS_NUMBER))):
        with pytest.raises(ModelInferenceError, match="Expected logits of shape"):
            chemprop_predictor.predict(mol)

    # 5. Inf in logits
    inf_logits = torch.zeros((1, TASKS_NUMBER))
    inf_logits[0, 0] = float("inf")
    with patch.object(chemprop_predictor, "forward_from_embeddings", return_value=inf_logits):
        with pytest.raises(ModelInferenceError, match="NaN or Inf detected in model output logits"):
            chemprop_predictor.predict(mol)
