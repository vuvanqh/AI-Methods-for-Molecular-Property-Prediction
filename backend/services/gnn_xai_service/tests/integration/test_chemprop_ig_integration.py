import pytest
import torch
from rdkit import Chem

from src.config import TASKS_NUMBER
from src.gnn_models.chemprop import Chemprop
from src.xai.integrated_gradients import IntegratedGradientsExplainer


def test_chemprop_and_ig_satisfy_completeness_axiom_with_production_steps(
    real_chemprop: Chemprop,
    production_ig_explainer: IntegratedGradientsExplainer,
) -> None:
    """IG attributions sum to F(x) - F(0) per task without mutating model weights, grads, or eval mode."""
    mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
    assert mol is not None

    state_before = {
        name: tensor.detach().clone()
        for name, tensor in real_chemprop.model.state_dict().items()
    }

    embeddings = real_chemprop.predict(mol)["atom_embeddings"]
    attributions = production_ig_explainer.explain(embeddings, TASKS_NUMBER)

    with torch.no_grad():
        fx = real_chemprop.forward_from_embeddings(embeddings).squeeze(0)
        f0 = real_chemprop.forward_from_embeddings(
            torch.zeros_like(embeddings)
        ).squeeze(0)
    expected_deltas = (fx - f0).tolist()

    assert len(attributions) == TASKS_NUMBER
    attribution_sums = [sum(scores) for scores in attributions]
    assert all(abs(total) > 1e-6 for total in attribution_sums)
    for task_idx in range(TASKS_NUMBER):
        assert attribution_sums[task_idx] == pytest.approx(
            expected_deltas[task_idx], rel=1e-2, abs=5e-5
        )

    assert real_chemprop.model.training is False
    assert all(param.grad is None for param in real_chemprop.model.parameters())
    assert all(
        torch.equal(tensor, state_before[name])
        for name, tensor in real_chemprop.model.state_dict().items()
    )


def test_chemprop_predict_matches_native_mpnn_forward(real_chemprop: Chemprop) -> None:
    """Manual `message_passing -> forward_from_embeddings -> sigmoid` matches native `MPNN.forward`."""
    mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
    assert mol is not None

    probs = real_chemprop.predict(mol)["probabilities"].detach()
    batch_graph = real_chemprop.build_batch_mol_graph(mol)
    with torch.no_grad():
        native_probs = real_chemprop.model(batch_graph).squeeze(0)

    assert probs.shape == (TASKS_NUMBER,)
    torch.testing.assert_close(probs, native_probs, rtol=1e-6, atol=1e-6)


def test_chemprop_and_ig_are_permutation_equivariant_under_atom_renumbering(
    real_chemprop: Chemprop,
    fast_ig_explainer: IntegratedGradientsExplainer,
) -> None:
    """Reversing atom indices preserves chiral predictions and permutes IG attributions identically."""
    mol_orig = Chem.MolFromSmiles("C[C@H](O)Cl")
    assert mol_orig is not None

    perm = list(reversed(range(mol_orig.GetNumAtoms())))
    mol_renum = Chem.RenumberAtoms(mol_orig, perm)

    pred_orig = real_chemprop.predict(mol_orig)
    pred_renum = real_chemprop.predict(mol_renum)

    torch.testing.assert_close(
        pred_orig["probabilities"], pred_renum["probabilities"], rtol=1e-4, atol=1e-6
    )
    assert torch.equal(pred_orig["predictions"], pred_renum["predictions"])

    attr_orig = torch.tensor(
        fast_ig_explainer.explain(pred_orig["atom_embeddings"], TASKS_NUMBER),
        dtype=torch.float32,
    )
    attr_renum = torch.tensor(
        fast_ig_explainer.explain(pred_renum["atom_embeddings"], TASKS_NUMBER),
        dtype=torch.float32,
    )
    torch.testing.assert_close(attr_renum, attr_orig[:, perm], rtol=1e-4, atol=1e-6)


def test_chemprop_and_ig_produce_nontrivial_distinct_outputs_for_different_structures(
    real_chemprop: Chemprop,
    fast_ig_explainer: IntegratedGradientsExplainer,
) -> None:
    """Distinct structures yield distinct probabilities and non-degenerate per-atom IG spread on every task."""
    mol_ethanol = Chem.MolFromSmiles("CCO")
    mol_nitro = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
    assert mol_ethanol is not None and mol_nitro is not None

    probs_ethanol = real_chemprop.predict(mol_ethanol)["probabilities"].detach()
    pred_nitro = real_chemprop.predict(mol_nitro)
    probs_nitro = pred_nitro["probabilities"].detach()

    assert not torch.allclose(probs_ethanol, probs_nitro, atol=1e-5)

    nitro_attr = fast_ig_explainer.explain(pred_nitro["atom_embeddings"], TASKS_NUMBER)
    spreads = [max(scores) - min(scores) for scores in nitro_attr]
    assert all(spread > 1e-5 for spread in spreads)
