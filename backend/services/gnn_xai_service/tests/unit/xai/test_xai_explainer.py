"""Unit tests for `IntegratedGradientsExplainer` axioms, step convergence, and input/output guards."""

import pytest
import torch
from rdkit import Chem

from src.exceptions import ModelInferenceError
from src.gnn_models.gnn_interface import GNNPredictor
from src.xai.integrated_gradients import IntegratedGradientsExplainer


class _LinearPredictorStub(GNNPredictor):
    """Differentiable linear predictor: target k scales atom-feature sum by `(k + 1) * (-1)^k`."""

    def __init__(self, num_targets: int = 2) -> None:
        self.num_targets = num_targets

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        total = atom_embeddings.sum()
        return torch.stack(
            [((-1.0) ** k) * (k + 1.0) * total for k in range(self.num_targets)]
        ).unsqueeze(0)


class _NonlinearPredictorStub(GNNPredictor):
    """Differentiable non-polynomial predictor for completeness, dummy-atom, and step convergence."""

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        t0 = (
            torch.exp(atom_embeddings[:, 0]) + torch.tanh(2.5 * atom_embeddings[:, 1])
        ).sum()
        t1 = (
            torch.sigmoid(4.0 * atom_embeddings[:, 0])
            * torch.exp(-1.5 * atom_embeddings[:, 1])
        ).sum()
        return torch.stack([t0, t1]).unsqueeze(0)


class _FaultyValuePredictorStub(GNNPredictor):
    """Predictor stub that injects a chosen non-finite scalar (`nan` or `inf`) into logits."""

    def __init__(self, faulty_scalar: float) -> None:
        self.faulty_scalar = faulty_scalar

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        return (atom_embeddings * self.faulty_scalar).sum().reshape(1, 1)


def test_computes_exact_per_atom_scores_for_linear_model() -> None:
    explainer = IntegratedGradientsExplainer(_LinearPredictorStub(3), steps=20)
    embeddings = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [-1.0, 0.5]], dtype=torch.float32
    )

    scores = torch.tensor(explainer.explain(embeddings, num_targets=3))

    expected = torch.tensor(
        [[3.0, 0.0, -0.5], [-6.0, 0.0, 1.0], [9.0, 0.0, -1.5]], dtype=torch.float32
    )
    torch.testing.assert_close(scores, expected, rtol=1e-4, atol=1e-5)


def test_satisfies_completeness_dummy_and_step_convergence_on_nonlinear_model() -> None:
    predictor = _NonlinearPredictorStub()
    embeddings = torch.tensor(
        [[1.8, -1.2], [0.0, 0.0], [-1.5, 1.4]], dtype=torch.float32
    )
    delta_f = (
        predictor.forward_from_embeddings(embeddings)
        - predictor.forward_from_embeddings(torch.zeros_like(embeddings))
    ).squeeze(0)

    coarse = torch.tensor(
        IntegratedGradientsExplainer(predictor, steps=2).explain(embeddings, 2)
    )
    fine = torch.tensor(
        IntegratedGradientsExplainer(predictor, steps=50).explain(embeddings, 2)
    )

    coarse_err = (coarse.sum(dim=1) - delta_f).abs().max().item()
    fine_err = (fine.sum(dim=1) - delta_f).abs().max().item()
    assert coarse_err > 1e-2
    assert fine_err < coarse_err
    torch.testing.assert_close(fine.sum(dim=1), delta_f, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        fine[:, 1], torch.zeros((2,), dtype=torch.float32), atol=1e-6, rtol=0.0
    )


def test_works_inside_torch_no_grad_and_preserves_input() -> None:
    explainer = IntegratedGradientsExplainer(_LinearPredictorStub(2), steps=15)
    embeddings = torch.tensor([[1.0, -0.5], [2.0, 0.25]], dtype=torch.float32)
    snapshot = embeddings.clone()

    scores_normal = explainer.explain(embeddings, num_targets=2)
    with torch.no_grad():
        scores_no_grad = explainer.explain(embeddings, num_targets=2)
    scores_grad = explainer.explain(embeddings.clone().requires_grad_(True), 2)

    assert scores_no_grad == scores_normal == scores_grad
    assert torch.equal(embeddings, snapshot) and embeddings.requires_grad is False


def test_preserves_dimensions_at_minimal_squeeze_boundary() -> None:
    explainer = IntegratedGradientsExplainer(_LinearPredictorStub(1), steps=1)

    scores = explainer.explain(torch.tensor([[2.5]], dtype=torch.float32), 1)

    assert len(scores) == 1 and len(scores[0]) == 1
    assert scores[0][0] == pytest.approx(2.5, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    ("bad_embeddings", "num_targets"),
    [(None, 2), (torch.ones(3), 2), (torch.ones((0, 2)), 2), (torch.ones((3, 2)), 0)],
)
def test_rejects_invalid_inputs(bad_embeddings: object, num_targets: int) -> None:
    explainer = IntegratedGradientsExplainer(_LinearPredictorStub(2), steps=5)

    with pytest.raises(ModelInferenceError):
        explainer.explain(bad_embeddings, num_targets=num_targets)  # type: ignore[arg-type]


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_rejects_non_finite_input_embeddings(non_finite: float) -> None:
    explainer = IntegratedGradientsExplainer(_LinearPredictorStub(1), steps=5)
    embeddings = torch.tensor([[non_finite, 1.0], [1.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ModelInferenceError):
        explainer.explain(embeddings, num_targets=1)


@pytest.mark.parametrize("faulty_scalar", [float("nan"), float("inf")])
def test_rejects_non_finite_output_attributions(faulty_scalar: float) -> None:
    explainer = IntegratedGradientsExplainer(
        _FaultyValuePredictorStub(faulty_scalar), steps=5
    )

    with pytest.raises(ModelInferenceError):
        explainer.explain(torch.ones((3, 2), dtype=torch.float32), num_targets=1)


@pytest.mark.parametrize("invalid_steps", [0, -5])
def test_rejects_non_positive_steps(invalid_steps: int) -> None:
    with pytest.raises(ValueError):
        IntegratedGradientsExplainer(
            predictor=_LinearPredictorStub(), steps=invalid_steps
        )
