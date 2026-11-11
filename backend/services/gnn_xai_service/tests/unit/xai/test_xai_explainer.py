import pytest
import torch
from rdkit import Chem

from src.exceptions import ModelInferenceError
from src.gnn_models.gnn_interface import GNNPredictor
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.xai_interface import XAIExplainer


class _StubPredictor(GNNPredictor):
    """Simple `GNNPredictor` stub: target 0 sums atom features (1x), target 1 scales them by -2x."""

    def __init__(self, return_nan: bool = False) -> None:
        self.return_nan = return_nan

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if self.return_nan:
            return (atom_embeddings * float("nan")).sum().reshape(1, 1)

        total = atom_embeddings.sum()
        return torch.stack([total, -2.0 * total]).unsqueeze(0)


def test_xai_explainer_interface_cannot_be_instantiated_without_explain() -> None:
    class IncompleteExplainer(XAIExplainer):
        pass

    with pytest.raises(TypeError):
        IncompleteExplainer()


def test_ig_explainer_implements_xai_explainer_interface() -> None:
    explainer = IntegratedGradientsExplainer(predictor=_StubPredictor(), steps=10)
    assert isinstance(explainer, XAIExplainer)


def test_ig_explainer_zero_embeddings_produce_zero_attributions() -> None:
    explainer = IntegratedGradientsExplainer(predictor=_StubPredictor(), steps=10)
    zero_embeddings = torch.zeros((3, 2), dtype=torch.float32)

    scores = torch.tensor(
        explainer.explain(atom_embeddings=zero_embeddings, num_targets=2)
    )

    torch.testing.assert_close(scores, torch.zeros((2, 3), dtype=torch.float32))


def test_ig_explainer_computes_expected_per_atom_scores() -> None:
    explainer = IntegratedGradientsExplainer(predictor=_StubPredictor(), steps=20)
    # Atom 0 sum = 3.0, Atom 1 sum = 0.0, Atom 2 sum = -0.5
    embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [0.0, 0.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )

    scores = torch.tensor(explainer.explain(atom_embeddings=embeddings, num_targets=2))

    expected = torch.tensor(
        [
            [3.0, 0.0, -0.5],  # target 0: 1x atom sum
            [-6.0, 0.0, 1.0],  # target 1: -2x atom sum
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(scores, expected, rtol=1e-4, atol=1e-5)


def test_ig_explainer_preserves_atom_order_when_rows_are_swapped() -> None:
    explainer = IntegratedGradientsExplainer(predictor=_StubPredictor(), steps=20)
    embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [0.0, 0.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )

    scores = torch.tensor(explainer.explain(atom_embeddings=embeddings, num_targets=2))
    swapped_order = [2, 0, 1]
    swapped_scores = torch.tensor(
        explainer.explain(atom_embeddings=embeddings[swapped_order], num_targets=2)
    )

    torch.testing.assert_close(
        swapped_scores, scores[:, swapped_order], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    ("bad_embeddings", "num_targets"),
    [
        (None, 2),
        ([[1.0, 2.0]], 2),
        (torch.ones(3), 2),
        (torch.ones((1, 3, 2)), 2),
        (torch.ones((0, 2)), 2),
        (torch.ones((3, 2)), 0),
        (torch.ones((3, 2)), -1),
    ],
)
def test_ig_explainer_rejects_invalid_inputs(
    bad_embeddings: object,
    num_targets: int,
) -> None:
    explainer = IntegratedGradientsExplainer(predictor=_StubPredictor(), steps=5)
    with pytest.raises(ModelInferenceError):
        explainer.explain(bad_embeddings, num_targets=num_targets)


def test_ig_explainer_raises_model_inference_error_on_nan_attributions() -> None:
    explainer = IntegratedGradientsExplainer(
        predictor=_StubPredictor(return_nan=True), steps=5
    )
    with pytest.raises(ModelInferenceError):
        explainer.explain(torch.ones((3, 2), dtype=torch.float32), num_targets=1)
