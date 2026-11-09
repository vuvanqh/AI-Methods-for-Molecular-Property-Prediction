import pytest
import torch

from src.config import BINARY_PREDICTION_THRESHOLD, IG_STEPS, TASKS_NUMBER
from src.gnn_models.chemprop import Chemprop
from src.service import MolecularInferenceService
from src.xai.integrated_gradients import IntegratedGradientsExplainer


def pytest_configure(config: pytest.Config) -> None:
    """
    Built-in pytest startup hook that registers custom `@pytest.mark.<name>` decorators
    so pytest recognizes `-m fast_rdkit` / `-m random_weights` without emitting
    `PytestUnknownMarkWarning`.
    """
    config.addinivalue_line(
        "markers",
        "fast_rdkit: Fast unit tests that do not initialize PyTorch/Chemprop models.",
    )
    config.addinivalue_line(
        "markers",
        "random_weights: Tests that run with a deterministically seeded Chemprop model.",
    )


@pytest.fixture(scope="module")
def chemprop_predictor() -> Chemprop:
    """
    Shared, deterministically initialized Chemprop instance on CPU.
    Note: `Chemprop.__init__` already sets `self.model.eval()` and moves the model to `device`.
    """
    torch.manual_seed(42)
    return Chemprop(
        n_tasks=TASKS_NUMBER,
        device=torch.device("cpu"),
        threshold=BINARY_PREDICTION_THRESHOLD,
    )


@pytest.fixture(scope="module")
def ig_explainer(chemprop_predictor: Chemprop) -> IntegratedGradientsExplainer:
    """Shared IntegratedGradientsExplainer bound to the deterministic Chemprop predictor."""
    return IntegratedGradientsExplainer(
        predictor=chemprop_predictor,
        steps=IG_STEPS,
    )


@pytest.fixture(scope="module")
def inference_service() -> MolecularInferenceService:
    """Shared full MolecularInferenceService instance with deterministic model weights."""
    torch.manual_seed(42)
    return MolecularInferenceService()


@pytest.fixture
def uninitialized_service() -> MolecularInferenceService:
    """
    Lightweight MolecularInferenceService bypassing `__init__` (no PyTorch model allocation)
    so input validation and helper unit tests run instantaneously.
    """
    return MolecularInferenceService.__new__(MolecularInferenceService)
