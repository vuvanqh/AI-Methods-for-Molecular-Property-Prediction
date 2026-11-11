import math

import pytest
import torch
from rdkit import Chem

from src.config import TASKS, TASKS_NUMBER
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.gnn_models.gnn_interface import GNNPredictor
from src.model import GNNXAIInferenceResponse
from src.service import MolecularInferenceService
from src.xai.xai_interface import XAIExplainer


class _StubPredictor(GNNPredictor):
    """Stub `GNNPredictor` dependency for testing `MolecularInferenceService` in isolation."""

    def __init__(self, custom_output: dict[str, torch.Tensor] | None = None) -> None:
        self._custom_output = custom_output
        self.predict_call_count = 0

    def predict(self, mol: Chem.Mol) -> dict[str, torch.Tensor]:
        self.predict_call_count += 1
        if self._custom_output is not None:
            return self._custom_output

        atom_embeddings = torch.tensor(
            [
                [float(atom.GetAtomicNum()), float(atom.GetDegree())]
                for atom in mol.GetAtoms()
            ],
            dtype=torch.float32,
        )
        probabilities = torch.linspace(
            0.1, 0.9, steps=TASKS_NUMBER, dtype=torch.float32
        )
        predictions = (probabilities > 0.5).int()
        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "atom_embeddings": atom_embeddings,
        }

    def forward_from_embeddings(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, TASKS_NUMBER), dtype=torch.float32)


class _StubExplainer(XAIExplainer):
    """Stub `XAIExplainer` dependency for testing `MolecularInferenceService` in isolation."""

    def __init__(self, custom_scores: list[list[float]] | None = None) -> None:
        self._custom_scores = custom_scores
        self.explain_call_count = 0

    def explain(
        self, atom_embeddings: torch.Tensor, num_targets: int
    ) -> list[list[float]]:
        self.explain_call_count += 1
        if self._custom_scores is not None:
            return self._custom_scores

        atom_sums = atom_embeddings.sum(dim=-1).tolist()
        return [
            [float(val) + 0.01 * idx for val in atom_sums] for idx in range(num_targets)
        ]


@pytest.fixture
def service_and_stubs() -> tuple[
    MolecularInferenceService, _StubPredictor, _StubExplainer
]:
    predictor = _StubPredictor()
    explainer = _StubExplainer()
    return (
        MolecularInferenceService(predictor=predictor, explainer=explainer),
        predictor,
        explainer,
    )


@pytest.mark.parametrize(
    ("smiles", "expected_graph_atoms"),
    [
        ("CCO", 3),
        ("CC(C)O", 4),
        ("c1ccccc1", 6),
        ("C=CC(=O)O", 5),
        ("CC(=O)[O-]", 4),
        ("C[C@H](O)Cl", 4),
        ("F/C=C/F", 4),
        ("[2H]CCO", 4),
        ("CCO.[Na+]", 4),
    ],
)
def test_predict_and_explain_returns_valid_response_for_bonded_molecules(
    service_and_stubs: tuple[MolecularInferenceService, _StubPredictor, _StubExplainer],
    smiles: str,
    expected_graph_atoms: int,
) -> None:
    service, predictor, explainer = service_and_stubs
    result = service.predict_and_explain(smiles)

    assert predictor.predict_call_count == 1
    assert explainer.explain_call_count == 1

    validated = GNNXAIInferenceResponse(**result)
    assert set(validated.predictions.keys()) == set(TASKS)
    assert set(validated.probabilities.keys()) == set(TASKS)
    assert set(validated.atomscores.keys()) == set(TASKS)
    assert len(validated.predictions) == TASKS_NUMBER
    assert "V2000" in validated.molblock and "M  END" in validated.molblock

    parsed_molblock = Chem.MolFromMolBlock(validated.molblock, removeHs=False)
    assert parsed_molblock is not None
    total_atoms_with_h = parsed_molblock.GetNumAtoms()
    assert total_atoms_with_h > expected_graph_atoms

    for task in TASKS:
        prob = validated.probabilities[task]
        assert isinstance(validated.predictions[task], bool)
        assert 0.0 <= prob <= 1.0
        assert math.isfinite(prob)

        scores_for_task = validated.atomscores[task]
        assert list(scores_for_task.keys()) == list(range(total_atoms_with_h))
        for idx in range(expected_graph_atoms):
            assert isinstance(scores_for_task[idx], float)
            assert math.isfinite(scores_for_task[idx])
        for idx in range(expected_graph_atoms, total_atoms_with_h):
            assert scores_for_task[idx] is None


@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CCO", "[CH3][CH2][OH]"),
        ("CCO", "[H]C([H])([H])C([H])([H])O"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("C[C@H](O)Cl", "O[C@@H](C)Cl"),
    ],
)
def test_predict_and_explain_is_invariant_across_equivalent_smiles(
    service_and_stubs: tuple[MolecularInferenceService, _StubPredictor, _StubExplainer],
    smiles_a: str,
    smiles_b: str,
) -> None:
    service, _, _ = service_and_stubs
    assert service.predict_and_explain(smiles_a) == service.predict_and_explain(
        smiles_b
    )


@pytest.mark.parametrize(
    "empty_or_whitespace",
    [
        "",
        " ",
        "    ",
        "\t",
        "\n",
        " \t\r\n ",
        None,
    ],
)
def test_predict_and_explain_rejects_empty_or_whitespace_without_calling_model(
    empty_or_whitespace: str | None,
    service_and_stubs: tuple[MolecularInferenceService, _StubPredictor, _StubExplainer],
) -> None:
    service, predictor, explainer = service_and_stubs
    with pytest.raises(MoleculeValidationError):
        service.predict_and_explain(empty_or_whitespace)

    assert predictor.predict_call_count == 0
    assert explainer.explain_call_count == 0


@pytest.mark.parametrize(
    "invalid_smiles",
    [
        "not_a_smiles",
        "C(C",
        "c1cccc1",
        "C1CC",
        "C(C)(C)(C)(C)C",
        "F(F)F",
    ],
)
def test_predict_and_explain_rejects_unparseable_smiles_without_calling_model(
    invalid_smiles: str,
    service_and_stubs: tuple[MolecularInferenceService, _StubPredictor, _StubExplainer],
) -> None:
    service, predictor, explainer = service_and_stubs
    with pytest.raises(MoleculeValidationError):
        service.predict_and_explain(invalid_smiles)

    assert predictor.predict_call_count == 0
    assert explainer.explain_call_count == 0


@pytest.mark.parametrize(
    "zero_bond_smiles",
    [
        "C",
        "O",
        "N",
        "S",
        "[Na+]",
        "[Cl-]",
        "[Fe+2]",
        "[He]",
        "[2H]",
        "[H]C([H])([H])[H]",
        "[H]O[H]",
        "[Na+].[Cl-]",
        "O.O",
    ],
)
def test_predict_and_explain_rejects_zero_bond_molecules_before_entering_model(
    zero_bond_smiles: str,
    service_and_stubs: tuple[MolecularInferenceService, _StubPredictor, _StubExplainer],
) -> None:
    service, predictor, explainer = service_and_stubs
    with pytest.raises(MoleculeValidationError):
        service.predict_and_explain(zero_bond_smiles)

    assert predictor.predict_call_count == 0
    assert explainer.explain_call_count == 0


@pytest.mark.parametrize(
    "faulty_prediction",
    [
        # Missing 'probabilities'
        {
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # Wrong probabilities length
        {
            "probabilities": torch.full((3,), 0.5),
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # Out-of-range probability > 1.0
        {
            "probabilities": torch.full((TASKS_NUMBER,), 1.5),
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # Negative probability < 0.0
        {
            "probabilities": torch.full((TASKS_NUMBER,), -0.1),
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # NaN probability
        {
            "probabilities": torch.full((TASKS_NUMBER,), float("nan")),
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # Missing 'predictions'
        {
            "probabilities": torch.full((TASKS_NUMBER,), 0.5),
            "atom_embeddings": torch.ones((3, 2)),
        },
        # Wrong predictions length
        {
            "probabilities": torch.full((TASKS_NUMBER,), 0.5),
            "predictions": torch.zeros((2,), dtype=torch.int32),
            "atom_embeddings": torch.ones((3, 2)),
        },
    ],
)
def test_service_raises_model_inference_error_on_malformed_predictor_outputs(
    faulty_prediction: dict[str, torch.Tensor],
) -> None:
    service = MolecularInferenceService(
        predictor=_StubPredictor(custom_output=faulty_prediction),
        explainer=_StubExplainer(),
    )
    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")


@pytest.mark.parametrize(
    "faulty_scores",
    [
        # Wrong number of targets
        [[0.1, 0.2, 0.3]],
        # Wrong number of atoms for 3-atom CCO
        [[0.1, 0.2] for _ in range(TASKS_NUMBER)],
    ],
)
def test_service_raises_model_inference_error_on_malformed_explainer_outputs(
    faulty_scores: list[list[float]],
) -> None:
    service = MolecularInferenceService(
        predictor=_StubPredictor(),
        explainer=_StubExplainer(custom_scores=faulty_scores),
    )
    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")
