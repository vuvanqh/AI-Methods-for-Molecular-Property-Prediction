from types import SimpleNamespace
from typing import Any

import pytest

from src.rag.augmentation import prompt_builder
from src.rag.augmentation.prompt_builder import build_prompt
from src.rag.result_types.molecular_evidence import MolecularEvidence
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from tests.unit.conftest import make_evidence, make_neighbour

DEFAULT_MOLECULE = {"smiles": "CCCN", "formula": "C3H9N", "molar_mass": 59.11}
EMPTY_GNN = {"predictions": {}, "probabilities": {}, "atomscores": {}}


@pytest.fixture(autouse=True)
def prompt_limits(monkeypatch: pytest.MonkeyPatch):
    """Pin deterministic prompt budget defaults so `.env` never flakes unit tests."""

    def _set(
        *,
        max_evidence: int = 10,
        abstract_max_chars: int = 500,
        top_attributed_atoms: int = 3,
    ) -> None:
        monkeypatch.setattr(
            prompt_builder,
            "promptBuilderSettings",
            SimpleNamespace(
                max_evidence_per_molecule=max_evidence,
                abstract_max_chars=abstract_max_chars,
                top_attributed_atoms=top_attributed_atoms,
            ),
        )

    _set()
    return _set


def _ev(tid: str, name: str | None = None, **kw: Any) -> MolecularEvidence:
    return make_evidence(target_identifier=tid, target_name=name or tid, **kw)


def _build(
    *,
    smiles: str = "CCCN",
    gnn_data: dict | None = None,
    neighbours: list[RetrievedMolecule] | None = None,
    evidence: list[list[MolecularEvidence]] | None = None,
    unavailable: set[str] | None = None,
) -> str:
    nbs = neighbours or []
    ev = evidence if evidence is not None else [[] for _ in range(len(nbs) + 1)]
    return build_prompt(
        smiles,
        DEFAULT_MOLECULE,
        gnn_data or EMPTY_GNN,
        nbs,
        ev,
        unavailable or set(),
    )


def test_evidence_potency_ranking_and_zero_value_handling() -> None:
    rows = [
        _ev("T_NEG", log_potency=-1.0, value=1.0, units="nM"),
        _ev("T_UM", log_potency=None, value=10.0, units="uM"),
        # log_potency=0.0 is falsy and must NOT fall through to 1 nM (rank 9.0)
        _ev("T_ZERO_LP", log_potency=0.0, value=1.0, units="nM"),
        _ev("T_ZERO_VAL", log_potency=None, value=0.0, units="nM"),
        _ev("T_NM_WEAK", log_potency=None, value=10000.0, units="nM"),
        _ev("T_NM_STRONG", log_potency=None, value=10.0, units="nM"),
        _ev("T_LP_HIGH", log_potency=8.5, value=3.0, units="nM"),
    ]

    prompt = _build(evidence=[rows])

    assert (
        prompt.index("T_LP_HIGH")
        < prompt.index("T_NM_STRONG")
        < prompt.index("T_NM_WEAK")
        < prompt.index("T_ZERO_LP")
        < prompt.index("T_NEG")
    )
    assert prompt.index("T_NM_WEAK") < prompt.index("T_UM")
    assert "T_ZERO_VAL" in prompt and "0 nM" in prompt and "-log10[M] 0" in prompt


def test_dedup_filtering_relations_and_max_evidence_cap(prompt_limits) -> None:
    rows = [
        _ev("T1", log_potency=4.0, assay_description="weak duplicate assay"),
        _ev(
            "T1",
            measurement="IC50",
            relation=">",
            value=14.0,
            log_potency=7.85,
            assay_description="strong winning assay",
        ),
        _ev("T2", measurement="Ki", relation="<", value=5.0, log_potency=None),
        _ev(
            "T3",
            measurement="Activity",
            relation=None,
            value=None,
            units=None,
            log_potency=None,
            outcome="inactive",
        ),
        _ev("T4", value=None, log_potency=None, outcome=None),
        _ev("T5", value=None, log_potency=None, outcome=""),
    ]

    full = _build(evidence=[rows])
    assert "strong winning assay" in full and "weak duplicate assay" not in full
    assert "IC50 > 14 nM" in full and "Ki < 5 nM" in full
    assert "T3" in full and "inactive" in full
    assert "T4" not in full and "T5" not in full

    prompt_limits(max_evidence=2)
    capped = _build(evidence=[rows])
    assert "T2" in capped and "T1" in capped and "T3" not in capped


def test_single_abstract_inclusion_and_truncation(prompt_limits) -> None:
    prompt_limits(abstract_max_chars=10)
    rows = [
        _ev(
            "T1", log_potency=9.0, document_title="No Abstract", document_abstract=None
        ),
        _ev(
            "T2",
            log_potency=8.0,
            document_title="First Paper",
            document_abstract="ABCDEFGHIJ_OVERFLOW",
        ),
        _ev(
            "T3",
            log_potency=7.0,
            document_title="Second Paper",
            document_abstract="SECOND_OMITTED",
        ),
    ]

    prompt = _build(evidence=[rows])

    assert "First Paper" in prompt and "ABCDEFGHIJ" in prompt
    assert "_OVERFLOW" not in prompt and "SECOND_OMITTED" not in prompt


def test_unreachable_sources_vs_no_measured_data() -> None:
    unreachable = _build(
        neighbours=[make_neighbour("CCCCN", score=0.5)],
        evidence=[[], []],
        unavailable={"PubChem", "ChEMBL"},
    )
    assert "ChEMBL" in unreachable and "PubChem" in unreachable
    assert (
        "could not be reached" in unreachable
        and "no measured data found" not in unreachable
    )

    healthy_empty = _build(evidence=[[]], unavailable=set())
    assert (
        "no measured data found" in healthy_empty
        and "could not be reached" not in healthy_empty
    )


def test_multi_neighbour_numbering_alignment_and_zero_neighbours() -> None:
    neighbours = [
        make_neighbour("CCCCN", score=0.85, mol_id="n1"),
        make_neighbour("CCCCCN", score=0.62, mol_id="n2"),
    ]
    prompt = _build(
        neighbours=neighbours,
        evidence=[[_ev("Q_TGT")], [_ev("NB1_TGT")], [_ev("NB2_TGT")]],
    )
    assert (
        prompt.index("Q_TGT")
        < prompt.index("[1] CCCCN")
        < prompt.index("NB1_TGT")
        < prompt.index("[2] CCCCCN")
        < prompt.index("NB2_TGT")
    )
    assert "[1]" not in _build(neighbours=[], evidence=[[]])


def test_positive_task_attributions_index_raw_smiles_and_cap_top_atoms(
    prompt_limits,
) -> None:
    prompt_limits(top_attributed_atoms=2)
    # In raw non-canonical "NCCC", nitrogen is atom 0; canonical "CCCN" puts nitrogen at atom 3.
    gnn_data = {
        "predictions": {"hERG_Karim": True, "CYP3A4": False, "AMES": True},
        "probabilities": {"hERG_Karim": 0.91, "CYP3A4": 0.12, "AMES": 0.65},
        "atomscores": {
            "hERG_Karim": {"0": 0.85, "1": 0.40, "2": 0.05, "3": 0.01, "4": None},
            "CYP3A4": {"0": 0.99},
            "AMES": {"0": None, "1": None},
        },
    }

    prompt = _build(smiles="NCCC", gnn_data=gnn_data)
    lines = {
        t: next(line for line in prompt.splitlines() if f"- {t}:" in line)
        for t in ("hERG_Karim", "CYP3A4", "AMES")
    }

    herg = lines["hERG_Karim"]
    assert "positive" in herg and herg.index("N0") < herg.index("C1")
    assert "N3" not in herg and "C2" not in herg and "C3" not in herg
    assert "negative" in lines["CYP3A4"] and "contributing atoms" not in lines["CYP3A4"]
    assert "positive" in lines["AMES"]
