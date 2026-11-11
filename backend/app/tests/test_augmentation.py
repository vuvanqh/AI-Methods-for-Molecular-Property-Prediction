import asyncio
from dataclasses import replace
from types import SimpleNamespace

from src.rag.augmentation.client_retrieval_base import ClientStrategyBase
from src.rag.augmentation.evidence_collection import gather_evidence
from src.rag.augmentation.prompt_builder import _select, build_prompt
from src.rag.result_types.molecular_evidence import MolecularEvidence
from src.rag.result_types.retrieved_molecules import RetrievedMolecule

EVIDENCE = MolecularEvidence(
    provider="ChEMBL",
    molecule_identifier="CHEMBL1",
    target_identifier="CHEMBL240",
    target_name="KCNH2",
    measurement="IC50",
    relation="=",
    value=14.0,
    units="nM",
    log_potency=7.85,
    outcome=None,
    assay_description="hERG patch clamp",
    document_title="hERG paper",
    document_abstract=None,
    document_year=2001,
    document_url=None,
)


def neighbour(smiles: str, sources: list[tuple[str, str]] = ()) -> RetrievedMolecule:
    molecule = SimpleNamespace(
        standardized_smiles=smiles,
        molecule_sources=[
            SimpleNamespace(provider=p, source_identifier=i) for p, i in sources
        ],
    )
    return RetrievedMolecule(molecule=molecule, score=0.75)


class FakeSource(ClientStrategyBase):
    def __init__(self, provider: str = "ChEMBL", fail_on: str | None = None):
        self.provider = provider
        self.queries = []
        self.fail_on = fail_on

    async def retrieve_molecular_evidence(self, query):
        self.queries.append(query)
        if query == self.fail_on:
            raise ConnectionError("down")
        return [replace(EVIDENCE, molecule_identifier=query)]


def registry(*sources: FakeSource) -> list[FakeSource]:
    return list(sources)


def test_queried_molecule_is_asked_by_structure_and_neighbours_by_stored_id():
    source = FakeSource()

    evidence, _ = asyncio.run(
        gather_evidence(
            registry(source),
            "CCO",
            [neighbour("CCN", [("ChEMBL", "CHEMBL9")])],
        )
    )

    assert source.queries == ["CCO", "CHEMBL9"]
    assert [e[0].molecule_identifier for e in evidence] == ["CCO", "CHEMBL9"]


def test_each_neighbour_goes_only_to_the_providers_it_was_ingested_from():
    chembl, pubchem = FakeSource(), FakeSource("PubChem")

    asyncio.run(
        gather_evidence(
            registry(chembl, pubchem),
            "CCO",
            [
                neighbour("CCN", [("ChEMBL", "CHEMBL9")]),
                neighbour("CCC", [("PubChem", "P1")]),
            ],
        )
    )

    assert chembl.queries == ["CCO", "CHEMBL9"]
    assert pubchem.queries == ["CCO", "P1"]


def test_neighbour_from_a_provider_without_a_strategy_is_skipped():
    source = FakeSource()

    evidence, _ = asyncio.run(
        gather_evidence(
            registry(source), "CCO", [neighbour("CCC", [("Drugbank", "DB1")])]
        )
    )

    assert source.queries == ["CCO"]
    assert evidence[1] == []


def test_a_failing_source_only_drops_its_own_evidence():
    evidence, unavailable = asyncio.run(
        gather_evidence(
            registry(FakeSource(fail_on="CCO")),
            "CCO",
            [neighbour("CCN", [("ChEMBL", "CHEMBL9")])],
        )
    )

    assert evidence[0] == []
    assert len(evidence[1]) == 1
    assert unavailable == {"ChEMBL"}


def test_selection_keeps_most_potent_row_per_target_and_drops_empty_rows():
    weaker = replace(EVIDENCE, log_potency=5.0)
    other_target = replace(
        EVIDENCE,
        target_identifier="CHEMBL340",
        value=None,
        log_potency=None,
        outcome="inactive",
    )
    empty = replace(EVIDENCE, target_identifier="CHEMBL1", value=None, log_potency=None)

    assert _select([weaker, other_target, EVIDENCE, empty]) == [EVIDENCE, other_target]


def test_screening_row_without_log_potency_is_ranked_by_its_concentration():
    # 50 uM screening hit: 9 - log10(50000 nM) = 4.3, so it outranks a weaker 1 mM row
    screening = replace(
        EVIDENCE,
        target_identifier="CHEMBL1871",
        log_potency=None,
        value=50000.0,
        outcome="active",
    )
    weak = replace(EVIDENCE, target_identifier="CHEMBL340", log_potency=3.0)

    assert _select([weak, screening]) == [screening, weak]


def test_prompt_contains_predictions_atoms_and_numbered_evidence():
    gnn_data = {
        "predictions": {"hERG_Karim": True, "CYP3A4": False},
        "probabilities": {"hERG_Karim": 0.91, "CYP3A4": 0.12},
        # keys are strings after JSON; index 3 is the nitrogen in CCCN
        "atomscores": {
            "hERG_Karim": {"0": 0.01, "1": 0.02, "2": 0.05, "3": 0.4, "4": None},
            "CYP3A4": {},
        },
    }
    molecule = {"smiles": "CCCN", "formula": "C3H9N", "molar_mass": 59.11}
    with_abstract = replace(EVIDENCE, document_abstract="Basic nitrogen binds Tyr652.")

    prompt = build_prompt(
        "CCCN", molecule, gnn_data, [neighbour("CCCCN")], [[], [with_abstract]], set()
    )

    assert (
        "- hERG_Karim: positive (p=0.91); most contributing atoms: N3 in CN (0.4)"
        in prompt
    )
    assert "- CYP3A4: negative (p=0.12)" in prompt
    assert "Measured data for the queried molecule:\n- no measured data found" in prompt
    assert "[1] CCCCN (similarity 0.75)" in prompt
    assert (
        "- KCNH2 (ChEMBL): IC50 = 14 nM (-log10[M] 7.85); assay: hERG patch clamp; hERG paper (2001)"
        in prompt
    )
    assert 'Abstract of "hERG paper": Basic nitrogen binds Tyr652.' in prompt


def test_an_unreachable_source_is_not_reported_as_absent():
    gnn_data = {"predictions": {}, "probabilities": {}, "atomscores": {}}
    molecule = {"smiles": "CCCN", "formula": "C3H9N", "molar_mass": 59.11}

    prompt = build_prompt(
        "CCCN", molecule, gnn_data, [neighbour("CCCCN")], [[], []], {"ChEMBL"}
    )

    assert "- unknown: ChEMBL could not be reached" in prompt
    assert "no measured data found" not in prompt
