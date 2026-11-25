import asyncio

import pytest

from src.rag.augmentation.evidence_collection import gather_evidence
from tests.unit.conftest import FakeEvidenceSource, make_evidence, make_neighbour


def _pairs(rows) -> set[tuple[str, str]]:
    return {(e.provider, e.molecule_identifier) for e in rows}


def test_gather_evidence_routes_flattens_and_skips_unmatched_providers() -> None:
    chembl = FakeEvidenceSource(
        "ChEMBL",
        evidence_by_query={
            "CCO": [make_evidence(provider="ChEMBL", molecule_identifier="CCO")],
            "CHEMBL9": [
                make_evidence(molecule_identifier="CHEMBL9", target_identifier="T1"),
                make_evidence(molecule_identifier="CHEMBL9", target_identifier="T2"),
            ],
            "CHEMBL77": [make_evidence(molecule_identifier="CHEMBL77")],
        },
    )
    pubchem = FakeEvidenceSource("PubChem")
    neighbours = [
        make_neighbour("CCN", [("ChEMBL", "CHEMBL9")]),
        make_neighbour("CCC", [("PubChem", "CID10")]),
        make_neighbour("CCCC", [("ChEMBL", "CHEMBL77"), ("PubChem", "CID77")]),
        make_neighbour("CCCl", [("DrugBank", "DB00001")]),
        make_neighbour("CCBr", []),
    ]

    evidence, unavailable = asyncio.run(
        gather_evidence([chembl, pubchem], "CCO", neighbours)
    )

    assert set(chembl.queries) == {"CCO", "CHEMBL9", "CHEMBL77"}
    assert set(pubchem.queries) == {"CCO", "CID10", "CID77"}
    assert unavailable == set()
    assert len(evidence) == 6
    assert _pairs(evidence[0]) == {("ChEMBL", "CCO"), ("PubChem", "CCO")}
    assert [e.target_identifier for e in evidence[1]] == ["T1", "T2"]
    assert _pairs(evidence[2]) == {("PubChem", "CID10")}
    assert _pairs(evidence[3]) == {("ChEMBL", "CHEMBL77"), ("PubChem", "CID77")}
    assert evidence[4] == [] and evidence[5] == []


def test_empty_strategies_and_empty_neighbours_boundaries() -> None:
    nb = [make_neighbour("CCN", [("ChEMBL", "CHEMBL9")])]
    assert asyncio.run(gather_evidence([], "CCO", nb)) == ([[], []], set())

    evidence, unavailable = asyncio.run(
        gather_evidence([FakeEvidenceSource("ChEMBL")], "CCO", [])
    )
    assert len(evidence) == 1 and evidence[0][0].molecule_identifier == "CCO"
    assert unavailable == set()


def test_provider_outages_are_isolated_and_marked_unavailable() -> None:
    neighbours = [make_neighbour("CCN", [("ChEMBL", "CHEMBL9"), ("PubChem", "CID9")])]
    evidence, unavailable = asyncio.run(
        gather_evidence(
            [
                FakeEvidenceSource("ChEMBL", fail_on={"CCO"}),
                FakeEvidenceSource("PubChem"),
            ],
            "CCO",
            neighbours,
        )
    )
    assert _pairs(evidence[0]) == {("PubChem", "CCO")}
    assert _pairs(evidence[1]) == {("ChEMBL", "CHEMBL9"), ("PubChem", "CID9")}
    assert unavailable == {"ChEMBL"}

    down = [
        FakeEvidenceSource("ChEMBL", error=ConnectionError("down")),
        FakeEvidenceSource("PubChem", error=ConnectionError("down")),
    ]
    assert asyncio.run(gather_evidence(down, "CCO", neighbours)) == (
        [[], []],
        {"ChEMBL", "PubChem"},
    )


def test_non_connection_error_from_strategy_propagates() -> None:
    broken = FakeEvidenceSource("ChEMBL", error=ValueError("parse bug"))
    with pytest.raises(ValueError, match="parse bug"):
        asyncio.run(gather_evidence([broken], "CCO", []))
