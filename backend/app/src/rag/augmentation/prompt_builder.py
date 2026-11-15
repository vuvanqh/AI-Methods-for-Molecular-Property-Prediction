import math

from rdkit import Chem

from ..config import promptBuilderSettings
from ..result_types.molecular_evidence import MolecularEvidence
from ..result_types.retrieved_molecules import RetrievedMolecule

INSTRUCTIONS = """Write a scientific interpretation of the GNN predictions above for a chemist, in at most 250 words of plain text (no markdown, no lists).
- Explain the positive predictions and which substructures drive them, using the most contributing atoms.
- Use the measured data to support or question each prediction. You decide which measured targets relate to which property (for example KCNH2 is the hERG channel); ignore unrelated measurements.
- Data of the queried molecule itself is direct evidence. Data of similar molecules is indirect: weigh it by score (higher means a closer match) and cite the molecule as [n].
- Say clearly when the evidence contradicts the model or when there is no relevant evidence. Never invent measurements or references.
- A source shown as unknown or unreachable means the data was not retrieved, not that it does not exist. Never conclude a molecule has no measured data in that case; say the check could not be completed."""


def _rank(e: MolecularEvidence) -> float:
    """
    The client returns thousands of measurements, but the prompt budget is limited.
    We want the most potent ones first, hence we assing a rank to each measurement.
    The rank is the -log10 of the molar concentration, so a lower IC50 is better.

    Potency on one scale, so screening rows that carry a concentration but no
    -log10 value compete with the rest instead of always sorting last.
    """
    if e.log_potency is not None:
        return e.log_potency
    if e.units == "nM" and e.value:
        return 9 - math.log10(e.value)
    return 0


def _select(evidence: list[MolecularEvidence]) -> list[MolecularEvidence]:
    """
    We have a limited prompt budget, so we select the most potent measurement per target.

    The most potent measurement per target, so the budget covers many targets
    instead of repeating one.
    """
    measured = [e for e in evidence if e.value is not None or e.outcome]
    measured.sort(key=_rank, reverse=True)
    selected, seen_targets = [], set()
    for e in measured:
        if e.target_identifier not in seen_targets:
            seen_targets.add(e.target_identifier)
            selected.append(e)
    return selected[: promptBuilderSettings.max_evidence_per_molecule]


def _format_measurement(e: MolecularEvidence) -> str:
    value = None if e.value is None else f"{e.value:g}"
    text = " ".join(p for p in (e.measurement, e.relation, value, e.units) if p)
    if e.log_potency is not None:
        text += f" (-log10[M] {e.log_potency:g})"
    if e.outcome:
        text += f" [{e.outcome}]"
    reference = f"; {e.document_title} ({e.document_year})" if e.document_title else ""
    return f"- {e.target_name} ({e.provider}): {text}; assay: {e.assay_description}{reference}"


def _format_evidence(
    evidence: list[MolecularEvidence], unavailable: set[str]
) -> list[str]:
    selected = _select(evidence)
    if not selected:
        if unavailable:
            return [f"- unknown: {', '.join(sorted(unavailable))} could not be reached"]
        return ["- no measured data found"]
    lines = [_format_measurement(e) for e in selected]
    for e in selected:
        if e.document_abstract:
            abstract = e.document_abstract[: promptBuilderSettings.abstract_max_chars]
            lines.append(f'  Abstract of "{e.document_title}": {abstract}')
            break
    return lines


def _atom_environment(mol: Chem.Mol, index: int) -> str:
    atom = mol.GetAtomWithIdx(index)
    neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
    return Chem.MolFragmentToSmiles(mol, atomsToUse=[index, *neighbours])


def _top_atoms(mol: Chem.Mol, scores: dict) -> str:
    # keys arrive as strings after JSON; hydrogens carry no score
    ranked = sorted(
        ((score, int(index)) for index, score in scores.items() if score is not None),
        reverse=True,
    )[: promptBuilderSettings.top_attributed_atoms]
    return ", ".join(
        f"{mol.GetAtomWithIdx(i).GetSymbol()}{i} in {_atom_environment(mol, i)} ({score:.3g})"
        for score, i in ranked
    )


def build_prompt(
    smiles: str,
    molecule: dict,
    gnn_data: dict,
    neighbours: list[RetrievedMolecule],
    evidence: list[list[MolecularEvidence]],
    unavailable: set[str],
) -> str:
    """
    Combine the GNN predictions, their atom attributions and the retrieved evidence
    into one prompt. evidence[0] belongs to the queried molecule, evidence[n] to
    neighbour n, matching the numbered sources the frontend shows.
    """
    # the GNN indexes atoms in the order of the SMILES it received, not the canonical one
    mol = Chem.MolFromSmiles(smiles)
    lines = [
        f"Queried molecule: {molecule['smiles']} ({molecule['formula']}, {molecule['molar_mass']:.1f} g/mol)",
        "",
        "GNN predictions (probability of the positive class):",
    ]
    for task, positive in gnn_data["predictions"].items():
        line = f"- {task}: {'positive' if positive else 'negative'} (p={gnn_data['probabilities'][task]:.2f})"
        if positive:
            line += f"; most contributing atoms: {_top_atoms(mol, gnn_data['atomscores'][task])}"
        lines.append(line)

    lines += [
        "",
        "Measured data for the queried molecule:",
        *_format_evidence(evidence[0], unavailable),
    ]
    lines += ["", "Similar molecules:"]
    for number, (neighbour, neighbour_evidence) in enumerate(
        zip(neighbours, evidence[1:]), start=1
    ):
        lines.append(
            f"[{number}] {neighbour.molecule.standardized_smiles} (score {neighbour.score:.2f})"
        )
        lines += _format_evidence(neighbour_evidence, unavailable)

    return "\n".join([*lines, "", INSTRUCTIONS])
