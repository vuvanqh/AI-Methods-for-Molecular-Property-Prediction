from dataclasses import dataclass


@dataclass(frozen=True)
class MolecularEvidence:
    """
    One experimental measurement of a molecule, as any evidence provider reports it.
    """

    provider: str
    molecule_identifier: str
    target_identifier: str | None
    target_name: str | None
    measurement: str | None
    relation: str | None
    value: float | None
    units: str | None
    # -log10 of the molar value, so 6 means 1 uM; ChEMBL reports it as pchembl_value
    log_potency: float | None
    # a verdict such as "active" when the provider reports no number
    outcome: str | None
    assay_description: str | None
    document_title: str | None
    document_abstract: str | None
    document_year: int | None
    document_url: str | None
