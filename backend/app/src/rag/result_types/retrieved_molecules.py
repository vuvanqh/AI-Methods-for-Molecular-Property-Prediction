from dataclasses import dataclass

from ...db.models import Molecule


@dataclass(frozen=True)
class RetrievedMolecule:
    """
    Represents a molecule retrieved from the database along with its similarity score.
    """

    molecule: Molecule
    score: float
