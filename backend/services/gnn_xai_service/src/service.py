import math

from rdkit import Chem

from .config import BINARY_PREDICTION_THRESHOLD, DEVICE, IG_STEPS, TASKS, TASKS_NUMBER
from .exceptions import ModelInferenceError, MoleculeValidationError
from .gnn_models.chemprop import Chemprop
from .mol_block_generation import mol_block_generation
from .xai.integrated_gradients import IntegratedGradientsExplainer


class MolecularInferenceService:
    """Orchestrates SMILES validation, Chemprop GNN prediction, Integrated Gradients XAI, and 3D MolBlock generation."""

    def __init__(self):
        """Initialize the Chemprop predictor and Integrated Gradients explainer once at startup."""
        self.predictor = Chemprop(TASKS_NUMBER, DEVICE, BINARY_PREDICTION_THRESHOLD)

        self.explainer = IntegratedGradientsExplainer(
            predictor=self.predictor,
            steps=IG_STEPS,
        )

    def predict_and_explain(self, smiles: str):
        """
        Validate a SMILES string, run multi-task GNN prediction and atom-level IG attribution,
        and generate an index-aligned MolBlock for 3Dmol.js visualization.
        """
        # 1. Validate SMILES input
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            raise MoleculeValidationError("SMILES must not be empty.")

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise MoleculeValidationError("RDKit could not parse the provided SMILES.")

        num_atoms_without_h = mol.GetNumAtoms()
        if num_atoms_without_h == 0:
            raise MoleculeValidationError(
                "The molecule must contain at least one atom."
            )

        # Normalize to canonical SMILES so chiral tags (@/@@) and atom indices are traversal-independent
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise MoleculeValidationError(
                "RDKit could not re-parse the canonicalized SMILES."
            )
        num_atoms_without_h = mol.GetNumAtoms()

        # 2. Run GNN model prediction
        result = self.predictor.predict(mol)

        # Extract and validate probabilities
        probabilities_tensor = result.get("probabilities")
        if probabilities_tensor is None:
            raise ModelInferenceError("Missing 'probabilities' in predictor output.")

        probabilities_list = probabilities_tensor.detach().cpu().tolist()
        if len(probabilities_list) != len(TASKS):
            raise ModelInferenceError(
                f"Probabilities length mismatch: expected {len(TASKS)}, got {len(probabilities_list)}"
            )

        probabilities = {}
        for task_name, prob in zip(TASKS, probabilities_list):
            prob_float = float(prob)
            if math.isnan(prob_float) or prob_float < 0.0 or prob_float > 1.0:
                raise ModelInferenceError(
                    f"Invalid probability value {prob_float} for task {task_name}."
                )
            probabilities[task_name] = prob_float

        # Extract and validate predictions
        predictions_tensor = result.get("predictions")
        if predictions_tensor is None:
            raise ModelInferenceError("Missing 'predictions' in predictor output.")

        predictions_list = predictions_tensor.detach().cpu().tolist()
        if len(predictions_list) != len(TASKS):
            raise ModelInferenceError(
                f"Predictions length mismatch: expected {len(TASKS)}, got {len(predictions_list)}"
            )

        predictions = {
            task_name: bool(pred) for task_name, pred in zip(TASKS, predictions_list)
        }

        # 3. Calculate atom-level XAI attributions (by target indices)
        raw_attribution_scores = self.explainer.explain(
            atom_embeddings=result["atom_embeddings"],
            mol_graph_index=result["mol_graph_index"],
            num_targets=len(TASKS),
        )

        if len(raw_attribution_scores) != len(TASKS):
            raise ModelInferenceError(
                f"Explainer returned {len(raw_attribution_scores)} target attributions, expected {len(TASKS)}"
            )

        # 4. Generate 3D MolBlock representation and get mol_with_h
        molblock, mol_with_h = mol_block_generation(mol)
        num_atoms_with_h = mol_with_h.GetNumAtoms()

        if num_atoms_with_h < num_atoms_without_h:
            raise ModelInferenceError(
                f"Atom count mismatch: mol_with_h has {num_atoms_with_h} atoms, which is less than heavy atoms ({num_atoms_without_h})"
            )

        # to make sure there is no mismatch in atoms ordering between prediction and visualization:
        self.validate_mol_atomic_indices(
            mol, num_atoms_without_h, mol_with_h, num_atoms_with_h
        )

        # 5. Map target index to task names and build atom scores dictionary using real Atom.GetIdx()
        atom_scores = {}

        for task_index, task_name in enumerate(TASKS):
            scores = raw_attribution_scores[task_index]
            if len(scores) != num_atoms_without_h:
                raise ModelInferenceError(
                    f"Task {task_name} attribution score count ({len(scores)}) != num_atoms_without_h ({num_atoms_without_h})."
                )

            atom_scores[task_name] = {
                atom.GetIdx(): (
                    float(scores[atom.GetIdx()])
                    if atom.GetIdx() < num_atoms_without_h
                    else None
                )
                for atom in mol_with_h.GetAtoms()
            }

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "atomscores": atom_scores,
            "molblock": molblock,
        }

    def validate_mol_atomic_indices(
        self, mol, num_atoms_without_h, mol_with_h, num_atoms_with_h
    ):
        """Verify that heavy atoms remain at indices 0 .. K-1 and appended atoms at K .. N-1 are all hydrogens."""
        # Verify heavy-atom indices and atomic numbers match between `mol` and `mol_with_h`
        for i in range(num_atoms_without_h):
            heavy_atom = mol.GetAtomWithIdx(i)
            atom_in_h_mol = mol_with_h.GetAtomWithIdx(i)
            if heavy_atom.GetIdx() != atom_in_h_mol.GetIdx():
                raise ModelInferenceError(
                    f"Atom index mismatch at {i}: original {heavy_atom.GetIdx()} != with_h {atom_in_h_mol.GetIdx()}"
                )
            if heavy_atom.GetAtomicNum() != atom_in_h_mol.GetAtomicNum():
                raise ModelInferenceError(
                    f"Atom type mismatch at index {i}: original Z={heavy_atom.GetAtomicNum()} != with_h Z={atom_in_h_mol.GetAtomicNum()}"
                )

        # Check invariant: remaining atoms in mol_with_h must all be Hydrogen (atomic number 1)
        for i in range(num_atoms_without_h, num_atoms_with_h):
            h_atom = mol_with_h.GetAtomWithIdx(i)
            if h_atom.GetAtomicNum() != 1:
                raise ModelInferenceError(
                    f"Expected Hydrogen at index {i}, but got atomic number Z={h_atom.GetAtomicNum()}"
                )
