from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from .config import TASKS_NUMBER, DEVICE, BINARY_PREDICTION_THRESHOLD, TASKS, IG_STEPS
from .gnn_models.chemprop import Chemprop
from .xai.integrated_gradients import IntegratedGradientsExplainer
from .mol_block_generation import mol_block_generation
import threading

class MolecularInferenceService:

    def __init__(self):

        # Initialize the GNN and XAI once.
        self.predictor = Chemprop(TASKS_NUMBER, DEVICE, BINARY_PREDICTION_THRESHOLD)

        self.explainer = IntegratedGradientsExplainer(
            predictor=self.predictor,
            task_names=TASKS,
            steps=IG_STEPS,
        )
        self.inference_lock = threading.Lock()

    def predict_and_explain(
        self,
        smiles
    ):
        with self.inference_lock:
            #validate smiles

            if not smiles or not smiles.strip():
                raise ValueError(
                    "SMILES must not be empty."
                )
            
            mol = Chem.MolFromSmiles(smiles) 

            if mol is None:
                raise ValueError(
                    "RDKit could not parse the provided SMILES."
                )

            if mol.GetNumAtoms() == 0:
                raise ValueError(
                    "The molecule must contain at least one atom."
                )

            result = self.predictor.predict(mol)

            # get probabilities
            probabilities_list = result["probabilities"].detach().cpu().tolist()

            probabilities = {
                task_name: float(probability)
                for task_name, probability in zip(
                    TASKS,
                    probabilities_list
                )
            }

            # get predictions
            predictions_list = result["predictions"].detach().cpu().tolist()
            
            predictions = {
                task_name: bool(prediction)
                for task_name, prediction in zip(
                    TASKS,
                    predictions_list
                )
            }

            # get atom scores for each task
            atom_scores_lists = self.explainer.explain(
                atom_embeddings=result["atom_embeddings"],
                mol_graph_index=result["mol_graph_index"]
            )

            molblock = mol_block_generation(mol)


            molblock_mol = Chem.MolFromMolBlock(molblock, removeHs=False)

            atom_scores = {}

            for task_name in TASKS:
                scores = atom_scores_lists[task_name]

                atom_scores[task_name] = {
                    atom.GetIdx(): 
                    (
                        # none for hydrogen atoms
                        float(scores[atom.GetIdx()]) if atom.GetIdx() < mol.GetNumAtoms() else None 
                    )
                    for atom in molblock_mol.GetAtoms()
                }

            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "atomscores": atom_scores,
                "molblock": molblock
            }