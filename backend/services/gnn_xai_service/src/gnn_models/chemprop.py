import torch
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BinaryClassificationFFN, BondMessagePassing, NormAggregation

from ..exceptions import ModelInferenceError
from .gnn_interface import GNNPredictor


class Chemprop(GNNPredictor):
    """Wrapper around a Chemprop v2 MPNN exposing intermediate atom embeddings for Integrated Gradients."""

    def __init__(self, n_tasks, device, thresholds):
        """Initialize the featurizer, BondMessagePassing, NormAggregation, and BinaryClassificationFFN."""
        self.device = device
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
        self.n_tasks = n_tasks

        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

        mp = BondMessagePassing() 
        agg = NormAggregation() 
        ffn = BinaryClassificationFFN(n_tasks) 
        self.model = MPNN(mp, agg, ffn)

        self.model.eval()
        self.model = self.model.to(self.device)


    def predict(self, mol):
        """Predict molecular properties for one RDKit molecule.
        https://chemprop.readthedocs.io/en/latest/_modules/chemprop/models/model.html
        """
        if mol is None:
            raise ModelInferenceError("Cannot run Chemprop prediction on None molecule.")

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            raise ModelInferenceError("Molecule contains 0 atoms.")

        # batch of size 1 in our case
        batch_graph = self.build_batch_mol_graph(mol) 

        atom_embeddings = self.get_embeddings(batch_graph)

        # Invariant checks on embeddings
        if atom_embeddings.ndim != 2:
            raise ModelInferenceError(
                f"Expected 2D atom embeddings [num_atoms, hidden_dim], got shape {list(atom_embeddings.shape)}"
            )
        if atom_embeddings.shape[0] != num_atoms:
            raise ModelInferenceError(
                f"Atom count mismatch in embeddings: mol has {num_atoms} atoms, but embeddings has {atom_embeddings.shape[0]}"
            )
        if torch.isnan(atom_embeddings).any() or torch.isinf(atom_embeddings).any():
            raise ModelInferenceError("NaN or Inf values detected in atom embeddings.")

        logits = self.forward_from_embeddings(atom_embeddings)

        # Invariant checks on logits
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != self.n_tasks:
            raise ModelInferenceError(
                f"Expected logits of shape [1, {self.n_tasks}], got {list(logits.shape)}"
            )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ModelInferenceError("NaN or Inf detected in model output logits.")

        probabilities = logits.sigmoid().squeeze(0)
        predictions = (probabilities > self.thresholds).int()

        if probabilities.shape[0] != self.n_tasks:
            raise ModelInferenceError(
                f"Expected {self.n_tasks} probabilities, got {probabilities.shape[0]}"
            )

        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "atom_embeddings": atom_embeddings,
        }

    def build_batch_mol_graph(self, mol):
        """
        Convert one RDKit molecule into a Chemprop BatchMolGraph.
        https://chemprop.readthedocs.io/en/latest/tutorial/python/featurizers/molgraph_molecule_featurizer.html
        """
        mol_graph = self.featurizer(mol)
        if mol_graph is None:
            raise ModelInferenceError("Failed to featurize molecule into MolGraph.")
        mol_graphs = [mol_graph]
        batch_graph = BatchMolGraph(mol_graphs)
        batch_graph.to(self.device)

        return batch_graph

    def get_embeddings(self, batch_graph):
        """
        Generate atom embeddings with Chemprop message passing.
        https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/nn/message_passing/index.html
        """
        with torch.no_grad():
            atom_embeddings = self.model.message_passing(batch_graph)
        return atom_embeddings

    def forward_from_embeddings(self, atom_embeddings):
        """Chemprop forward pass from atom embeddings to prediction logits (used in integrated gradients)"""
        mol_graph_index = torch.zeros(
            atom_embeddings.shape[0],
            dtype=torch.long,
            device=atom_embeddings.device,
        )

        # aggregate from atom embeddings to one molecule embedding
        molecule_embeddings = self.model.agg(
            atom_embeddings,
            mol_graph_index, 
        )

        # batchnorm
        molecule_embeddings = self.model.bn(molecule_embeddings)

        # logits
        logits = self.model.predictor.ffn(molecule_embeddings)

        return logits
