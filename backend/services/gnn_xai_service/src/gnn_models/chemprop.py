import torch

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, NormAggregation, BinaryClassificationFFN

class Chemprop:
    '''https://chemprop.readthedocs.io/en/latest/tutorial/python/models/basic_mpnn_model.html'''
    def __init__(self, n_tasks, device, threshold):
        self.device = device
        self.threshold = threshold 

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
            # batch of size 1 in our case
            batch_graph = self.build_batch_mol_graph(mol) 
    
            atom_embeddings = self.get_embeddings(batch_graph)

            mol_graph_index = batch_graph.batch
    
            logits = self.forward_from_embeddings(
                atom_embeddings,
                mol_graph_index, 
            )
    
            probabilities = logits.sigmoid()

            predictions = (probabilities > self.threshold).int() 
    
            return {"probabilities": probabilities.squeeze(0),
                    "predictions": predictions.squeeze(0),
                    "atom_embeddings": atom_embeddings,
                    "mol_graph_index": mol_graph_index
                    }
        

    def build_batch_mol_graph(self, mol):
        """
        Convert one RDKit molecule into a Chemprop BatchMolGraph.
        https://chemprop.readthedocs.io/en/latest/tutorial/python/featurizers/molgraph_molecule_featurizer.html
        """
        mol_graphs = [self.featurizer(mol)]
        batch_graph = BatchMolGraph(mol_graphs)

        return batch_graph.to(self.device)
    

    def get_embeddings(self, batch_graph):
        """
        Generate atom embeddings with Chemprop message passing.
        https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/nn/message_passing/index.html
        """
        with torch.no_grad():
            atom_embeddings = self.model.message_passing(batch_graph)
        return atom_embeddings


    def forward_from_embeddings(
        self,
        atom_embeddings,
        mol_graph_index,
    ):
        """Chemprop forward pass from atom embeddings to prediction logits (used in integrated gradients)"""
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
    

   
    

    