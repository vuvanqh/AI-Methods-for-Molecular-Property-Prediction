import torch
from captum.attr import IntegratedGradients
from ..config import TASKS


class IntegratedGradientsExplainer:

    def __init__(
        self,
        predictor,
        task_names: list[str] = TASKS,
        steps: int = 100
    ):
        self.predictor = predictor
        self.task_names = task_names
        self.steps = steps


    def explain(
        self,
        atom_embeddings,
        mol_graph_index
    ):
        """
        Calculate atom-level Integrated Gradients for all endpoints.

        Input:
            atom_embeddings:
                [number_of_atoms, hidden_dim]

            atom_to_molecule:
                [number_of_atoms]

        Output:
            atom_scores:
                Attribution scores per atom per endpoint.

        sources: 
        https://chemprop.readthedocs.io/en/latest/tutorial/python/models/message_passing.html
        https://captum.ai/api/integrated_gradients.html

        """

        # IG expects the first dimension to be the number of exaples.
        inputs = atom_embeddings.unsqueeze(0) 

        # IG baseline is a zero embedding matrix of the same size as the molecule (might be changed later)
        baselines = torch.zeros_like(inputs)

        atom_scores_lists = {}

        def forward_func(embeddings_batched):
            """
            Captum passes embeddings with shape:
            [batch_size, number_of_atoms, hidden_dim]
            """

            logits_list = []
            for embeddings in embeddings_batched:
                logits = self.predictor.forward_from_embeddings(
                    embeddings,
                    mol_graph_index,
                )
                logits_list.append(logits)

            return torch.cat(logits_list, dim=0) 

        # We attribute to atom embeddings not to input (might be changed later)
        ig = IntegratedGradients(forward_func)

        # Ensure gradients are enabled for IG method
        with torch.enable_grad():

            # for each task calculate different attribution
            for task_index, task_name in enumerate(self.task_names):

                attributions = ig.attribute(
                    inputs=inputs,
                    baselines=baselines,
                    target=task_index,
                    n_steps=self.steps  
                )

                # Sum attribution per atom over the embedding dimensions. [1, number_of_atoms, hidden_dim] -> [1, number_of_atoms]
                attributions = attributions.squeeze(0)
                scores_per_atom = attributions.sum(dim=-1) 

                atom_scores_lists[task_name] = scores_per_atom.detach().cpu().tolist() 
                    
        return atom_scores_lists