import torch
from captum.attr import IntegratedGradients
from ..exceptions import ModelInferenceError


class IntegratedGradientsExplainer:
    """Computes atom-level Integrated Gradients attributions over Chemprop message-passing embeddings."""

    def __init__(
        self,
        predictor,
        steps: int = 100
    ):
        """Store the Chemprop predictor wrapper and number of Riemann interpolation steps."""
        self.predictor = predictor
        self.steps = steps

    def explain(
        self,
        atom_embeddings,
        mol_graph_index,
        num_targets: int
    ) -> list[list[float]]:
        """
        Calculate atom-level Integrated Gradients for all model targets.

        Input:
            atom_embeddings:
                [number_of_atoms, hidden_dim]
            mol_graph_index:
                [number_of_atoms]
            num_targets:
                Number of target endpoints to explain.

        Output:
            raw_scores:
                List of attribution score lists indexed by target: [target_index][atom_index]
        """
        # Shape validations on input
        if atom_embeddings is None or not isinstance(atom_embeddings, torch.Tensor):
            raise ModelInferenceError("atom_embeddings must be a valid torch.Tensor.")
        if atom_embeddings.ndim != 2:
            raise ModelInferenceError(f"atom_embeddings must be 2D [N, d], got shape {list(atom_embeddings.shape)}")

        num_atoms = atom_embeddings.shape[0]
        if num_atoms == 0:
            raise ModelInferenceError("Cannot explain molecule with 0 atom embeddings.")

        if mol_graph_index is None or mol_graph_index.shape[0] != num_atoms:
            raise ModelInferenceError(
                f"mol_graph_index shape mismatch: expected {num_atoms}, got {None if mol_graph_index is None else mol_graph_index.shape[0]}"
            )

        if num_targets <= 0:
            raise ModelInferenceError(f"num_targets must be positive, got {num_targets}")

        # IG expects the first dimension to be the number of examples.
        inputs = atom_embeddings.unsqueeze(0) 

        # IG baseline is a zero embedding matrix of the same size as the molecule
        baselines = torch.zeros_like(inputs)

        raw_scores: list[list[float]] = []

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

        # We attribute to atom embeddings not to input
        ig = IntegratedGradients(forward_func)

        # Ensure gradients are enabled for IG method
        with torch.enable_grad():

            for target_index in range(num_targets):

                attributions = ig.attribute(
                    inputs=inputs,
                    baselines=baselines,
                    target=target_index,
                    n_steps=self.steps  
                )

                # Validate attribution shape
                if attributions.shape != inputs.shape:
                    raise ModelInferenceError(
                        f"Captum attribution shape {list(attributions.shape)} != inputs shape {list(inputs.shape)}"
                    )

                # Sum attribution per atom over the embedding dimensions. [1, number_of_atoms, hidden_dim] -> [1, number_of_atoms]
                attributions = attributions.squeeze(0)
                scores_per_atom = attributions.sum(dim=-1)

                if scores_per_atom.shape[0] != num_atoms:
                    raise ModelInferenceError(
                        f"Expected {num_atoms} attribution scores for target {target_index}, got {scores_per_atom.shape[0]}"
                    )
                if torch.isnan(scores_per_atom).any() or torch.isinf(scores_per_atom).any():
                    raise ModelInferenceError(f"NaN or Inf detected in attribution scores for target {target_index}.")

                raw_scores.append(scores_per_atom.detach().cpu().tolist())
                    
        return raw_scores

