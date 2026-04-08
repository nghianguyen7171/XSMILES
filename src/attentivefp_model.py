"""
AttentiveFP model for molecular toxicity prediction.

AttentiveFP (Xiong et al., 2020) uses gated attention on both atom and
molecule levels. The intrinsic attention weights provide faithful atom
importance without post-hoc attribution, resolving the compensation and
positional-encoding shortcut issues seen in SMILESGNN.

Reference:
    Xiong et al., "Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism", JCIM 2020.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Data


class AttentiveFPPredictor(nn.Module):
    """
    AttentiveFP backbone with a linear prediction head.

    Wraps torch_geometric's AttentiveFP so that the backbone returns an
    embedding vector (shape: B × hidden_channels) which the head maps to
    task logits.  Supports both single-task (squeeze to B) and multi-task
    (B × num_tasks) output.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.2,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.backbone = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,   # returns embedding, not logits
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_channels, num_tasks)
        self.num_tasks = num_tasks

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.backbone(x, edge_index, edge_attr, batch)  # (B, hidden)
        out = self.head(h)                                   # (B, T)
        if out.shape[-1] == 1:
            return out.squeeze(-1)                           # (B,) single-task
        return out                                           # (B, T) multi-task

    def get_atom_importance(
        self,
        data: Data,
        task_idx: int = 0,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Compute GradCAM atom importance scores for a single molecule.

        Hooks are attached to the last GRUCell in the backbone's atom_grus
        list to capture activations and gradients simultaneously.  The
        importance per atom is ReLU(sum(grad * activation, dim=-1)).

        Args:
            data:     PyG Data object for a single molecule (batch=0 for all).
            task_idx: Index of the task to explain (0 for single-task).
            device:   "cpu" or "cuda".

        Returns:
            Numpy array of shape (n_atoms,) with non-negative importance scores.
        """
        self.eval()
        data = data.to(device)

        # Ensure batch attribute is present
        if data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        activations: dict = {}
        gradients: dict = {}

        def fwd_hook(module, inp, out):
            activations["h"] = out          # GRUCell returns (n_atoms, hidden)

        def bwd_hook(module, grad_in, grad_out):
            gradients["g"] = grad_out[0]    # gradient w.r.t. GRUCell output

        target_gru = self.backbone.atom_grus[-1]
        fh = target_gru.register_forward_hook(fwd_hook)
        bh = target_gru.register_full_backward_hook(bwd_hook)

        self.zero_grad()
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)

        # Select the score to differentiate
        if out.dim() == 2:
            score = out[0, task_idx]
        else:
            score = out[0]

        score.backward()

        fh.remove()
        bh.remove()

        h = activations["h"]    # (n_atoms, hidden)
        g = gradients["g"]      # (n_atoms, hidden)
        importance = torch.relu((g * h).sum(dim=-1))
        return importance.detach().cpu().numpy()


def create_attentivefp_model(
    node_feat_dim: int,
    edge_feat_dim: int,
    hidden_channels: int = 200,
    num_layers: int = 2,
    num_timesteps: int = 2,
    dropout: float = 0.2,
    num_tasks: int = 1,
) -> AttentiveFPPredictor:
    """
    Factory function for AttentiveFPPredictor.

    Args:
        node_feat_dim:   Dimension of atom (node) features.
        edge_feat_dim:   Dimension of bond (edge) features.
        hidden_channels: Width of all internal representations.
        num_layers:      Number of graph attention layers.
        num_timesteps:   Number of molecule-level GRU timesteps.
        dropout:         Dropout probability.
        num_tasks:       Number of output tasks (1 = single-task).

    Returns:
        AttentiveFPPredictor instance (not yet moved to device).
    """
    return AttentiveFPPredictor(
        in_channels=node_feat_dim,
        hidden_channels=hidden_channels,
        edge_dim=edge_feat_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        num_tasks=num_tasks,
    )
