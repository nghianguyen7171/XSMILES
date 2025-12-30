"""
Graph Isomorphism Network (GIN) model for molecular property prediction.

GIN uses MLP-based message passing and is provably as powerful as the Weisfeiler-Lehman
graph isomorphism test, making it theoretically stronger than many GNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import JumpingKnowledge, BatchNorm
from typing import Optional, Literal


class MLP(nn.Module):
    """Multi-layer perceptron for GIN layers."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GINMolecularPredictor(nn.Module):
    """
    GIN (Graph Isomorphism Network) based molecular property predictor.
    
    Architecture:
    - GIN layers with MLP-based message passing
    - Residual connections
    - Jumping Knowledge (optional)
    - Global pooling (mean/max/sum)
    - MLP predictor head
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_jk: bool = True,
        jk_mode: Literal["cat", "max", "mean", "lstm"] = "cat",
        pooling: Literal["mean", "max", "sum", "meanmax"] = "meanmax",
        output_dim: int = 1,
        train_eps: bool = True  # Learnable epsilon for GIN
    ):
        """
        Initialize GINMolecularPredictor.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features (not used in GIN, but kept for API consistency)
            hidden_dim: Hidden dimension size
            num_layers: Number of GIN layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_jk: Whether to use Jumping Knowledge connections
            jk_mode: JK aggregation mode ("cat", "max", "mean", "lstm")
            pooling: Pooling strategy ("mean", "max", "sum", "meanmax")
            output_dim: Output dimension (1 for binary classification)
            train_eps: Whether to use learnable epsilon parameter in GIN
        """
        super().__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_jk = use_jk
        self.pooling = pooling
        
        # Input projection for node features
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.eps = nn.ParameterList() if train_eps else None
        
        for i in range(num_layers):
            # Learnable epsilon parameter for GIN
            if train_eps:
                eps_param = nn.Parameter(torch.tensor([0.0]))
                self.eps.append(eps_param)
            else:
                eps_param = None
            
            # GIN layer with MLP
            mlp = MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                output_dim=hidden_dim,
                num_layers=2,
                dropout=dropout
            )
            
            conv = GINConv(nn=mlp, train_eps=train_eps)
            self.convs.append(conv)
            
            # Batch normalization
            self.norms.append(BatchNorm(hidden_dim))
        
        # Jumping Knowledge (aggregate features from all layers)
        if use_jk:
            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
            if jk_mode == "cat":
                jk_out_dim = hidden_dim * num_layers
            elif jk_mode == "lstm":
                jk_out_dim = hidden_dim
            else:
                jk_out_dim = hidden_dim
        else:
            self.jk = None
            jk_out_dim = hidden_dim
        
        # Graph-level pooling
        if pooling == "meanmax":
            pool_out_dim = jk_out_dim * 2
        else:
            pool_out_dim = jk_out_dim
        
        # MLP predictor head
        self.predictor = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: torch_geometric.data.Data or Batch object with:
                - x: Node features (num_nodes, num_node_features)
                - edge_index: Edge connectivity (2, num_edges)
                - batch: Batch vector (num_nodes,) [optional, for batched graphs]
        
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Create batch vector if not provided (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed node features
        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store initial features for residual connections
        if self.use_residual:
            x_residual = x
        
        # GIN layers with residual connections
        layer_outputs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Forward pass through GIN layer
            x_new = conv(x, edge_index)
            
            # Apply normalization
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual:
                x_new = x_new + x_residual
                x_residual = x_new
            
            x = x_new
            layer_outputs.append(x)
        
        # Jumping Knowledge (aggregate features from all layers)
        if self.use_jk:
            x = self.jk(layer_outputs)
        else:
            x = layer_outputs[-1]
        
        # Graph-level pooling
        if self.pooling == "meanmax":
            graph_repr = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        elif self.pooling == "mean":
            graph_repr = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_repr = global_max_pool(x, batch)
        elif self.pooling == "sum":
            graph_repr = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predictor head
        logits = self.predictor(graph_repr)
        
        return logits


def create_gin_model(
    num_node_features: int,
    num_edge_features: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.2,
    use_residual: bool = True,
    use_jk: bool = True,
    jk_mode: str = "cat",
    pooling: str = "meanmax",
    train_eps: bool = True,
    **kwargs
) -> GINMolecularPredictor:
    """
    Factory function to create a GINMolecularPredictor model.
    
    Args:
        num_node_features: Number of input node features
        num_edge_features: Number of input edge features (not used in GIN)
        hidden_dim: Hidden dimension size
        num_layers: Number of GIN layers
        dropout: Dropout probability
        use_residual: Whether to use residual connections
        use_jk: Whether to use Jumping Knowledge
        jk_mode: JK aggregation mode
        pooling: Pooling strategy
        train_eps: Whether to use learnable epsilon in GIN
        **kwargs: Additional arguments
    
    Returns:
        GINMolecularPredictor instance
    """
    return GINMolecularPredictor(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_residual=use_residual,
        use_jk=use_jk,
        jk_mode=jk_mode,
        pooling=pooling,
        train_eps=train_eps,
        **kwargs
    )

