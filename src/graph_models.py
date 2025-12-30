"""
Graph neural network models for molecular property prediction.

Implements GATv2-based models with attention mechanisms, hierarchical pooling,
and other advanced features for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import Set2Set, JumpingKnowledge
from typing import Optional, Literal


class AttentiveGraphPooling(nn.Module):
    """
    Attention-based graph pooling module inspired by AttentiveFP.
    
    Uses attention mechanism to aggregate node features into graph-level
    representation, allowing the model to focus on important molecular substructures.
    """
    
    def __init__(self, hidden_dim: int, num_timesteps: int = 2, dropout: float = 0.0):
        """
        Initialize AttentiveGraphPooling.
        
        Args:
            hidden_dim: Hidden dimension size
            num_timesteps: Number of attention refinement steps
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        # Attention mechanism for graph-level pooling
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.softmax_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # GRU for iterative refinement
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attentive pooling.
        
        Args:
            x: Node features of shape (num_nodes, hidden_dim)
            batch: Batch vector of shape (num_nodes,)
        
        Returns:
            Graph-level representation of shape (batch_size, hidden_dim)
        """
        # Initial graph representation (mean pooling)
        graph_repr = global_mean_pool(x, batch)
        
        # Iterative attention refinement
        for _ in range(self.num_timesteps):
            # Compute attention scores for each node
            # We create a bipartite graph: nodes -> graph representations
            num_graphs = graph_repr.size(0)
            batch_counts = torch.bincount(batch, minlength=num_graphs)
            node_to_graph = torch.arange(num_graphs, device=x.device).repeat_interleave(
                batch_counts
            )
            
            # Compute attention weights
            gate_input = x + graph_repr[node_to_graph]
            gate = torch.sigmoid(self.gate_nn(gate_input))
            gate = F.dropout(gate, p=self.dropout, training=self.training)
            
            # Weighted aggregation
            weighted_x = x * gate
            new_graph_repr = global_mean_pool(weighted_x, batch)
            
            # GRU update
            graph_repr = self.gru(new_graph_repr, graph_repr)
            graph_repr = F.relu(graph_repr)
        
        return graph_repr


class GATv2MolecularPredictor(nn.Module):
    """
    GATv2-based molecular property predictor.
    
    Architecture:
    - GATv2 layers with multi-head attention
    - Residual connections and layer normalization
    - Jumping Knowledge (optional)
    - Hierarchical pooling (Set2Set, AttentiveFP, or global pooling)
    - MLP predictor head
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_jk: bool = True,
        jk_mode: Literal["cat", "max", "mean", "lstm"] = "cat",
        pooling: Literal["set2set", "attentive", "global_mean", "global_max", "global_sum"] = "set2set",
        num_timesteps: int = 2,  # For Set2Set or AttentiveFP
        output_dim: int = 1
    ):
        """
        Initialize GATv2MolecularPredictor.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GATv2 layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_jk: Whether to use Jumping Knowledge connections
            jk_mode: JK aggregation mode ("cat", "max", "mean", "lstm")
            pooling: Pooling strategy
            num_timesteps: Number of timesteps for Set2Set/AttentiveFP
            output_dim: Output dimension (1 for binary classification)
        """
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_jk = use_jk
        self.pooling = pooling
        
        # Input projection for node features
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # Input projection for edge features (if needed)
        if num_edge_features > 0:
            self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_embedding = None
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Multi-head attention
            # Output dimension per head: hidden_dim // num_heads
            out_dim_per_head = hidden_dim // num_heads
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
            
            conv = GATv2Conv(
                in_channels=hidden_dim if i == 0 else hidden_dim,
                out_channels=out_dim_per_head,
                heads=num_heads,
                edge_dim=hidden_dim if num_edge_features > 0 else None,
                dropout=dropout,
                add_self_loops=True,
                share_weights=False,
                residual=False  # We handle residual manually
            )
            self.convs.append(conv)
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(hidden_dim))
        
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
        self.pooling = pooling
        if pooling == "set2set":
            self.pool = Set2Set(in_channels=jk_out_dim, processing_steps=num_timesteps)
            pool_out_dim = jk_out_dim * 2  # Set2Set outputs 2x input dimension
        elif pooling == "attentive":
            self.pool = AttentiveGraphPooling(hidden_dim=jk_out_dim, num_timesteps=num_timesteps, dropout=dropout)
            pool_out_dim = jk_out_dim
        else:
            # For simple pooling methods, we'll handle them in forward
            self.pool = None
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
                - edge_attr: Edge features (num_edges, num_edge_features) [optional]
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
        
        # Embed edge features if available
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(data.edge_attr)
            edge_attr = F.relu(edge_attr)
        
        # Store initial features for residual connections
        if self.use_residual:
            x_residual = x
        
        # GATv2 layers with residual connections
        layer_outputs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Forward pass through GATv2 layer
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            
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
        if self.pool is not None:
            graph_repr = self.pool(x, batch)
        elif self.pooling == "global_mean":
            graph_repr = global_mean_pool(x, batch)
        elif self.pooling == "global_max":
            graph_repr = global_max_pool(x, batch)
        elif self.pooling == "global_sum":
            graph_repr = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predictor head
        logits = self.predictor(graph_repr)
        
        return logits


def create_gatv2_model(
    num_node_features: int,
    num_edge_features: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    use_residual: bool = True,
    use_jk: bool = True,
    jk_mode: str = "cat",
    pooling: str = "set2set",
    **kwargs
) -> GATv2MolecularPredictor:
    """
    Factory function to create a GATv2MolecularPredictor model.
    
    Args:
        num_node_features: Number of input node features
        num_edge_features: Number of input edge features
        hidden_dim: Hidden dimension size
        num_layers: Number of GATv2 layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_residual: Whether to use residual connections
        use_jk: Whether to use Jumping Knowledge
        jk_mode: JK aggregation mode
        pooling: Pooling strategy
        **kwargs: Additional arguments
    
    Returns:
        GATv2MolecularPredictor instance
    """
    return GATv2MolecularPredictor(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_residual=use_residual,
        use_jk=use_jk,
        jk_mode=jk_mode,
        pooling=pooling,
        **kwargs
    )

