"""
Hybrid SMILES + Graph model for molecular property prediction.

Combines sequence-based (SMILES Transformer) and graph-based (GNN) representations
to leverage the strengths of both approaches:
- SMILES: Global patterns, long-range dependencies via attention
- Graph: Local structural information, explicit connectivity

This multimodal fusion approach aims to achieve better performance than either
representation alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import JumpingKnowledge, BatchNorm
from typing import Optional, Literal

# Try to import transformers for SMILES encoding
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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


class SimpleSMILESEncoder(nn.Module):
    """
    Simple Transformer-based encoder for SMILES strings.
    
    Uses a transformer encoder to process SMILES token sequences
    and produce a fixed-size molecular representation.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,  # SMILES vocabulary size
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output normalization — controls magnitude before cross-attention
        self.output_norm = nn.LayerNorm(d_model)
        
        # Pooling layer (CLS token approach - use first token or mean pooling)
        self.pooling = 'mean'  # 'cls', 'mean', 'max'
    
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
        
        Returns:
            Molecular representation of shape (batch_size, d_model)
        """
        # Embed tokens
        x = self.token_embedding(token_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        # Create mask for padding (if attention_mask provided)
        if attention_mask is not None:
            # Convert attention mask to src_key_padding_mask format
            # True means ignore, False means attend
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool to get graph-level representation
        if self.pooling == 'cls':
            # Use first token (CLS token)
            graph_repr = x[:, 0, :]
        elif self.pooling == 'mean':
            if attention_mask is not None:
                # Mean pooling with attention mask
                mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                x_masked = x * mask
                graph_repr = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                graph_repr = x.mean(dim=1)
        else:  # max
            graph_repr = x.max(dim=1)[0]
        
        return self.output_norm(graph_repr)


class SMILESGraphHybridPredictor(nn.Module):
    """
    Hybrid model combining SMILES (sequence) and Graph representations.
    
    Architecture:
    1. SMILES Encoder: Transformer-based encoder for SMILES sequences
    2. Graph Encoder: GNN (GATv2 or GIN) for graph structures
    3. Fusion Module: Combines both representations (concatenation or attention)
    4. Predictor: MLP head for final prediction
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_graph_layers: int = 4,
        graph_model: Literal["gatv2", "gin"] = "gatv2",
        num_heads: int = 4,  # For GATv2
        dropout: float = 0.2,
        use_residual: bool = True,
        use_jk: bool = True,
        jk_mode: str = "cat",
        graph_pooling: str = "meanmax",  # mean, max, sum, meanmax
        smiles_vocab_size: int = 100,
        smiles_d_model: int = 128,
        smiles_num_layers: int = 3,
        fusion_method: Literal["concat", "attention", "weighted"] = "attention",
        output_dim: int = 1
    ):
        """
        Initialize SMILESGraphHybridPredictor.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_dim: Hidden dimension size (for graph encoder)
            num_graph_layers: Number of GNN layers
            graph_model: Graph model type ("gatv2" or "gin")
            num_heads: Number of attention heads for GATv2
            dropout: Dropout probability
            use_residual: Whether to use residual connections in graph encoder
            use_jk: Whether to use Jumping Knowledge in graph encoder
            jk_mode: JK aggregation mode
            graph_pooling: Graph-level pooling strategy
            smiles_vocab_size: SMILES vocabulary size
            smiles_d_model: SMILES encoder dimension
            smiles_num_layers: Number of transformer layers in SMILES encoder
            fusion_method: How to fuse SMILES and graph representations
            output_dim: Output dimension (1 for binary classification)
        """
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.smiles_d_model = smiles_d_model
        self.fusion_method = fusion_method
        
        # SMILES Encoder
        self.smiles_encoder = SimpleSMILESEncoder(
            vocab_size=smiles_vocab_size,
            d_model=smiles_d_model,
            nhead=4,
            num_layers=smiles_num_layers,
            dim_feedforward=smiles_d_model * 2,
            dropout=dropout,
            max_seq_len=128
        )
        
        # Graph Encoder - Node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # Graph Encoder - GNN layers
        self.graph_model_type = graph_model
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if graph_model == "gatv2":
            out_dim_per_head = hidden_dim // num_heads
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
            
            for i in range(num_graph_layers):
                conv = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=out_dim_per_head,
                    heads=num_heads,
                    edge_dim=hidden_dim if num_edge_features > 0 else None,
                    dropout=dropout,
                    add_self_loops=True,
                    share_weights=False,
                    residual=False
                )
                self.convs.append(conv)
                self.norms.append(nn.LayerNorm(hidden_dim))
        else:  # gin
            for i in range(num_graph_layers):
                mlp = MLP(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim * 2,
                    output_dim=hidden_dim,
                    num_layers=2,
                    dropout=dropout
                )
                from torch_geometric.nn import GINConv
                conv = GINConv(nn=mlp, train_eps=True)
                self.convs.append(conv)
                self.norms.append(BatchNorm(hidden_dim))
        
        # Edge embedding for graph (if needed)
        if num_edge_features > 0 and graph_model == "gatv2":
            self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_embedding = None
        
        # Jumping Knowledge
        if use_jk:
            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_graph_layers)
            if jk_mode == "cat":
                graph_out_dim = hidden_dim * num_graph_layers
            elif jk_mode == "lstm":
                graph_out_dim = hidden_dim
            else:
                graph_out_dim = hidden_dim
        else:
            self.jk = None
            graph_out_dim = hidden_dim
        
        # Graph pooling
        self.graph_pooling = graph_pooling
        if graph_pooling == "meanmax":
            graph_repr_dim = graph_out_dim * 2
        else:
            graph_repr_dim = graph_out_dim
        
        # Fusion Module
        if fusion_method == "concat":
            # Simple concatenation
            fused_dim = smiles_d_model + graph_repr_dim
            self.fusion = None
        elif fusion_method == "attention":
            # Attention-based fusion
            self.fusion = nn.MultiheadAttention(
                embed_dim=smiles_d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            # Project graph to smiles dimension + normalize for cross-attention stability
            self.graph_proj = nn.Linear(graph_repr_dim, smiles_d_model)
            self.graph_proj_norm = nn.LayerNorm(smiles_d_model)
            fused_dim = smiles_d_model * 2  # [smiles_repr, attended_graph_repr]
        elif fusion_method == "weighted":
            # Learned weighted combination
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
            # Project both to same dimension
            self.smiles_proj = nn.Linear(smiles_d_model, hidden_dim)
            self.graph_proj = nn.Linear(graph_repr_dim, hidden_dim)
            fused_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
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
        
        self.use_residual = use_residual
        self.dropout_param = dropout
    
    def encode_graph(self, data) -> torch.Tensor:
        """Encode graph representation."""
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed node features
        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_param, training=self.training)
        
        # Embed edge features if available
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(data.edge_attr)
            edge_attr = F.relu(edge_attr)
        
        x_residual = x if self.use_residual else None
        
        # GNN layers
        layer_outputs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.graph_model_type == "gatv2":
                x_new = conv(x, edge_index, edge_attr=edge_attr)
            else:  # gin
                x_new = conv(x, edge_index)
            
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout_param, training=self.training)
            
            if self.use_residual:
                x_new = x_new + x_residual
                x_residual = x_new
            
            x = x_new
            layer_outputs.append(x)
        
        # Jumping Knowledge
        if self.jk:
            x = self.jk(layer_outputs)
        else:
            x = layer_outputs[-1]
        
        # Graph-level pooling
        if self.graph_pooling == "meanmax":
            graph_repr = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        elif self.graph_pooling == "mean":
            graph_repr = global_mean_pool(x, batch)
        elif self.graph_pooling == "max":
            graph_repr = global_max_pool(x, batch)
        else:  # sum
            from torch_geometric.nn import global_add_pool
            graph_repr = global_add_pool(x, batch)
        
        return graph_repr
    
    def forward(
        self,
        data,
        smiles_token_ids: Optional[torch.Tensor] = None,
        smiles_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: torch_geometric.data.Data or Batch object with graph information
            smiles_token_ids: Token IDs for SMILES strings of shape (batch_size, seq_len)
            smiles_attention_mask: Attention mask for SMILES of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        # Encode graph
        graph_repr = self.encode_graph(data)
        
        # Encode SMILES
        if smiles_token_ids is not None:
            smiles_repr = self.smiles_encoder(smiles_token_ids, smiles_attention_mask)
        else:
            # If SMILES not provided, use zero vector
            batch_size = graph_repr.size(0)
            smiles_repr = torch.zeros(batch_size, self.smiles_d_model, device=graph_repr.device)
        
        # Fusion
        if self.fusion_method == "concat":
            fused_repr = torch.cat([smiles_repr, graph_repr], dim=1)
        elif self.fusion_method == "attention":
            # Use SMILES as query, graph as key/value
            # LayerNorm on graph projection prevents cross-attention overflow on CPU
            graph_repr_proj = self.graph_proj_norm(self.graph_proj(graph_repr))  # (batch_size, smiles_d_model)
            graph_repr_proj = graph_repr_proj.unsqueeze(1)  # (batch_size, 1, smiles_d_model)
            smiles_repr_expanded = smiles_repr.unsqueeze(1)  # (batch_size, 1, smiles_d_model)
            
            attended_graph, _ = self.fusion(
                smiles_repr_expanded,
                graph_repr_proj,
                graph_repr_proj
            )
            attended_graph = attended_graph.squeeze(1)  # (batch_size, smiles_d_model)
            
            fused_repr = torch.cat([smiles_repr, attended_graph], dim=1)
        else:  # weighted
            smiles_proj = self.smiles_proj(smiles_repr)
            graph_proj = self.graph_proj(graph_repr)
            
            weights = F.softmax(self.fusion_weight, dim=0)
            fused_repr = weights[0] * smiles_proj + weights[1] * graph_proj
        
        # Predictor
        logits = self.predictor(fused_repr)
        
        # Clamp logits to prevent NaN from extreme values (CPU stability)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        return logits


def create_hybrid_model(
    num_node_features: int,
    num_edge_features: int,
    hidden_dim: int = 128,
    num_graph_layers: int = 4,
    graph_model: str = "gatv2",
    num_heads: int = 4,
    dropout: float = 0.2,
    use_residual: bool = True,
    use_jk: bool = True,
    jk_mode: str = "cat",
    graph_pooling: str = "meanmax",
    smiles_vocab_size: int = 100,
    smiles_d_model: int = 128,
    smiles_num_layers: int = 3,
    fusion_method: str = "attention",
    **kwargs
) -> SMILESGraphHybridPredictor:
    """
    Factory function to create a SMILESGraphHybridPredictor model.
    
    Args:
        num_node_features: Number of input node features
        num_edge_features: Number of input edge features
        hidden_dim: Hidden dimension size
        num_graph_layers: Number of GNN layers
        graph_model: Graph model type ("gatv2" or "gin")
        num_heads: Number of attention heads for GATv2
        dropout: Dropout probability
        use_residual: Whether to use residual connections
        use_jk: Whether to use Jumping Knowledge
        jk_mode: JK aggregation mode
        graph_pooling: Graph pooling strategy
        smiles_vocab_size: SMILES vocabulary size
        smiles_d_model: SMILES encoder dimension
        smiles_num_layers: Number of transformer layers
        fusion_method: Fusion method ("concat", "attention", "weighted")
        **kwargs: Additional arguments
    
    Returns:
        SMILESGraphHybridPredictor instance
    """
    return SMILESGraphHybridPredictor(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_graph_layers=num_graph_layers,
        graph_model=graph_model,
        num_heads=num_heads,
        dropout=dropout,
        use_residual=use_residual,
        use_jk=use_jk,
        jk_mode=jk_mode,
        graph_pooling=graph_pooling,
        smiles_vocab_size=smiles_vocab_size,
        smiles_d_model=smiles_d_model,
        smiles_num_layers=smiles_num_layers,
        fusion_method=fusion_method,
        **kwargs
    )

