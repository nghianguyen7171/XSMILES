# SMILESGNN Architecture

This document describes the architecture of the SMILESGNN (SMILES-Graph Neural Network) model, a multimodal deep learning model for molecular property prediction that combines sequence-based and graph-based representations.

## Overview

SMILESGNN combines two complementary molecular representations:
- **SMILES Sequence**: Processed by a Transformer encoder to capture sequential patterns and long-range dependencies
- **Graph Structure**: Processed by a Graph Neural Network (GNN) to capture local structural information and explicit connectivity

The model uses an attention-based fusion mechanism to combine both representations, achieving superior performance compared to either representation alone.

## Data Flow: From Raw SMILES to Model Inputs

The architecture starts with **raw SMILES strings** (e.g., "CCO", "CCN") from the dataset. Each SMILES string is processed through **two parallel pipelines** to create the dual inputs required by SMILESGNN:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Raw Data: SMILES String                                   │
│                    Example: "CCO" (ethanol)                                 │
└────────────────────────────┬──────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Pipeline 1:          │   │  Pipeline 2:          │
    │  SMILES Sequence      │   │  Molecular Graph      │
    │  Processing           │   │  Processing           │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  SMILESTokenizer      │   │  RDKit                 │
    │  - Tokenize:          │   │  - Parse SMILES        │
    │    "CCO" →            │   │  - Create Molecule     │
    │    ["C","C","O"]      │   │    object              │
    │  - Encode to IDs      │   │                        │
    │  - Create masks       │   │  - Extract atoms       │
    │                        │   │  - Extract bonds      │
    └───────────┬───────────┘   └───────────┬───────────┘
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Output:              │   │  Output:               │
    │  - Token IDs          │   │  - Node Features       │
    │    (batch, seq_len)   │   │    (num_nodes, 25)     │
    │  - Attention Mask     │   │  - Edge Features       │
    │    (batch, seq_len)   │   │    (num_edges, 17)     │
    │                       │   │  - Edge Index          │
    │                       │   │    (2, num_edges)      │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                └───────────┬───────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  SMILESGNN Model       │
                │  (Dual Input)          │
                └───────────────────────┘
```

### Pipeline 1: SMILES Sequence Processing

**Input:** Raw SMILES string (e.g., "CCO")

**Steps:**
1. **Tokenization** (`SMILESTokenizer._tokenize_smiles()`)
   - Breaks SMILES into tokens using regex patterns
   - Example: "CCO" → ["C", "C", "O"]
   - Handles special patterns: rings, bonds, branching, aromatic atoms

2. **Vocabulary Mapping** (`SMILESTokenizer.encode()`)
   - Maps tokens to integer IDs using learned vocabulary
   - Adds special tokens: `<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`
   - Example: ["C", "C", "O"] → [id_C, id_C, id_O]

3. **Padding/Truncation**
   - Pads sequences to `max_length=128`
   - Creates attention mask (1 for real tokens, 0 for padding)

**Output:**
- `token_ids`: `(batch_size, seq_len)` - Integer token IDs
- `attention_mask`: `(batch_size, seq_len)` - Mask for valid tokens

**Implementation:** `src/smiles_tokenizer.py`

### Pipeline 2: Molecular Graph Processing

**Input:** Raw SMILES string (e.g., "CCO")

**Steps:**
1. **SMILES Parsing** (`smiles_to_mol()`)
   - Uses RDKit to parse SMILES into molecule object
   - Validates and sanitizes molecular structure

2. **Node Feature Extraction** (`get_atom_features()`)
   - Extracts 25 features per atom:
     - Atomic number (one-hot, 10 dims: C, N, O, F, P, S, Cl, Br, I, Other)
     - Formal charge (1 dim)
     - Hybridization (one-hot, 5 dims: SP, SP2, SP3, SP3D, SP3D2)
     - Chirality (one-hot, 3 dims)
     - Ring membership (1 dim)
     - Aromaticity (1 dim)
     - Number of heavy atom neighbors (1 dim)
     - Number of hydrogen neighbors (1 dim)
     - Valence minus attached hydrogens (1 dim)
     - Degree (1 dim)

3. **Edge Feature Extraction** (`get_bond_features()`)
   - Extracts 17 features per bond:
     - Bond type (one-hot, 4 dims: single, double, triple, aromatic)
     - Bond direction (one-hot, 5 dims)
     - Ring membership (1 dim)
     - Conjugation (1 dim)
     - Stereochemistry (one-hot, 6 dims)

4. **Edge Index Construction**
   - Creates connectivity matrix in COO format
   - Each bond creates two edges (undirected graph)
   - Shape: `(2, num_edges)`

**Output:**
- `x`: `(num_nodes, 25)` - Node (atom) feature matrix
- `edge_index`: `(2, num_edges)` - Edge connectivity
- `edge_attr`: `(num_edges, 17)` - Edge (bond) feature matrix

**Implementation:** `src/graph_data.py`

### Key Insight

Both pipelines process the **same SMILES string** but create different representations:
- **Sequence representation**: Linear order of tokens (captures sequential patterns)
- **Graph representation**: Explicit atom-atom connections (captures structural patterns)

This dual representation enables SMILESGNN to leverage both sequential and structural information.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SMILESGNN Architecture                             │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: SMILES String              Input: Molecular Graph
    ┌──────────────────────┐          ┌──────────────────────┐
    │ Token IDs            │          │ Node Features        │
    │ (batch, seq_len)     │          │ (num_nodes, 25)      │
    │                      │          │ Edge Features        │
    │ Attention Mask       │          │ (num_edges, 17)      │
    │ (batch, seq_len)     │          │ Edge Index           │
    └──────────┬───────────┘          │ (2, num_edges)       │
               │                      └──────────┬───────────┘
               │                                  │
               │                                  │
    ┌──────────▼──────────────────────────────────▼──────────┐
    │                                                         │
    │  ┌────────────────────┐    ┌─────────────────────┐   │
    │  │  SMILES ENCODER    │    │   GRAPH ENCODER     │   │
    │  │  (Transformer)     │    │   (GNN: GATv2/GIN)  │   │
    │  └────────────────────┘    └─────────────────────┘   │
    │                                                         │
    └─────────────────────┬───────────────────┬──────────────┘
                          │                   │
              ┌───────────▼────────┐  ┌───────▼─────────┐
              │ SMILES Repr        │  │ Graph Repr      │
              │ (batch, d_model)   │  │ (batch, repr_dim)│
              │ d_model = 96       │  │ repr_dim = 192  │
              └──────────┬─────────┘  │ (meanmax pool)  │
                         │            └────────┬────────┘
                         │                     │
                         │    ┌────────────────▼──────────────┐
                         │    │   Fusion Module               │
                         │    │   (Attention-based)           │
                         │    │                               │
                         │    │  Query: SMILES Repr           │
                         │    │  Key/Value: Graph Repr (proj) │
                         │    │                               │
                         │    │  Output: Fused Repr           │
                         │    │  (batch, d_model * 2 = 192)   │
                         │    └───────────────┬───────────────┘
                         │                    │
                         └────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Predictor Head    │
                          │  (MLP: 4 layers)   │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Output            │
                          │  (batch, 1)        │
                          │  Logits            │
                          └────────────────────┘
```

## Component Details

### 1. SMILES Encoder (SimpleSMILESEncoder)

The SMILES encoder processes tokenized SMILES sequences using a Transformer architecture.

```
Input: Token IDs (batch_size, seq_len)
       Attention Mask (batch_size, seq_len)
       
┌─────────────────────────────────────────────┐
│ 1. Token Embedding                         │
│    Embedding(vocab_size=100, d_model=96)   │
│    → (batch, seq_len, d_model)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. Positional Encoding                      │
│    Learnable positional embeddings          │
│    → (batch, seq_len, d_model)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. Transformer Encoder                      │
│    - num_layers = 2                         │
│    - nhead = 4                              │
│    - dim_feedforward = 192                  │
│    - dropout = 0.4                          │
│    → (batch, seq_len, d_model)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. Mean Pooling (masked)                    │
│    Average over sequence length             │
│    → (batch, d_model) = (batch, 96)        │
└─────────────────────────────────────────────┘
```

**Key Features:**
- Vocabulary size: 100 tokens
- Model dimension (d_model): 96
- Number of transformer layers: 2
- Attention heads: 4
- Feedforward dimension: 192 (d_model * 2)
- Pooling: Mean pooling with attention mask support

### 2. Graph Encoder

The graph encoder processes molecular graphs using a Graph Neural Network (GATv2 or GIN).

```
Input: Node Features (num_nodes, 25)
       Edge Features (num_edges, 17)
       Edge Index (2, num_edges)
       Batch Index
       
┌─────────────────────────────────────────────┐
│ 1. Node Embedding                           │
│    Linear(25 → 96)                          │
│    ReLU + Dropout                           │
│    → (num_nodes, hidden_dim=96)            │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. Edge Embedding (if edge features)        │
│    Linear(17 → 96)                          │
│    ReLU                                     │
│    → (num_edges, hidden_dim=96)            │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. GNN Layers (num_graph_layers = 3)        │
│    ┌───────────────────────────────────┐   │
│    │ For each layer i:                 │   │
│    │  - GATv2Conv or GINConv           │   │
│    │  - LayerNorm                      │   │
│    │  - ReLU                           │   │
│    │  - Dropout                        │   │
│    │  - Residual connection (optional) │   │
│    │  → (num_nodes, hidden_dim=96)    │   │
│    └───────────────────────────────────┘   │
│                                             │
│    Layer outputs: [h₁, h₂, h₃]             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. Jumping Knowledge (JK)                   │
│    Mode: concatenation                      │
│    JK([h₁, h₂, h₃])                        │
│    → (num_nodes, hidden_dim * 3 = 288)     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 5. Graph-Level Pooling                      │
│    Mean-Max Pooling (concatenated)          │
│    [mean(h), max(h)]                        │
│    → (batch, 288 * 2 = 576)                │
│    With JK (cat): hidden_dim * layers * 2  │
│    96 * 3 * 2 = 576                        │
└─────────────────────────────────────────────┘
```

**Key Features:**
- Number of node features: 25
- Number of edge features: 17
- Hidden dimension: 96
- Number of GNN layers: 3
- Graph model: GATv2 (default) with 4 attention heads
- Residual connections: Enabled
- Jumping Knowledge: Enabled (concatenation mode)
- Graph pooling: Mean-Max pooling (concatenated)
- Output dimension: 576 (hidden_dim * num_layers * 2 = 96 * 3 * 2)

**Note:** The actual implementation uses `hidden_dim=96` and `num_graph_layers=3`. With JK concatenation and meanmax pooling, the graph representation dimension is `96 * 3 * 2 = 576`, but it gets projected in the fusion module.

### 3. Fusion Module

The fusion module combines SMILES and graph representations using attention mechanism.

```
Input: SMILES Repr (batch, 96)
       Graph Repr (batch, 576)  [with JK and meanmax]
       
┌─────────────────────────────────────────────┐
│ 1. Project Graph Repr                       │
│    Linear(576 → 96)                         │
│    → (batch, 96)                            │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. Expand Dimensions                        │
│    Graph Repr: (batch, 1, 96)               │
│    SMILES Repr: (batch, 1, 96)              │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. Multi-Head Attention                     │
│    Query: SMILES Repr (batch, 1, 96)        │
│    Key: Graph Repr (batch, 1, 96)           │
│    Value: Graph Repr (batch, 1, 96)         │
│    num_heads = 4                             │
│    → Attended Graph Repr (batch, 1, 96)     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. Concatenation                            │
│    [SMILES Repr, Attended Graph Repr]       │
│    → (batch, 192)                           │
└─────────────────────────────────────────────┘
```

**Key Features:**
- Fusion method: Attention-based (default)
- Query: SMILES representation
- Key/Value: Projected graph representation
- Attention heads: 4
- Output dimension: 192 (d_model * 2)

### 4. Predictor Head

The predictor head is a multi-layer perceptron that produces the final prediction.

```
Input: Fused Repr (batch, 192)

┌─────────────────────────────────────────────┐
│ Layer 1                                     │
│ Linear(192 → 192)                           │
│ BatchNorm1d                                 │
│ ReLU                                        │
│ Dropout(0.4)                                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Layer 2                                     │
│ Linear(192 → 96)                            │
│ BatchNorm1d                                 │
│ ReLU                                        │
│ Dropout(0.4)                                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Layer 3                                     │
│ Linear(96 → 48)                             │
│ BatchNorm1d                                 │
│ ReLU                                        │
│ Dropout(0.4)                                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Layer 4 (Output)                            │
│ Linear(48 → 1)                              │
│ → (batch, 1) Logits                         │
└─────────────────────────────────────────────┘
```

**Key Features:**
- Number of layers: 4
- Architecture: 192 → 192 → 96 → 48 → 1
- Activation: ReLU (except output layer)
- Regularization: BatchNorm + Dropout (0.4) at each hidden layer
- Output: Single logit for binary classification

## Forward Pass Flow

```
forward(data, smiles_token_ids, smiles_attention_mask):
    
    1. Encode Graph:
       graph_repr = encode_graph(data)
       → (batch, 576)
    
    2. Encode SMILES:
       smiles_repr = smiles_encoder(smiles_token_ids, smiles_attention_mask)
       → (batch, 96)
    
    3. Fusion (Attention-based):
       graph_repr_proj = Linear(576 → 96)(graph_repr)  # (batch, 96)
       attended_graph = MultiHeadAttention(
           query=smiles_repr,  # (batch, 1, 96)
           key=graph_repr_proj,  # (batch, 1, 96)
           value=graph_repr_proj  # (batch, 1, 96)
       )  # → (batch, 1, 96)
       fused_repr = concat([smiles_repr, attended_graph])  # (batch, 192)
    
    4. Predict:
       logits = predictor(fused_repr)  # (batch, 1)
    
    return logits
```

## Key Hyperparameters

Based on the configuration file (`config/smilesgnn_config.yaml`):

- **Graph Encoder:**
  - `hidden_dim`: 96
  - `num_graph_layers`: 3
  - `graph_model`: "gatv2"
  - `num_heads`: 4
  - `use_residual`: True
  - `use_jk`: True
  - `jk_mode`: "cat"
  - `graph_pooling`: "meanmax"

- **SMILES Encoder:**
  - `smiles_d_model`: 96
  - `smiles_num_layers`: 2
  - `vocab_size`: 100 (from tokenizer)

- **Fusion:**
  - `fusion_method`: "attention"

- **Regularization:**
  - `dropout`: 0.4
  - `weight_decay`: 0.0001

- **Training:**
  - Loss: Focal Loss (alpha=0.25, gamma=2.0)
  - Optimizer: AdamW
  - Learning rate: 0.0005
  - Batch size: 32

## Model Dimensions Summary

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| Token Embedding | (batch, seq_len) | (batch, seq_len, 96) | vocab_size × 96 |
| SMILES Encoder | (batch, seq_len, 96) | (batch, 96) | ~200K |
| Node Embedding | (num_nodes, 25) | (num_nodes, 96) | 25 × 96 |
| Edge Embedding | (num_edges, 17) | (num_edges, 96) | 17 × 96 |
| GNN Layers (×3) | (num_nodes, 96) | (num_nodes, 96) | ~150K |
| Graph Pooling | (num_nodes, 288) | (batch, 576) | - |
| Graph Projection | (batch, 576) | (batch, 96) | 576 × 96 |
| Fusion (Attention) | (batch, 96), (batch, 96) | (batch, 192) | ~37K |
| Predictor | (batch, 192) | (batch, 1) | ~50K |

**Total Parameters:** ~437K

## Implementation Details

- **Framework:** PyTorch + PyTorch Geometric
- **Graph Model:** GATv2 (Graph Attention Network v2) or GIN (Graph Isomorphism Network)
- **SMILES Tokenizer:** Custom tokenizer with vocabulary built from training data
- **Device:** CPU/GPU compatible
- **Batch Processing:** Supports batched graph processing via PyTorch Geometric

## Advantages of Multimodal Fusion

1. **Complementary Representations:**
   - SMILES: Captures global sequential patterns and long-range dependencies
   - Graph: Captures local structural information and explicit connectivity

2. **Attention-Based Fusion:**
   - Allows the model to dynamically weigh the importance of each representation
   - SMILES representation acts as query, focusing attention on relevant graph features

3. **Robustness:**
   - Can handle cases where one representation might be ambiguous
   - Cross-modal validation improves prediction confidence

## References

- Model implementation: `src/graph_models_hybrid.py`
- Configuration: `config/smilesgnn_config.yaml`
- Training script: `scripts/train_hybrid.py`

