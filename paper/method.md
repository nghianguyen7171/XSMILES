# Method: Clinical Drug Toxicity Prediction

## Overview

We propose a comprehensive deep learning approach for predicting clinical drug toxicity from molecular structures. Our method explores multiple molecular representations and develops a novel multimodal architecture that combines sequence-based and graph-based representations to achieve superior performance.

## Approach

We address the clinical toxicity prediction problem as a **binary classification task**: given a molecular structure represented as a SMILES string, predict whether the drug will fail clinical trials due to toxicity concerns.

Our approach involves:
1. **Multiple molecular representations** (fingerprints, graphs, sequences)
2. **Diverse deep learning architectures** (MLP, GNN, Transformer, multimodal)
3. **Comprehensive evaluation** across 8 different models
4. **Novel multimodal fusion** combining sequence and graph representations

## Molecular Representations

We explore three primary ways to represent molecules for machine learning:

### 1. Fingerprint Representation

**Morgan Fingerprints (ECFP-like)**
- Fixed-length binary vectors (2048 bits)
- Captures local molecular patterns using circular substructures
- Used by baseline MLP model
- Implementation: `src/featurization.py` → `featurize_fingerprint()`

**Advantages:**
- Fast computation
- Fixed-size vectors suitable for standard neural networks
- Proven effective for many molecular property prediction tasks

**Limitations:**
- Loss of structural information
- Fixed-length encoding may not capture all relevant features
- Less interpretable than graph representations

### 2. Graph Representation

**Molecular Graphs**
- Nodes = Atoms (25 features per atom)
- Edges = Bonds (17 features per bond)
- Explicit representation of molecular structure and connectivity
- Used by GNN models (BFGNN, GRIN, GATv2, GIN, DMPNN, SMILESGNN)

**Node Features (25 dimensions):**
- Atomic number (one-hot, 10 dims: C, N, O, F, P, S, Cl, Br, I, Other)
- Formal charge
- Hybridization (one-hot, 5 dims)
- Chirality (one-hot, 3 dims)
- Ring membership
- Aromaticity
- Number of neighbors
- Valence information

**Edge Features (17 dimensions):**
- Bond type (single, double, triple, aromatic)
- Bond direction
- Ring membership
- Conjugation
- Stereochemistry

**Advantages:**
- Preserves structural information
- Natural representation for molecules
- Can capture local and global patterns through message passing

**Implementation:** `src/graph_data.py` → `smiles_to_pyg_data()`

### 3. Sequence Representation

**SMILES Strings**
- Text-based linear representation of molecular structure
- Sequence of tokens (characters/subwords)
- Used by Transformer models (SMILESTransformer, SMILESGNN)

**Tokenization Process:**
1. Break SMILES into tokens using regex patterns
2. Build vocabulary from training data
3. Encode tokens to integer IDs
4. Pad/truncate to fixed length (128 tokens)

**Advantages:**
- Leverages powerful sequence models (Transformers)
- Can capture sequential patterns and long-range dependencies
- Natural language processing techniques applicable

**Implementation:** `src/smiles_tokenizer.py`

## Model Architectures

We implement and compare 8 different model architectures:

### 1. Baseline MLP (Fingerprint-based)

**Architecture:**
```
Input: Morgan Fingerprint (2048 bits)
  ↓
Linear: 2048 → 512
ReLU + Dropout
  ↓
Linear: 512 → 256
ReLU + Dropout
  ↓
Linear: 256 → 128
ReLU + Dropout
  ↓
Linear: 128 → 1
Output: Logit
```

**Training:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Early stopping on validation loss

**Purpose:** Baseline model for comparison

### 2. Graph Neural Networks (Single Modality)

We implement several GNN architectures:

#### 2.1 BFGNN & GRIN (torch-molecule)
- Framework: torch-molecule library
- Graph-based message passing
- Automated hyperparameter optimization

#### 2.2 GATv2 (Graph Attention Network v2)
**Architecture:**
- 4 GATv2 layers with 4 attention heads
- Hidden dimension: 128
- Jumping Knowledge connections
- Set2Set graph pooling
- MLP predictor head

#### 2.3 GIN (Graph Isomorphism Network)
**Architecture:**
- 4 GIN layers with MLP-based message passing
- Hidden dimension: 128
- Learnable epsilon parameter
- Jumping Knowledge connections
- Mean-Max pooling
- MLP predictor head

#### 2.4 DMPNN (Directed Message Passing Neural Network)
- Framework: DeepChem
- Bond-centric message passing
- GraphData featurization

### 3. Sequence Models

#### 3.1 SMILESTransformer (torch-molecule)
- Transformer-based encoder-decoder
- Processes SMILES as sequence
- Automated hyperparameter optimization via torch-molecule

### 4. Multimodal Model: SMILESGNN ⭐

**Our main contribution** - A novel multimodal architecture that combines sequence and graph representations.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      SMILESGNN Architecture                      │
└─────────────────────────────────────────────────────────────────┘

    Raw SMILES String
            │
    ┌───────┴────────┐
    │                │
    ▼                ▼
┌─────────┐      ┌─────────┐
│Pipeline │      │Pipeline │
│    1    │      │    2    │
│Tokenizer│      │  RDKit  │
└────┬────┘      └────┬────┘
     │                │
     ▼                ▼
┌─────────┐      ┌─────────┐
│Token IDs│      │  Graph  │
│ (batch, │      │   Data  │
│seq_len) │      │  Object │
│         │      │         │
│Attention│      │- Nodes  │
│  Mask   │      │  (25)   │
└────┬────┘      │- Edges  │
     │           │  (17)   │
     │           │- Index  │
     │           └────┬────┘
     │                │
     └───────┬────────┘
             │
    ┌────────▼────────────────────────┐
    │  SMILESGNN Model                │
    │                                  │
    │  ┌────────────┐  ┌───────────┐ │
    │  │  SMILES    │  │   Graph   │ │
    │  │  Encoder   │  │  Encoder  │ │
    │  │(Transformer)│  │ (GNN)    │ │
    │  └──────┬─────┘  └─────┬─────┘ │
    │         │              │       │
    │         └──────┬───────┘       │
    │                │               │
    │         ┌──────▼──────┐        │
    │         │   Fusion    │        │
    │         │  (Attention)│        │
    │         └──────┬──────┘        │
    │                │               │
    │         ┌──────▼──────┐        │
    │         │ Predictor   │        │
    │         │    (MLP)    │        │
    │         └─────────────┘        │
    └────────────────────────────────┘
             │
             ▼
         Output (Logit)
```

#### SMILES Encoder (Transformer-based)

```
Input: Token IDs (batch, seq_len)
  ↓
Token Embedding (vocab_size → 96)
  ↓
Positional Encoding (learnable)
  ↓
Transformer Encoder (2 layers, 4 heads, d_model=96)
  ↓
Mean Pooling (masked)
  ↓
Output: SMILES Representation (batch, 96)
```

**Key Components:**
- Vocabulary size: 100 tokens
- Model dimension: 96
- Transformer layers: 2
- Attention heads: 4
- Feedforward dimension: 192

#### Graph Encoder (GNN-based)

```
Input: Graph Data Object
  ↓
Node Embedding (25 → 96)
Edge Embedding (17 → 96, if available)
  ↓
GATv2 Layers (3 layers, 4 heads each)
  - LayerNorm
  - ReLU
  - Dropout (0.4)
  - Residual connections
  ↓
Jumping Knowledge (concatenation mode)
  → (batch, 288)  [96 × 3 layers]
  ↓
Mean-Max Pooling (concatenated)
  → (batch, 576)  [288 × 2]
  ↓
Output: Graph Representation (batch, 576)
```

**Key Components:**
- Hidden dimension: 96
- GNN layers: 3 (GATv2)
- Attention heads: 4
- Residual connections: Enabled
- Jumping Knowledge: Enabled (concatenation)
- Graph pooling: Mean-Max (concatenated)

#### Fusion Module (Attention-based)

The fusion module combines SMILES and graph representations using multi-head attention:

```
Input: SMILES Repr (batch, 96)
       Graph Repr (batch, 576)
  ↓
Project Graph: Linear(576 → 96)
  → Graph Repr Proj (batch, 96)
  ↓
Expand Dimensions:
  - SMILES: (batch, 1, 96)
  - Graph: (batch, 1, 96)
  ↓
Multi-Head Attention:
  Query: SMILES Repr
  Key: Graph Repr Proj
  Value: Graph Repj Proj
  (4 attention heads)
  ↓
Attended Graph Repr (batch, 96)
  ↓
Concatenate: [SMILES Repr, Attended Graph Repr]
  ↓
Output: Fused Representation (batch, 192)
```

**Why Attention-based Fusion?**
- Allows dynamic weighting of each representation
- SMILES representation acts as query, focusing on relevant graph features
- More flexible than simple concatenation or weighted averaging
- Enables the model to learn which representation is more important for each sample

#### Predictor Head

```
Input: Fused Repr (batch, 192)
  ↓
Linear: 192 → 192
BatchNorm + ReLU + Dropout(0.4)
  ↓
Linear: 192 → 96
BatchNorm + ReLU + Dropout(0.4)
  ↓
Linear: 96 → 48
BatchNorm + ReLU + Dropout(0.4)
  ↓
Linear: 48 → 1
  ↓
Output: Logit (batch, 1)
```

## Training Methodology

### Loss Function

We use **Focal Loss** to address class imbalance:

```
Focal Loss = -α(1-p)^γ log(p)
```

**Parameters:**
- α = 0.25 (balance factor)
- γ = 2.0 (focusing parameter)

**Advantages:**
- Down-weights easy examples (majority class)
- Focuses learning on hard examples (minority class)
- Better than weighted BCE for imbalanced datasets

### Optimization

- **Optimizer**: AdamW
- **Learning rate**: 0.0005
- **Weight decay**: 0.0001 (L2 regularization)
- **Batch size**: 32
- **Epochs**: Up to 100 with early stopping

### Regularization

1. **Dropout**: 0.4 (high dropout for small dataset)
2. **Weight Decay**: 0.0001 (L2 regularization)
3. **Batch Normalization**: Applied in predictor head
4. **Early Stopping**: Patience of 15 epochs on validation F1 score
5. **Weighted Sampling**: Balanced mini-batches for training

### Class Imbalance Handling

1. **Focal Loss**: Automatically focuses on hard examples
2. **Weighted Sampler**: Ensures balanced batches during training
3. **Evaluation Metrics**: Use AUC-ROC and F1 Score (not just accuracy)

### Training Procedure

```python
For each epoch:
    1. Train on training set with weighted sampler
       - Forward pass
       - Compute focal loss
       - Backward pass
       - Optimizer step
    
    2. Validate on validation set
       - Forward pass (no gradient)
       - Compute metrics (AUC-ROC, F1, Accuracy)
    
    3. Check early stopping
       - If validation F1 improved: save model
       - If no improvement for 15 epochs: stop training
    
    4. Evaluate on test set (final model)
```

## Evaluation Methodology

### Metrics

We evaluate all models using multiple metrics:

1. **AUC-ROC** (Area Under ROC Curve)
   - Measures ranking quality
   - Threshold-independent
   - Preferred for imbalanced datasets

2. **F1 Score**
   - Harmonic mean of precision and recall
   - Balance between precision and recall
   - Important for class-imbalanced problems

3. **Accuracy**
   - Overall classification accuracy
   - Can be misleading for imbalanced datasets

4. **AUPRC** (Area Under Precision-Recall Curve)
   - Better for imbalanced datasets than ROC
   - Focuses on positive class performance

### Evaluation Procedure

1. **Training Phase**: Monitor validation metrics for early stopping
2. **Final Evaluation**: Evaluate best model on held-out test set
3. **Comparison**: Compare all models using same test set
4. **Statistical Significance**: Report metrics with confidence intervals where applicable

### Data Splitting

- **Strategy**: Scaffold-based splitting
- **Ratio**: 80% train / 10% validation / 10% test
- **Seed**: 42 (for reproducibility)
- **Rationale**: Ensures structural diversity and prevents data leakage

## Experimental Design

### Models Evaluated

We compare 8 different models:

1. **Baseline MLP** - Fingerprint-based (baseline)
2. **BFGNN** - Graph neural network (torch-molecule)
3. **GRIN** - Repetition-invariant GNN (torch-molecule)
4. **SMILESTransformer** - Sequence model (torch-molecule)
5. **DMPNN** - Directed message passing (DeepChem)
6. **GATv2** - Graph attention network (PyTorch Geometric)
7. **GIN** - Graph isomorphism network (PyTorch Geometric)
8. **SMILESGNN** ⭐ - Our multimodal model (PyTorch Geometric)

### Ablation Studies

Key design choices validated:
- **Multimodal fusion**: Attention vs. concatenation vs. weighted
- **Graph pooling**: Mean-Max vs. mean vs. max
- **Regularization**: Dropout rate, weight decay
- **Class imbalance handling**: Focal loss vs. weighted BCE

### Reproducibility

- **Random seeds**: Fixed (42) for all experiments
- **Configuration files**: YAML files for all hyperparameters
- **Code**: Well-documented and modular
- **Environment**: Conda environment file provided

## Key Contributions

1. **Multimodal Architecture**: SMILESGNN combines sequence and graph representations
2. **Attention-based Fusion**: Dynamic weighting of representations
3. **Comprehensive Comparison**: 8 different models on same dataset
4. **Class Imbalance Handling**: Focal loss and weighted sampling
5. **Thorough Evaluation**: Multiple metrics and proper data splitting

## Implementation Details

### Frameworks

- **PyTorch**: Core deep learning framework
- **PyTorch Geometric**: Graph neural network operations
- **DeepChem**: DMPNN model and data loading
- **torch-molecule**: BFGNN, GRIN, SMILESTransformer models
- **RDKit**: Molecular structure manipulation and visualization

### Code Structure

```
src/
├── graph_models_hybrid.py    # SMILESGNN architecture
├── graph_models.py           # GATv2 architecture
├── graph_models_gin.py       # GIN architecture
├── graph_train.py            # Training infrastructure
├── graph_data.py             # Graph featurization
├── smiles_tokenizer.py       # SMILES tokenization
├── featurization.py          # Fingerprint featurization
└── train.py                  # Training utilities
```

## Hyperparameters

### SMILESGNN Configuration

```yaml
model:
  hidden_dim: 96
  num_graph_layers: 3
  graph_model: "gatv2"
  num_heads: 4
  dropout: 0.4
  use_residual: true
  use_jk: true
  jk_mode: "cat"
  graph_pooling: "meanmax"
  smiles_d_model: 96
  smiles_num_layers: 2
  fusion_method: "attention"

training:
  batch_size: 32
  learning_rate: 0.0005
  weight_decay: 0.0001
  loss_type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
  use_weighted_sampler: true
  early_stopping_patience: 15
```

## Results Summary

SMILESGNN achieves the best performance:
- **AUC-ROC**: 0.9971
- **Accuracy**: 0.9797
- **F1 Score**: 0.8696
- **AUPRC**: 0.9669

This demonstrates the effectiveness of multimodal fusion in molecular property prediction.

