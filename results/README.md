# Model Performance Results

This directory contains consolidated results from all trained models for the ClinTox toxicity prediction task.

## Overall Results Summary

The `overall_results.csv` and `overall_results.md` files contain a comparison table of all models tested on the ClinTox test set.

## Model Performance

### Test Set Metrics (ClinTox)

Models sorted by AUC-ROC (lowest to highest).

| Model | AUC-ROC | Accuracy | F1 Score | AUPRC |
|-------|---------|----------|----------|-------|
| Baseline MLP | 0.7167 | 0.9392 | 0.4706 | 0.4497 |
| GRIN (torch-molecule) | 0.8225 | 0.9459 | 0.4286 | 0.3794 |
| **GIN** (PyTorch Geometric) | 0.8638 | **0.9527** | **0.5882** | 0.5034 |
| GATv2 (PyTorch Geometric) | 0.8848 | 0.8919 | 0.3846 | 0.4664 |
| DMPNN (DeepChem) | 0.8862 | 0.8667 | 0.3333 | 0.5962 |
| BFGNN (torch-molecule) | 0.9188 | 0.9392 | 0.1818 | 0.6164 |
| **SMILESTransformer** (torch-molecule) | **0.9804** | **0.9662** | **0.7826** | 0.6651 |
| **SMILESGNN** (PyTorch Geometric) | **0.9971** | **0.9797** | **0.8696** | **0.9669** |

*Best values in bold*

## Key Observations

1. **SMILESGNN** achieves excellent performance (AUC-ROC: 0.9971, F1: 0.8696) by combining sequence and graph representations, demonstrating the power of multimodal fusion. Results were improved through regularization to prevent overfitting.
2. **SMILESTransformer** achieves the second-best overall performance (AUC-ROC: 0.9804, F1: 0.7826), highlighting the effectiveness of transformer architectures on sequence-based molecular representations
3. **GIN** achieves the **best F1 score (0.5882) among single-modality graph-based models**, demonstrating superior minority class prediction
4. **BFGNN** has high AUC-ROC (0.9188) but very low F1 (0.1818), indicating poor minority class prediction despite good ranking
5. **GATv2** achieves competitive AUC-ROC (0.8848) with moderate F1 (0.3846)
6. Class imbalance (1089 non-toxic vs 95 toxic in training) remains a challenge, but the hybrid model and GIN show the best handling
7. The hybrid model's attention-based fusion of SMILES and graph representations allows it to leverage both global sequence patterns and local structural information

## Model Details

### Baseline MLP
- **Input**: Morgan fingerprints (2048-bit)
- **Architecture**: MLP (2048 → 512 → 256 → 128 → 1)
- **Framework**: Pure PyTorch

### BFGNN (torch-molecule)
- **Input**: SMILES strings (automatically converted to graphs)
- **Architecture**: Graph Neural Network with message passing
- **Framework**: torch-molecule

### GRIN (torch-molecule)
- **Input**: SMILES strings (graph representation)
- **Architecture**: Repetition-Invariant Graph Neural Network
- **Framework**: torch-molecule

### SMILESTransformer (torch-molecule)
- **Input**: SMILES strings (sequence representation)
- **Architecture**: Transformer-based encoder-decoder
- **Framework**: torch-molecule
- **Best Performing Model** ✨

### DMPNN (DeepChem)
- **Input**: GraphData objects (DMPNN featurization)
- **Architecture**: Directed Message Passing Neural Network
- **Framework**: DeepChem

### GATv2 (PyTorch Geometric)
- **Input**: PyTorch Geometric Data objects with rich node/edge features
- **Architecture**: 
  - GATv2 layers (4 layers, 4 heads, 128 hidden dim)
  - Jumping Knowledge connections
  - Set2Set pooling
  - MLP predictor head
- **Framework**: PyTorch Geometric
- **Training**: Focal loss (alpha=0.25, gamma=2.0) with weighted sampling
- **Validation Metrics**: AUC-ROC: 0.9402, F1: 0.6667, PR-AUC: 0.7309

### GIN (PyTorch Geometric)
- **Input**: PyTorch Geometric Data objects with rich node/edge features
- **Architecture**: 
  - GIN layers (4 layers, MLP-based message passing, 128 hidden dim)
  - Learnable epsilon parameter
  - Jumping Knowledge connections
  - Mean-Max pooling (concatenated)
  - MLP predictor head
- **Framework**: PyTorch Geometric
- **Training**: Focal loss (alpha=0.25, gamma=2.0) with weighted sampling
- **Key Feature**: MLP-based message passing (provably as powerful as Weisfeiler-Lehman test)
- **Test Metrics**: AUC-ROC: 0.8638, Accuracy: 0.9527, **F1: 0.5882** (best among single-modality graph models)
- **Validation Metrics**: AUC-ROC: 0.9463, F1: 0.7273, PR-AUC: 0.7645

### SMILESGNN (PyTorch Geometric)
- **Input**: 
  - Graph: PyTorch Geometric Data objects with rich node/edge features
  - SMILES: Tokenized SMILES sequences
- **Architecture**: 
  - **Graph Encoder**: GATv2 layers (4 layers, 4 heads, 128 hidden dim) with Jumping Knowledge and Mean-Max pooling
  - **SMILES Encoder**: Transformer encoder (3 layers, 128 hidden dim) with positional encoding and mean pooling
  - **Fusion Module**: Attention-based fusion that uses SMILES representation as query and graph representation as key/value
  - **Predictor**: MLP head (4 layers with batch norm and dropout)
- **Framework**: PyTorch Geometric + PyTorch Transformer
- **Training**: Focal loss (alpha=0.25, gamma=2.0) with weighted sampling
- **Key Feature**: Multimodal fusion combining sequence (SMILES) and graph representations via attention mechanism
- **Regularization**: Increased dropout (0.4), reduced model complexity (3 graph layers, 2 transformer layers), stronger weight decay (0.0001)
- **Test Metrics**: AUC-ROC: 0.9971, Accuracy: 0.9797, **F1: 0.8696** (best performance among all models)
- **Validation Metrics**: AUC-ROC: 0.9858, F1: 0.6316, PR-AUC: 0.9048

## Files in this Directory

- `overall_results.csv` - Consolidated results in CSV format
- `overall_results.md` - Consolidated results in Markdown table format
- `training_log.txt` - GATv2 model training log

## Individual Model Metrics

Individual model metrics files are located in:
- `models/baseline_metrics.txt`
- `models/torch_molecule_metrics.txt` (BFGNN)
- `models/grin_model_metrics.txt`
- `models/smilestransformer_model_metrics.txt`
- `models/dmpnn_model/dmpnn_model_metrics.txt`
- `models/gatv2_model/gatv2_model_metrics.txt`
- `models/gin_model/gin_model_metrics.txt`
- `models/hybrid_model/hybrid_model_metrics.txt`

