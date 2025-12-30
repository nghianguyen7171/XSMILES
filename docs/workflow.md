# Project Workflow

This document describes the complete workflow of the Clinical Drug Toxicity Prediction project, from data loading and preprocessing through model training, validation, and results analysis.

## Overview

The project workflow consists of several key stages:
1. **Setup & Environment Configuration**
2. **Data Loading & Preprocessing**
3. **Model Training** (Multiple models)
4. **Validation & Evaluation**
5. **Results Consolidation**
6. **Visualization & Analysis**

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Project Workflow Overview                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  1. SETUP       │
│  - Environment  │
│  - Dependencies │
│  - Structure    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. DATA        │
│  - Load ClinTox │
│  - Split        │
│  - Preprocess   │
└────────┬────────┘
         │
         ├─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────┐                            ┌─────────────────┐
│  3a. FEATURE    │                            │  3b. FEATURE    │
│  EXTRACTION     │                            │  EXTRACTION     │
│  (Sequence)     │                            │  (Graph)        │
│                 │                            │                 │
│  - SMILES       │                            │  - RDKit Parse  │
│  Tokenization   │                            │  - Node Feat.   │
│  - Vocabulary   │                            │  - Edge Feat.   │
│  - Encoding     │                            │  - PyG Data     │
└────────┬────────┘                            └────────┬────────┘
         │                                            │
         └────────────────────┬───────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │    4. MODEL TRAINING (8 Models)       │
         │                                        │
         │  ┌──────────┐  ┌──────────┐          │
         │  │ Baseline │  │  BFGNN   │          │
         │  │   MLP    │  │(torch-m) │          │
         │  └──────────┘  └──────────┘          │
         │                                        │
         │  ┌──────────┐  ┌──────────┐          │
         │  │   GRIN   │  │SMILESTrans│         │
         │  │(torch-m) │  │(torch-m)  │         │
         │  └──────────┘  └──────────┘          │
         │                                        │
         │  ┌──────────┐  ┌──────────┐          │
         │  │  DMPNN   │  │  GATv2   │          │
         │  │(DeepChem)│  │  (PyG)   │          │
         │  └──────────┘  └──────────┘          │
         │                                        │
         │  ┌──────────┐  ┌──────────┐          │
         │  │   GIN    │  │SMILESGNN │          │
         │  │  (PyG)   │  │  (PyG) ⭐ │          │
         │  └──────────┘  └──────────┘          │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │    5. VALIDATION & EVALUATION         │
         │                                        │
         │  - Load trained models                │
         │  - Run on test set                    │
         │  - Compute metrics:                   │
         │    • AUC-ROC                          │
         │    • Accuracy                         │
         │    • F1 Score                         │
         │    • AUPRC                            │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │    6. RESULTS CONSOLIDATION            │
         │                                        │
         │  - Gather metrics from all models     │
         │  - Create comparison table            │
         │  - Generate CSV/Markdown reports      │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │    7. VISUALIZATION & ANALYSIS        │
         │                                        │
         │  - ROC/PR curves                      │
         │  - Sample predictions                 │
         │  - Model comparisons                  │
         │  - Error analysis                     │
         └────────────────────────────────────────┘
```

## Stage 1: Setup & Environment Configuration

**Purpose:** Prepare the development environment and verify dependencies

**Files:**
- `env/environment.yml` - Conda environment specification
- `env/requirements.txt` - Pip requirements

**Steps:**
```bash
# Create conda environment
conda env create -f env/environment.yml
conda activate drug-tox-env

# Or install via pip
pip install -r env/requirements.txt
```

**Notebook:** `notebooks/00_setup_and_structure.ipynb`
- Verifies all dependencies
- Creates directory structure
- Checks key imports

**Output:**
- ✓ Environment activated
- ✓ All dependencies installed
- ✓ Project structure created

## Stage 2: Data Loading & Preprocessing

**Purpose:** Load the ClinTox dataset and prepare it for training

**Implementation:** `src/data.py`

**Main Function:**
```python
train_df, val_df, test_df = load_clintox(
    cache_dir="./data",
    split_type="scaffold",  # Scaffold-based splitting
    seed=42
)
```

**Steps:**

1. **Dataset Loading**
   - Downloads ClinTox dataset (if not cached)
   - Source: DeepChem or PyTDC
   - Tasks: `FDA_APPROVED`, `CT_TOX` (we use CT_TOX)

2. **Data Splitting**
   - Split type: **Scaffold-based** (ensures structural diversity)
   - Splits: Train (80%), Validation (10%), Test (10%)
   - Random seed: 42 (for reproducibility)

3. **Data Format**
   - DataFrame with columns: `smiles`, `CT_TOX`
   - `CT_TOX`: Binary label (0 = non-toxic, 1 = toxic)
   - Handles missing values

**Notebook:** `notebooks/01_data_exploration.ipynb`
- Dataset statistics
- Class distribution analysis
- Sample molecule visualizations

**Output:**
- `data/clintox-featurized/` - Cached dataset
- `data/clintox.csv.gz` - Raw dataset
- Train/Val/Test DataFrames

**Statistics:**
- Total samples: ~1,480 molecules
- Training: ~1,184 samples (Toxic: 95, Non-toxic: 1,089)
- Validation: ~148 samples
- Test: ~148 samples

## Stage 3: Feature Extraction

The workflow branches into two parallel feature extraction pipelines, both processing the same SMILES strings:

### 3a. Sequence Feature Extraction (for Transformer models)

**Purpose:** Convert SMILES strings to token sequences

**Implementation:** `src/smiles_tokenizer.py`

**Process:**
```
Raw SMILES → Tokenization → Vocabulary Mapping → Encoding
"CCO"      → ["C","C","O"] → [id_C,id_C,id_O] → Tensor
```

**Steps:**

1. **Tokenization** (`SMILESTokenizer._tokenize_smiles()`)
   - Breaks SMILES into tokens using regex patterns
   - Handles: rings, bonds, branching, aromatic atoms

2. **Vocabulary Building** (`SMILESTokenizer.build_vocab()`)
   - Builds vocabulary from training data
   - Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
   - Vocabulary size: 100 tokens (default)

3. **Encoding** (`SMILESTokenizer.encode()`)
   - Maps tokens to integer IDs
   - Pads/truncates to max_length=128
   - Creates attention masks

**Output:**
- `token_ids`: (batch_size, seq_len) - Integer token IDs
- `attention_mask`: (batch_size, seq_len) - Mask for valid tokens

**Used by:**
- SMILESTransformer (torch-molecule)
- SMILESGNN (sequence branch)

### 3b. Graph Feature Extraction (for GNN models)

**Purpose:** Convert SMILES strings to graph representations

**Implementation:** `src/graph_data.py`

**Process:**
```
Raw SMILES → RDKit Molecule → Feature Extraction → PyG Data Object
"CCO"      → Molecule obj    → Atoms/Bonds feat  → Data(x,edge_index,edge_attr)
```

**Steps:**

1. **SMILES Parsing** (`smiles_to_mol()`)
   - Uses RDKit to parse SMILES
   - Validates and sanitizes structure

2. **Node Feature Extraction** (`get_atom_features()`)
   - Extracts 25 features per atom:
     - Atomic number (one-hot, 10 dims)
     - Formal charge, hybridization, chirality
     - Ring membership, aromaticity
     - Neighbor counts, valence, degree

3. **Edge Feature Extraction** (`get_bond_features()`)
   - Extracts 17 features per bond:
     - Bond type (single, double, triple, aromatic)
     - Bond direction, ring membership
     - Conjugation, stereochemistry

4. **Graph Construction** (`smiles_to_pyg_data()`)
   - Creates PyTorch Geometric Data object
   - Builds edge index (connectivity matrix)
   - Assigns labels

**Output:**
- `x`: (num_nodes, 25) - Node feature matrix
- `edge_index`: (2, num_edges) - Edge connectivity
- `edge_attr`: (num_edges, 17) - Edge feature matrix
- `y`: Scalar label

**Used by:**
- GATv2, GIN (PyTorch Geometric)
- DMPNN (DeepChem)
- SMILESGNN (graph branch)

## Stage 4: Model Training

**Purpose:** Train 8 different models for comparison

The project trains multiple models in parallel (or sequentially):

### 4.1 Baseline MLP Model

**Script:** `notebooks/02_training_baseline.ipynb`

**Input:** Morgan fingerprints (2048-bit)
- Generated via `src/featurization.py` → `featurize_fingerprint()`

**Architecture:**
- MLP: 2048 → 512 → 256 → 128 → 1
- Framework: Pure PyTorch

**Training:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Early stopping on validation loss

**Output:**
- `models/baseline_mlp_model.pt`
- `models/baseline_metrics.txt`

### 4.2 torch-molecule Models (BFGNN, GRIN, SMILESTransformer)

**Scripts:**
- `notebooks/03_training_gnn.ipynb` (BFGNN)
- `notebooks/03_training_grin.ipynb` (GRIN)
- `notebooks/03_training_smilestransformer.ipynb` (SMILESTransformer)

**Input:** SMILES strings (handled by torch-molecule library)

**Framework:** torch-molecule (sklearn-style API)

**Training:**
- Uses `autofit()` for automated hyperparameter optimization
- sklearn-style: `model.fit(X_train, y_train)`

**Output:**
- `models/torch_molecule_model.pkl` (BFGNN)
- `models/grin_model.pkl`
- `models/smilestransformer_model.pkl`

### 4.3 DeepChem Models (DMPNN)

**Script:** `notebooks/03_training_deepchem_gcn.ipynb`

**Input:** GraphData objects (DMPNN featurization)

**Framework:** DeepChem

**Training:**
- Custom training loop
- DeepChem Dataset format

**Output:**
- `models/dmpnn_model/dmpnn_model_state_dict.pt`

### 4.4 PyTorch Geometric Models (GATv2, GIN, SMILESGNN)

**Scripts:**
- `scripts/train_gatv2.py`
- `scripts/train_gin.py`
- `scripts/train_hybrid.py` (SMILESGNN)

**Input:** PyTorch Geometric Data objects

**Training Configuration:**
- Config files: `config/gatv2_config.yaml`, `config/gin_config.yaml`, `config/smilesgnn_config.yaml`

**Training Process:**
```python
# Example: SMILESGNN training
python scripts/train_hybrid.py --config config/smilesgnn_config.yaml
```

**Training Details:**
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Optimizer: AdamW
- Learning rate: 0.0005
- Batch size: 32
- Weighted sampler for class imbalance
- Early stopping on F1 score

**Training Loop:**
1. Load dataset and create DataLoaders
2. Initialize model
3. For each epoch:
   - Train on training set
   - Validate on validation set
   - Compute metrics (AUC-ROC, F1, Accuracy)
   - Save best model (based on validation metric)
4. Early stopping if no improvement

**Output:**
- `models/gatv2_model/best_model.pt`
- `models/gin_model/best_model.pt`
- `models/smilesgnn_model/best_model.pt`
- Training logs and metrics files

## Stage 5: Validation & Evaluation

**Purpose:** Evaluate all trained models on the test set

**Implementation:** Model-specific evaluation functions

### Evaluation Metrics

All models are evaluated on the test set using:

- **AUC-ROC**: Area Under the ROC Curve (ranking quality)
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUPRC**: Area Under the Precision-Recall Curve (imbalanced data)

### Evaluation Process

For each model:

1. **Load trained model**
   ```python
   model = load_model(model_path)
   model.eval()
   ```

2. **Prepare test data**
   - Load test set
   - Apply same preprocessing as training

3. **Generate predictions**
   ```python
   predictions = model(test_data)
   probabilities = sigmoid(predictions)
   ```

4. **Compute metrics**
   ```python
   from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
   auc_roc = roc_auc_score(y_true, y_proba)
   accuracy = accuracy_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   ```

5. **Save metrics**
   - Text file: `models/*/model_metrics.txt`
   - Includes all metrics and confusion matrix

**Evaluation happens during training** (validation set) and **after training** (test set).

## Stage 6: Results Consolidation

**Purpose:** Gather metrics from all models into a unified comparison table

**Script:** `scripts/consolidate_results.py`

**Process:**
```bash
python scripts/consolidate_results.py
```

**Steps:**

1. **Load metrics from all models**
   - Reads metric files from `models/` directory
   - Parses text files or CSV files

2. **Extract key metrics**
   - AUC-ROC, Accuracy, F1 Score, AUPRC
   - Standardizes format across models

3. **Create comparison table**
   - Sorts models by performance
   - Formats as CSV and Markdown

4. **Save consolidated results**
   - `results/overall_results.csv`
   - `results/overall_results.md`

**Output Format:**
```
| Model | AUC-ROC | Accuracy | F1 Score | AUPRC |
|-------|---------|----------|----------|-------|
| Baseline MLP | 0.7167 | 0.9392 | 0.4706 | 0.4497 |
| ... | ... | ... | ... | ... |
| SMILESGNN ⭐ | 0.9971 | 0.9797 | 0.8696 | 0.9669 |
```

## Stage 7: Visualization & Analysis

**Purpose:** Generate visualizations and perform detailed analysis

### 7.1 ROC and PR Curves

**Script:** `scripts/generate_curves.py`

**Process:**
```bash
python scripts/generate_curves.py
```

**Steps:**
1. Load all trained models
2. Generate predictions on test set
3. Compute ROC and PR curves for each model
4. Plot all curves on same figure
5. Save to `results/roc_curves_all_models.png` and `results/pr_curves_all_models.png`

**Output:**
- ROC curves comparison
- PR curves comparison

### 7.2 Sample Prediction Visualizations

**Script:** `scripts/generate_sample_visualizations.py`

**Process:**
```bash
python scripts/generate_sample_visualizations.py
```

**Steps:**
1. Select representative samples from test set
2. Generate predictions for all models
3. Visualize molecules with RDKit
4. Show predictions and ground truth labels
5. Categorize: correct toxic, correct non-toxic, disagreements, diverse samples

**Output:**
- `results/sample_representative_models_correct_toxic.png`
- `results/sample_representative_models_correct_nontoxic.png`
- `results/sample_representative_models_disagreement.png`
- `results/sample_representative_models_diverse.png`

### 7.3 Top 2 Models Comparison

**Script:** `scripts/generate_top2_comparison.py`

**Process:**
```bash
python scripts/generate_top2_comparison.py
```

**Purpose:** Compare SMILESTransformer and SMILESGNN (best models)

**Output:**
- `results/top2_comparison_both_correct_toxic.png`
- `results/top2_comparison_both_correct_nontoxic.png`
- `results/top2_comparison_smilesgnn_wins.png`
- `results/top2_comparison_smilestransformer_wins.png`
- `results/top2_comparison_diverse.png`

### 7.4 SMILESGNN Architecture Visualization

**Script:** `scripts/generate_smiles_graph_pairs.py`

**Process:**
```bash
python scripts/generate_smiles_graph_pairs.py
```

**Purpose:** Show how SMILESGNN processes molecules in both representations

**Output:**
- `results/smilesgnn_smiles_graph_pairs.png` - Paired SMILES sequence and graph visualizations

### 7.5 SMILESGNN Molecular Graph Visualization

**Script:** `scripts/generate_smilesgnn_graph_visualization.py`

**Process:**
```bash
python scripts/generate_smilesgnn_graph_visualization.py
```

**Output:**
- `results/smilesgnn_molecular_graphs.png` - Graph structure visualizations

## Complete Workflow Summary

### Quick Start Commands

```bash
# 1. Setup environment
conda env create -f env/environment.yml
conda activate drug-tox-env

# 2. Train all models (run in sequence or parallel)
# Baseline MLP (via notebook)
jupyter notebook notebooks/02_training_baseline.ipynb

# PyTorch Geometric models
python scripts/train_gatv2.py --config config/gatv2_config.yaml
python scripts/train_gin.py --config config/gin_config.yaml
python scripts/train_hybrid.py --config config/smilesgnn_config.yaml

# 3. Consolidate results
python scripts/consolidate_results.py

# 4. Generate visualizations
python scripts/generate_curves.py
python scripts/generate_sample_visualizations.py
python scripts/generate_top2_comparison.py
python scripts/generate_smiles_graph_pairs.py
```

### File Dependencies

```
data/
  └── clintox-featurized/     (Stage 2: Data loading)

models/
  ├── baseline_mlp_model.pt          (Stage 4: Training)
  ├── torch_molecule_model.pkl       (Stage 4: Training)
  ├── grin_model.pkl                 (Stage 4: Training)
  ├── smilestransformer_model.pkl    (Stage 4: Training)
  ├── dmpnn_model/                   (Stage 4: Training)
  ├── gatv2_model/                   (Stage 4: Training)
  ├── gin_model/                     (Stage 4: Training)
  └── smilesgnn_model/               (Stage 4: Training)

results/
  ├── overall_results.csv            (Stage 6: Consolidation)
  ├── overall_results.md             (Stage 6: Consolidation)
  ├── roc_curves_all_models.png      (Stage 7: Visualization)
  ├── pr_curves_all_models.png       (Stage 7: Visualization)
  └── sample_*.png                   (Stage 7: Visualization)
```

### Execution Order

1. **Sequential (Required):**
   - Setup (1)
   - Data Loading (2)
   - Feature Extraction (3a/3b) - happens during training

2. **Parallel (Can run simultaneously):**
   - Model Training (4) - All 8 models can train in parallel
   
3. **Sequential (After training):**
   - Evaluation (5) - Happens during/after training
   - Results Consolidation (6)
   - Visualization (7)

### Key Design Decisions

1. **Scaffold-based splitting**: Ensures structural diversity between splits
2. **Multiple model types**: Allows comprehensive comparison
3. **Dual feature extraction**: Sequence and graph for SMILESGNN
4. **Focal Loss**: Handles class imbalance (1089 non-toxic vs 95 toxic)
5. **Early stopping**: Prevents overfitting
6. **Comprehensive evaluation**: Multiple metrics for thorough assessment

## Workflow Timeline

**Estimated Time (on CPU):**
- Setup: ~5 minutes
- Data Loading: ~2-5 minutes (depends on download)
- Model Training: ~2-4 hours total (varies by model)
  - Baseline MLP: ~5-15 minutes
  - torch-molecule models: ~10-30 minutes each
  - PyG models: ~30-60 minutes each
- Results Consolidation: ~1 minute
- Visualization Generation: ~5-10 minutes

**Total: ~3-5 hours** (can be parallelized for faster completion)

## Troubleshooting

### Common Issues

1. **Missing models**: Run training scripts first
2. **Data not found**: Run data loading notebook first
3. **Import errors**: Activate conda environment
4. **CUDA errors**: Set `--device cpu` in training scripts
5. **Out of memory**: Reduce batch size in config files

### Verification Checklist

- [ ] Environment activated: `conda activate drug-tox-env`
- [ ] Data cached in `data/clintox-featurized/`
- [ ] All models trained and saved in `models/`
- [ ] Metrics files exist in `models/*/model_metrics.txt`
- [ ] Consolidated results in `results/overall_results.csv`
- [ ] Visualizations generated in `results/`

