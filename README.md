# Clinical Drug Toxicity Prediction with torch-molecule

A complete research prototype for predicting clinical drug toxicity from molecular structures (SMILES strings) using deep learning, with visual explanations highlighting important molecular substructures.

![Project Overview](output/figures/05_model_performance_comparison.png)

## Overview

This project implements multiple deep learning approaches for clinical drug toxicity prediction:

- **Baseline MLP Model** - Multi-layer perceptron on Morgan fingerprints
- **BFGNN (torch-molecule)** - Graph Neural Network for molecular property prediction
- **GRIN (torch-molecule)** - Repetition-Invariant Graph Neural Network
- **SMILESTransformer (torch-molecule)** - Transformer-based model on SMILES sequences

The project includes comprehensive explainability methods, error analysis, and embedding visualizations.

**Main Library:** [torch-molecule](https://github.com/liugangcode/torch-molecule) - A molecular AI library with sklearn-style interface

**Primary Dataset:** ClinTox (clinical toxicity dataset) - Comparing FDA-approved drugs vs. drugs that failed clinical trials due to toxicity

## Project Structure

```
Torch_molecule/
├── env/
│   ├── environment.yml          # Conda environment specification
│   ├── requirements.txt         # Pip requirements (reference)
│   └── install-pip3.sh         # Automated installation script
├── notebooks/
│   ├── 00_setup_and_structure.ipynb        # Project setup
│   ├── 01_data_exploration.ipynb           # Dataset exploration & visualization
│   ├── 02_training_baseline.ipynb          # Train baseline MLP model
│   ├── 03_training_gnn.ipynb               # Train BFGNN model
│   ├── 03_training_grin.ipynb              # Train GRIN model
│   ├── 03_training_smilestransformer.ipynb # Train SMILESTransformer model
│   ├── 04_explainability_and_visualization.ipynb  # Model explanations
│   ├── 05_results_and_error_analysis.ipynb        # Comprehensive analysis
│   └── 06_representation.ipynb                     # Feature representation visualization
├── src/
│   ├── __init__.py
│   ├── data.py                  # Dataset loading (ClinTox, Tox21)
│   ├── featurization.py         # SMILES → fingerprints/graphs
│   ├── models.py                # Baseline MLP + torch-molecule models
│   ├── train.py                 # Training loops
│   ├── explain.py               # Attribution methods
│   ├── viz.py                   # RDKit visualization
│   ├── utils.py                 # Utilities (seed, config, metrics)
│   └── pipelines.py             # High-level training/evaluation pipelines
├── data/                        # Dataset cache (created at runtime)
├── models/                      # Saved models (created at runtime)
└── output/
    └── figures/                 # Generated visualizations
```

## Models and Results

### Model Performance Comparison

| Model | AUC-ROC | Accuracy | F1 Score | PR-AUC |
|-------|---------|----------|----------|--------|
| **Baseline MLP** | 0.7167 | 0.9392 | 0.4706 | 0.4497 |
| **BFGNN** | 0.9188 | 0.9392 | 0.1818 | 0.6164 |
| **GRIN** | 0.8225 | 0.9459 | 0.4286 | 0.3794 |
| **SMILESTransformer** | **0.9804** | **0.9662** | **0.7826** | **0.6651** |

*Note: Results are on the test set. SMILESTransformer achieves the best overall performance.*

![Model Performance Comparison](output/figures/05_model_performance_comparison.png)

### Model Architectures

#### 1. Baseline MLP (FingerprintMLP)
- **Input:** Morgan fingerprints (2048-bit ECFP-like) [[1]](#references)
- **Architecture:** 
  - Input Layer: 2048 → 512
  - Hidden Layer 1: 512 → 256
  - Hidden Layer 2: 256 → 128
  - Output Layer: 128 → 1
- **Framework:** Pure PyTorch
- **Training:** Custom training loop with early stopping
- **Reference:** Extended-Connectivity Fingerprints (ECFP) as introduced by Rogers & Hahn (2010)

#### 2. BFGNN (torch-molecule)
- **Input:** SMILES strings (automatically converted to graphs)
- **Architecture:** Graph Neural Network with message passing [[2, 3]](#references)
- **Framework:** torch-molecule (sklearn-style API)
- **Training:** Automated hyperparameter optimization via `autofit()`
- **Reference:** Based on Graph Convolutional Networks and Message Passing Neural Networks for molecular property prediction

#### 3. GRIN (torch-molecule)
- **Input:** SMILES strings (graph representation)
- **Architecture:** Repetition-Invariant Graph Neural Network [[2, 3]](#references)
- **Framework:** torch-molecule
- **Training:** Automated hyperparameter optimization
- **Reference:** Graph Neural Network architecture with repetition-invariant properties for molecular representation

#### 4. SMILESTransformer (torch-molecule)
- **Input:** SMILES strings (sequence representation)
- **Architecture:** Transformer-based encoder-decoder [[4]](#references)
- **Framework:** torch-molecule
- **Training:** Automated hyperparameter optimization
- **Best Performing Model** ✨
- **Reference:** Transformer architecture (Vaswani et al., 2017) adapted for SMILES sequence processing

### ROC and Precision-Recall Curves

![ROC and PR Curves](output/figures/05_roc_pr_curves.png)

### Model References

Each model used in this project is based on established research in molecular property prediction:

1. **Morgan Fingerprints (ECFP)**: Extended-Connectivity Fingerprints (ECFP) are circular topological fingerprints that capture local molecular substructures. Used in the Baseline MLP model.

2. **Graph Neural Networks**: Graph-based models (BFGNN, GRIN) process molecules as graphs, using message passing to aggregate information from neighboring atoms.

3. **Transformer Architecture**: The SMILESTransformer model leverages the Transformer architecture originally designed for sequence-to-sequence tasks, adapted for molecular SMILES sequences.

4. **ClinTox Dataset**: Part of the MoleculeNet benchmark [[5]](#references) for molecular machine learning.

### Additional References

- **torch-molecule Library**: Available at [https://github.com/liugangcode/torch-molecule](https://github.com/liugangcode/torch-molecule). Provides sklearn-style interfaces for various molecular AI models including BFGNN, GRIN, and SMILESTransformer.

- **Graph Neural Networks for Molecules**: The graph-based models (BFGNN, GRIN) draw inspiration from various GNN architectures designed for molecular property prediction, including Graph Convolutional Networks [[2]](#references) and Message Passing Neural Networks [[3]](#references).

## Dataset

### ClinTox Dataset

The ClinTox dataset compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity reasons. This is a binary classification task.

- **Total Size:** 1,480 compounds
- **Train Set:** 1,184 compounds
- **Validation Set:** 148 compounds
- **Test Set:** 148 compounds
- **Split Method:** Scaffold-based splitting (ensures molecular scaffolds don't overlap between splits)
- **Class Distribution:** Imbalanced dataset (majority class: non-toxic)

![Class Distribution](output/figures/01_class_distribution.png)

![Molecular Property Distributions](output/figures/01_molecular_property_distributions.png)

![Atom and Bond Distribution](output/figures/01_atom_bond_distribution.png)

## Setup

### Prerequisites

- Python 3.10+
- Conda or pip3 (macOS)
- Jupyter Notebook

### Local Setup (macOS)

#### Option 1: Using Conda (Recommended)

1. **Create the environment:**
   ```bash
   conda env create -f env/environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate drug-tox-env
   ```

3. **Install Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name drug-tox-env --display-name "Python (drug-tox-env)"
   ```

#### Option 2: Using pip3 (macOS)

**IMPORTANT:** `torch-scatter` requires `torch` to be installed first!

1. **Use the automated installation script (recommended):**
   ```bash
   ./env/install-pip3.sh
   ```

2. **Or install manually in steps:**
   ```bash
   pip3 install -r env/requirements-base.txt
   pip3 install -r env/requirements-torch-extensions.txt
   pip3 install -r env/requirements-optional.txt
   ```

3. **Install Jupyter:**
   ```bash
   pip3 install jupyter ipykernel
   python3 -m ipykernel install --user --name drug-tox-env --display-name "Python (drug-tox-env)"
   ```

**Note:** On macOS, always use `pip3` instead of `pip` for Python 3 packages.

### Google Colab Setup

In Colab, install dependencies directly:
```python
!pip install torch-molecule torch-geometric torch-scatter deepchem transformers rdkit-pypi networkx
```

## Usage

### Running the Experiment

Execute notebooks in the following sequence:

1. **00_setup_and_structure.ipynb**
   - Verify project structure
   - Test imports and dependencies
   - Create necessary directories

2. **01_data_exploration.ipynb**
   - Load ClinTox dataset
   - Analyze dataset statistics
   - Visualize molecular structures
   - Generate data exploration figures

3. **02_training_baseline.ipynb**
   - Train baseline MLP model
   - Evaluate on validation and test sets
   - Save model and metrics

4. **03_training_gnn.ipynb** / **03_training_grin.ipynb** / **03_training_smilestransformer.ipynb**
   - Train torch-molecule models (BFGNN, GRIN, or SMILESTransformer)
   - Automated hyperparameter optimization
   - Model comparison and evaluation
   - Save trained models

5. **04_explainability_and_visualization.ipynb**
   - Generate model explanations for all models
   - Visualize atom-level importance
   - Compare explanations across models
   - Analyze correct and incorrect predictions

6. **05_results_and_error_analysis.ipynb**
   - Comprehensive performance comparison
   - Confusion matrices for all models
   - Error overlap analysis
   - ROC and PR curves
   - Prediction probability distributions

7. **06_representation.ipynb**
   - Extract and visualize feature representations
   - t-SNE and UMAP embeddings for all models
   - Graph structure visualizations
   - Compare representations across model types

### Quick Start Example

```python
from src.data import load_clintox
from src.featurization import featurize_batch
from src.models import create_baseline_model
from src.train import train_baseline_model, evaluate_model
from src.explain import compute_gradient_attribution
from src.viz import plot_explained_molecule

# Load data
train_df, val_df, test_df = load_clintox()

# Featurize (for baseline model)
train_fps = featurize_batch(train_df['smiles'].tolist(), mode="fingerprint")

# Create and train model
model = create_baseline_model()
# ... training code ...

# Generate explanations
attributions = compute_gradient_attribution(model, smiles, input_tensor)
plot_explained_molecule(smiles, attributions)
```

## Visualizations

### Data Exploration

The project includes comprehensive visualizations of the dataset:

- **Class Distribution:** Shows the imbalance in toxic vs. non-toxic compounds
- **Molecular Properties:** Distributions of molecular weights, logP, etc.
- **Atom/Bond Statistics:** Analysis of molecular complexity

### Model Explanations

Atom-level importance visualizations for all models:

![MLP Explanations](output/figures/05_samples_mlp_correct_grin_wrong.png)

### Graph Structure Visualization

Graph-based models (BFGNN, GRIN) process molecules as graphs where atoms are nodes and bonds are edges:

![GRIN Graph Structures](output/figures/06_grin_graph_structures.png)

![BFGNN Graph Structures](output/figures/06_bfgnn_graph_structures.png)

### Feature Representations

t-SNE and UMAP visualizations of model embeddings:

![Model Representations Comparison](output/figures/06_all_models_representation_comparison.png)

![MLP Embeddings](output/figures/06_mlp_embeddings_umap.png)

![SMILESTransformer Embeddings](output/figures/06_smilestransformer_embeddings_umap.png)

### Error Analysis

Confusion matrices and error overlap analysis:

![Confusion Matrices](output/figures/05_confusion_matrices.png)

![Error Overlap](output/figures/05_error_overlap_venn.png)

### Prediction Samples

Examples of correct and incorrect predictions:

![Sample Predictions](output/figures/05_samples_both_correct_nontoxic.png)

## Key Features

### Data Loading
- ClinTox dataset with scaffold-based train/val/test splits
- Automatic handling of missing values
- DeepChem and PyTDC fallback support
- Dataset caching for faster subsequent loads

### Featurization
- **Fingerprint mode:** Morgan fingerprints (ECFP-like, 2048 bits) for baseline models
- **Graph mode:** SMILES strings automatically converted to graph representations
- **Sequence mode:** SMILES strings as token sequences for transformer models

### Models
- **Baseline:** Custom PyTorch MLP on fingerprint features
- **torch-molecule:** BFGNN, GRIN, SMILESTransformer with sklearn-style API
- Automated hyperparameter optimization via Optuna

### Explainability
- **Gradient-based attribution:** For fingerprint-based models (MLP)
- **Perturbation-based attribution:** For graph/sequence-based models (GNNs, SMILESTransformer)
- Atom-level importance mapping
- RDKit-based molecular visualization with importance coloring

### Visualization
- 2D molecule drawings with atom-level importance heatmaps
- NetworkX-based graph structure visualizations
- Grid visualizations for multiple molecules
- Embedding visualizations (t-SNE/UMAP)
- Performance comparison charts
- ROC and Precision-Recall curves

## Configuration

Default configuration is provided in `src/utils.py`. Key parameters:

### Fingerprint
- `radius`: 2
- `n_bits`: 2048

### Baseline Model
- `hidden_dims`: [512, 256, 128]
- `dropout`: 0.2

### Training
- `num_epochs`: 50
- `learning_rate`: 0.001
- `batch_size`: 128
- `device`: "cpu" (or "cuda" if available)

### torch-molecule
- `model_type`: "BFGNN", "GRIN", or "SMILESTransformer"
- `n_trials`: 20 (number of hyperparameter search trials)

## Dependencies

See `env/environment.yml` for full list. Key dependencies:

- **Python**: 3.10+
- **PyTorch**: For deep learning models
- **RDKit**: Molecular informatics and visualization
- **torch-molecule**: Molecular AI library
- **torch-geometric, torch-scatter**: Graph neural network operations
- **DeepChem**: Dataset loading (alternative: PyTDC)
- **NumPy, Pandas, Scikit-learn**: Data processing and metrics
- **Matplotlib, Seaborn**: Visualization
- **NetworkX**: Graph structure visualization
- **Jupyter**: Interactive notebooks
- **Optuna**: Hyperparameter optimization

## Results Summary

### Best Model: SMILESTransformer

The SMILESTransformer model achieves the best performance across all metrics:
- **AUC-ROC: 0.9804** (excellent discriminative ability)
- **Accuracy: 0.9662** (high overall accuracy)
- **F1 Score: 0.7826** (best balance for imbalanced dataset)
- **PR-AUC: 0.6651** (strong precision-recall performance)

### Key Findings

1. **Sequence-based models (SMILESTransformer)** outperform graph-based models on this task
2. **Graph-based models (BFGNN, GRIN)** show strong performance and provide interpretable graph structures
3. **Baseline MLP** provides a simple baseline but is outperformed by more sophisticated architectures
4. **Class imbalance** affects F1 scores, but PR-AUC provides better insight for this dataset

### Model Agreement

Error overlap analysis shows:
- Models often agree on correctly classified examples
- Different architectures make different types of errors
- Ensemble approaches could improve robustness

## Troubleshooting

### Common Issues

1. **torch-scatter installation fails (macOS)**
   - Solution: Install PyTorch first, then torch-scatter
   - Use the automated script: `./env/install-pip3.sh`

2. **RDKit import errors**
   - Solution: Install via conda (recommended) or `rdkit-pypi` via pip

3. **NetworkX not available**
   - Solution: `pip install networkx` for graph visualizations

4. **Dataset download fails**
   - Solution: Check internet connection and DeepChem/PyTDC installation
   - Dataset will cache in `data/` directory after first download

5. **Memory errors during training**
   - Solution: Reduce batch size in configuration
   - Use smaller fingerprint size for baseline model

## Citation

If you use this code in your research, please cite:

```bibtex
@software{clinical_toxicity_prediction,
  title={Clinical Drug Toxicity Prediction with torch-molecule},
  author={Nguyen, Nghia},
  year={2025},
  url={https://github.com/nghianguyen7171/molecule}
}
```

## References

### Model Papers

[1] **Rogers, D., & Hahn, M.** (2010). Extended-Connectivity Fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754. [DOI: 10.1021/ci100050t](https://pubs.acs.org/doi/10.1021/ci100050t)

[2] **Kipf, T. N., & Welling, M.** (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

[3] **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.** (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)

[4] **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### Dataset Papers

[5] **Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V.** (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*, 9, 513-530. [DOI: 10.1039/C7SC02664A](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)

**ClinTox Dataset**: The ClinTox dataset is part of the MoleculeNet benchmark, comparing drugs approved by the FDA with drugs that failed clinical trials due to toxicity.

### Libraries and Tools

- **torch-molecule**: [https://github.com/liugangcode/torch-molecule](https://github.com/liugangcode/torch-molecule) - A molecular AI library with sklearn-style interface
- **RDKit**: [https://www.rdkit.org/](https://www.rdkit.org/) - Open-source cheminformatics toolkit
- **DeepChem**: [https://deepchem.io/](https://deepchem.io/) - Deep learning library for drug discovery and quantum chemistry
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/) - Deep learning framework
- **NetworkX**: [https://networkx.org/](https://networkx.org/) - Network analysis library for graph visualizations

## License

This is a research prototype project.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated:** January 2025
