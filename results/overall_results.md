# Overall Model Performance Comparison

Results on ClinTox test set. Models sorted by AUC-ROC (lowest to highest).

| Model | AUC-ROC | Accuracy | F1 Score | AUPRC |
|-------|---------|----------|----------|-------|
| Baseline MLP | 0.7167 | 0.9392 | 0.4706 | 0.4497 |
| GRIN (torch-molecule) | 0.8225 | 0.9459 | 0.4286 | 0.3794 |
| GIN (PyTorch Geometric) | 0.8638 | 0.9527 | 0.5882 | 0.5034 |
| GATv2 (PyTorch Geometric) | 0.8848 | 0.8919 | 0.3846 | 0.4664 |
| DMPNN (DeepChem) | 0.8862 | 0.8667 | 0.3333 | 0.5962 |
| BFGNN (torch-molecule) | 0.9188 | 0.9392 | 0.1818 | 0.6164 |
| SMILESTransformer (torch-molecule) | 0.9804 | 0.9662 | 0.7826 | 0.6651 |
| SMILESGNN (PyTorch Geometric) | 0.9971 | 0.9797 | 0.8696 | 0.9669 |

