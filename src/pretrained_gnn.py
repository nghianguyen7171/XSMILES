"""
Pre-trained GIN backbone — Hu et al., NeurIPS 2020
"Strategies for Pre-training Graph Neural Networks"

Exact architecture match to load official pretrained weights (gin_supervised_masking.pth
etc.) from SNAP: https://snap.stanford.edu/gnn-pretrain/models/

Pretrained on 2M molecules from ZINC15 + ChEMBL with 4 strategies:
  masking, contextpred, infomax, edgepred

Featurization (different from our usual graph_data.py):
  Node x:         LongTensor (N, 2) — [atom_type_idx, chirality_idx]
  Edge edge_attr: LongTensor (E, 2) — [bond_type_idx, bond_dir_idx]
"""

import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

# ── Constants (must match Hu et al. exactly for weight loading) ────────────────

NUM_ATOM_TYPE     = 120   # x_embedding1 rows
NUM_CHIRALITY_TAG = 3     # x_embedding2 rows (clip CHI_OTHER → 2)
NUM_BOND_TYPE     = 6     # edge_embedding1 rows (4 types + self-loop=4 + mask=5)
NUM_BOND_DIR      = 3     # edge_embedding2 rows

_ATOMIC_NUMS = list(range(1, 119))       # 1..118 → idx 0..117; unknown → 118
_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

_GH_BASE = "https://github.com/snap-stanford/pretrain-gnns/raw/master/chem/model_gin"
PRETRAINED_URLS = {
    "masking":     f"{_GH_BASE}/supervised_masking.pth",
    "contextpred": f"{_GH_BASE}/supervised_contextpred.pth",
    "infomax":     f"{_GH_BASE}/supervised_infomax.pth",
    "edgepred":    f"{_GH_BASE}/supervised_edgepred.pth",
}


# ── Featurization ─────────────────────────────────────────────────────────────

def _safe_idx(lst: list, value) -> int:
    try:
        return lst.index(value)
    except ValueError:
        return len(lst)


def mol_to_graph_hu2020(
    smiles: str,
    label: Optional[np.ndarray] = None,
) -> Optional[Data]:
    """SMILES → PyG Data with Hu et al. (2020) featurization (integer indices)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: [atom_type_idx, chirality_idx]
    x = []
    for atom in mol.GetAtoms():
        atom_idx  = _safe_idx(_ATOMIC_NUMS, atom.GetAtomicNum())       # 0–118
        chir_idx  = min(_safe_idx(_CHIRALITY, atom.GetChiralTag()), 2) # 0–2
        x.append([atom_idx, chir_idx])
    x = torch.tensor(x, dtype=torch.long)  # (N, 2)

    # Edge features: both directions
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)
    else:
        rows, cols, attrs = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            btype = _safe_idx(_BOND_TYPES, bond.GetBondType())   # 0–4
            bdir  = _safe_idx(_BOND_DIRS,  bond.GetBondDir())    # 0–2
            rows += [i, j];  cols += [j, i]
            attrs += [[btype, bdir], [btype, bdir]]
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr  = torch.tensor(attrs,        dtype=torch.long)  # (E, 2)

    if label is not None:
        label_arr = np.array(label, dtype=np.float32)
        y = torch.tensor(label_arr).unsqueeze(0)  # (1, T)
    else:
        y = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


def smiles_list_to_hu_dataset(smiles_list: List[str], labels: np.ndarray) -> List[Data]:
    dataset = []
    for i, smi in enumerate(smiles_list):
        data = mol_to_graph_hu2020(smi, label=labels[i])
        if data is not None:
            dataset.append(data)
    return dataset


# ── GINConv (exact Hu et al. implementation) ──────────────────────────────────

class HuGINConv(MessagePassing):
    """
    GINConv matching Hu et al.:
      • adds self-loops (bond_type=4) before propagation
      • message:  x_j + edge_embedding
      • update:   mlp((1 + eps) * aggr_out)
    """

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps             = nn.Parameter(torch.zeros(1))
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIR,  emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Self-loop attrs: type=4 (self-loop), dir=0 (NONE)
        sl_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=edge_attr.device)
        sl_attr[:, 0] = 4
        edge_attr_sl = torch.cat([edge_attr, sl_attr], dim=0)

        edge_emb = (
            self.edge_embedding1(edge_attr_sl[:, 0])
            + self.edge_embedding2(edge_attr_sl[:, 1])
        )
        return self.propagate(edge_index_sl, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp((1.0 + self.eps) * aggr_out)


# ── GNN Backbone ──────────────────────────────────────────────────────────────

class HuGNNBackbone(nn.Module):
    """
    5-layer GIN with Hu et al. (2020) architecture.
    Matches pretrained checkpoint keys exactly.
    """

    def __init__(
        self,
        emb_dim:    int   = 300,
        num_layers: int   = 5,
        drop_ratio: float = 0.5,
        jk:         str   = "last",
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.jk         = jk

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE,     emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        self.gnns        = nn.ModuleList([HuGINConv(emb_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [h]
        for i, (conv, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = bn(conv(h_list[i], edge_index, edge_attr))
            if i < self.num_layers - 1:
                h = F.dropout(F.relu(h), p=self.drop_ratio, training=self.training)
            else:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.jk == "last":
            return h_list[-1]
        return torch.stack(h_list[1:], dim=0).sum(dim=0)


# ── Predictor ─────────────────────────────────────────────────────────────────

class GNNPretrainedPredictor(nn.Module):
    """HuGNNBackbone + global mean pool + multi-task linear head."""

    def __init__(self, backbone: HuGNNBackbone, num_tasks: int, head_dropout: float = 0.1):
        super().__init__()
        self.backbone     = backbone
        self.head_dropout = head_dropout
        self.head         = nn.Linear(backbone.emb_dim, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.backbone(x, edge_index, edge_attr)      # (N, emb_dim)
        g = global_mean_pool(h, batch)                    # (B, emb_dim)
        g = F.dropout(g, p=self.head_dropout, training=self.training)
        return self.head(g)                               # (B, num_tasks)


# ── Download + Factory ────────────────────────────────────────────────────────

def download_hu_pretrained(
    strategy:  str = "masking",
    cache_dir: str = "data/pretrained_gnns",
) -> Optional[Path]:
    """Download Hu et al. pretrained GIN weights from SNAP (cached locally)."""
    if strategy not in PRETRAINED_URLS:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(PRETRAINED_URLS)}")

    url  = PRETRAINED_URLS[strategy]
    dest = Path(cache_dir) / url.split("/")[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  Cached weights: {dest}")
        return dest

    print(f"  Downloading {strategy} weights from SNAP ...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Saved: {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Falling back to random initialization.")
        return None


def create_pretrained_gin_model(
    num_tasks:       int   = 12,
    strategy:        str   = "masking",
    cache_dir:       str   = "data/pretrained_gnns",
    emb_dim:         int   = 300,
    num_layers:      int   = 5,
    drop_ratio:      float = 0.5,
    jk:              str   = "last",
    head_dropout:    float = 0.1,
    freeze_backbone: bool  = False,
) -> GNNPretrainedPredictor:
    """
    Build GNNPretrainedPredictor and load Hu et al. pretrained backbone weights.

    emb_dim=300 and num_layers=5 are required to match pretrained checkpoint.
    """
    backbone  = HuGNNBackbone(emb_dim=emb_dim, num_layers=num_layers,
                               drop_ratio=drop_ratio, jk=jk)
    predictor = GNNPretrainedPredictor(backbone, num_tasks=num_tasks, head_dropout=head_dropout)

    weights_path = download_hu_pretrained(strategy=strategy, cache_dir=cache_dir)
    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location="cpu")
            missing, unexpected = predictor.backbone.load_state_dict(state, strict=False)
            n_loaded = len(state) - len(unexpected)
            print(f"  Loaded {n_loaded}/{len(state)} pretrained parameter tensors")
            if missing:
                print(f"  Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            if unexpected:
                print(f"  Unexpected: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        except Exception as e:
            print(f"  Could not load weights ({e}) — using random init")

    if freeze_backbone:
        for p in predictor.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen")

    return predictor
