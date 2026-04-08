"""
ClinTox dataset loader.

ClinTox contains FDA-approved drugs and compounds that failed clinical trials
due to toxicity. The CT_TOX label = 1 for clinical trial failures, 0 for
FDA-approved drugs.

Dataset size: ~1,478 compounds
Class balance: ~11.5 : 1 (non-toxic : toxic)
Reference: Wu et al., MoleculeNet, 2018.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def load_clintox(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ClinTox dataset with train/val/test splits.

    Args:
        cache_dir:  Directory to cache downloaded datasets.
        split_type: "scaffold" (recommended), "random", or "stratified".
        seed:       Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df). Each DataFrame has columns:
        - "smiles":  SMILES string
        - "CT_TOX":  Binary toxicity label (1 = toxic, 0 = non-toxic)
    """
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        import deepchem as dc
        from deepchem.molnet import load_clintox as _dc_load

        splitter = _get_splitter(dc, split_type)
        tasks, datasets, _ = _dc_load(
            data_dir=str(cache_path),
            save_dir=str(cache_path),
            featurizer=dc.feat.RawFeaturizer(),
            splitter=splitter,
            seed=seed,
        )
        train_ds, val_ds, test_ds = datasets

        def _to_df(ds):
            smiles = ds.ids
            # ClinTox has two tasks: [FDA_APPROVED, CT_TOX]; take CT_TOX (col 1)
            y = ds.y
            if y.ndim == 2 and y.shape[1] == 2:
                labels = y[:, 1]
            elif y.ndim == 2:
                labels = y[:, 0]
            else:
                labels = y.flatten()
            n = min(len(smiles), len(labels))
            return pd.DataFrame({"smiles": smiles[:n], "CT_TOX": labels[:n]})

        return _to_df(train_ds), _to_df(val_ds), _to_df(test_ds)

    except ImportError:
        pass

    # Fallback: PyTDC
    try:
        from tdc.single_pred import Tox

        data = Tox(name="ClinTox", path=str(cache_path))
        df = data.get_data().rename(columns={"Drug": "smiles", "Y": "CT_TOX"})
        return _random_split(df, seed)

    except ImportError:
        raise ImportError(
            "ClinTox loader requires DeepChem or PyTDC.\n"
            "  pip install deepchem   or   pip install pytdc"
        )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _get_splitter(dc, split_type: str):
    mapping = {
        "scaffold":   dc.splits.ScaffoldSplitter,
        "random":     dc.splits.RandomSplitter,
        "stratified": dc.splits.RandomStratifiedSplitter,
    }
    cls = mapping.get(split_type, dc.splits.RandomSplitter)
    return cls()


def _random_split(df: pd.DataFrame, seed: int):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )
