"""
Tox21 dataset loader.

Tox21 contains ~7,831 compounds tested against 12 nuclear receptor and
stress-response assays. Labels are sparse — many (compound, assay) pairs
are untested and encoded as NaN.

Assay groups:
  NR-* : Nuclear receptor disruption (7 tasks)
  SR-* : Stress response pathway activation (5 tasks)

Dataset size: ~7,831 compounds
Class balance: ~15-30 : 1 per task (highly imbalanced, varies per assay)
Reference: Huang et al., Tox21 Challenge, 2016.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.datasets.base import TOX21_TASKS


def load_tox21(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Tox21 multi-task toxicity dataset.

    Args:
        cache_dir:  Directory to cache downloaded datasets.
        split_type: "scaffold" (recommended), "random", or "stratified".
        seed:       Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df). Each DataFrame has columns:
        - "smiles":      SMILES string
        - One column per task (see TOX21_TASKS), binary {0, 1} or NaN.

    Missing labels are encoded as NaN (not -1). The MaskedMultiTaskLoss
    in src/graph_train.py handles NaN entries automatically.
    """
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        import deepchem as dc
        from deepchem.molnet import load_tox21 as _dc_load

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
            data = {"smiles": ds.ids}
            for i, task in enumerate(tasks):
                col = ds.y[:, i].copy().astype(float)
                # DeepChem marks missing labels with w == 0 (sample weight).
                # Set those entries to NaN so MaskedMultiTaskLoss can ignore them.
                col[ds.w[:, i] == 0] = np.nan
                data[task] = col
            df = pd.DataFrame(data)
            # Rename to canonical task names if DeepChem names differ
            df = _align_task_columns(df, tasks)
            return df

        return _to_df(train_ds), _to_df(val_ds), _to_df(test_ds)

    except ImportError:
        pass

    # Fallback: PyTDC (single-task only — provide a graceful error)
    raise ImportError(
        "Tox21 multi-task loader requires DeepChem.\n"
        "  conda activate drug-tox-env  (already installed)\n"
        "  or: pip install deepchem"
    )


def _align_task_columns(df: pd.DataFrame, dc_task_names: List[str]) -> pd.DataFrame:
    """
    Rename DeepChem task columns to canonical TOX21_TASKS names.

    DeepChem sometimes returns task names with slightly different casing or
    punctuation. This function maps them to the canonical list.
    """
    canonical = {t.lower().replace("-", "").replace("_", ""): t for t in TOX21_TASKS}
    rename_map = {}
    for dc_name in dc_task_names:
        key = dc_name.lower().replace("-", "").replace("_", "")
        if dc_name not in TOX21_TASKS and key in canonical:
            rename_map[dc_name] = canonical[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _get_splitter(dc, split_type: str):
    mapping = {
        "scaffold":   dc.splits.ScaffoldSplitter,
        "random":     dc.splits.RandomSplitter,
        "stratified": dc.splits.RandomStratifiedSplitter,
    }
    cls = mapping.get(split_type, dc.splits.RandomSplitter)
    return cls()
