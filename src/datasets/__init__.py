"""
Dataset loaders for molecular toxicity prediction.

Provides a unified interface for loading ClinTox and Tox21 datasets,
along with TaskConfig objects that carry task metadata through the pipeline.

Quick start:
    from src.datasets import load_dataset, get_task_config

    train_df, val_df, test_df = load_dataset("tox21", cache_dir="data")
    task_config = get_task_config("tox21")
    print(task_config.num_tasks)   # 12
    print(task_config.task_names)  # ["NR-AR", "NR-AR-LBD", ...]
"""

from typing import Tuple
import pandas as pd

from src.datasets.base import (
    TaskConfig,
    TOX21_TASKS,
    CLINTOX_TASKS,
    CLINTOX_CONFIG,
    TOX21_CONFIG,
    get_task_config,
)
from src.datasets.clintox import load_clintox
from src.datasets.tox21 import load_tox21


def load_dataset(
    dataset_name: str,
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a toxicity dataset by name.

    Args:
        dataset_name: "clintox" or "tox21".
        cache_dir:    Directory for dataset caching.
        split_type:   "scaffold", "random", or "stratified".
        seed:         Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
        Each DataFrame has a "smiles" column plus one column per task.
    """
    loaders = {"clintox": load_clintox, "tox21": load_tox21}
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {sorted(loaders.keys())}"
        )
    return loaders[dataset_name](
        cache_dir=cache_dir, split_type=split_type, seed=seed
    )


__all__ = [
    "TaskConfig",
    "TOX21_TASKS",
    "CLINTOX_TASKS",
    "CLINTOX_CONFIG",
    "TOX21_CONFIG",
    "get_task_config",
    "load_dataset",
    "load_clintox",
    "load_tox21",
]
