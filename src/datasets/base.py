"""
Task configuration for multi-dataset support.

Defines TaskConfig, a dataclass that carries all task-specific metadata
through the training pipeline, eliminating dataset-specific if/else chains.
"""

from dataclasses import dataclass, replace
from typing import List


# Canonical Tox21 task names (NR = nuclear receptor, SR = stress response)
TOX21_TASKS: List[str] = [
    "NR-AR",         # Androgen Receptor (full length)
    "NR-AR-LBD",     # Androgen Receptor Ligand-Binding Domain
    "NR-AhR",        # Aryl Hydrocarbon Receptor
    "NR-Aromatase",  # Aromatase enzyme inhibition
    "NR-ER",         # Estrogen Receptor alpha (full length)
    "NR-ER-LBD",     # Estrogen Receptor alpha Ligand-Binding Domain
    "NR-PPAR-gamma", # Peroxisome Proliferator-Activated Receptor gamma
    "SR-ARE",        # Antioxidant Response Element pathway
    "SR-ATAD5",      # Genotoxicity (ATAD5 reporter)
    "SR-HSE",        # Heat Shock Factor Response Element
    "SR-MMP",        # Mitochondrial Membrane Potential disruption
    "SR-p53",        # p53 tumour-suppressor pathway activation
]

CLINTOX_TASKS: List[str] = ["CT_TOX"]


@dataclass
class TaskConfig:
    """
    Carries all task-specific metadata through the training pipeline.

    Using a single TaskConfig object instead of string comparisons keeps
    every function dataset-agnostic — add a new dataset by adding a new
    TaskConfig, without touching any training or evaluation code.

    Attributes:
        name:           Dataset identifier ("clintox" | "tox21").
        task_names:     Ordered list of task column names in the DataFrame.
        primary_metric: Metric used for model selection and early stopping.
        loss_type:      Loss function identifier ("focal" | "masked_focal").
    """
    name: str
    task_names: List[str]
    primary_metric: str
    loss_type: str

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    @property
    def is_multitask(self) -> bool:
        return self.num_tasks > 1


# Pre-built instances — import these instead of constructing from scratch
CLINTOX_CONFIG = TaskConfig(
    name="clintox",
    task_names=CLINTOX_TASKS,
    primary_metric="f1",
    loss_type="focal",
)

TOX21_CONFIG = TaskConfig(
    name="tox21",
    task_names=TOX21_TASKS,
    primary_metric="mean_auc_roc",
    loss_type="masked_focal",
)


def get_task_config(dataset_name: str, loss_type: str = None) -> TaskConfig:
    """
    Return a TaskConfig for the given dataset name.

    Args:
        dataset_name: "clintox" or "tox21".
        loss_type:    Override the default loss function (optional).

    Returns:
        TaskConfig instance.
    """
    registry = {"clintox": CLINTOX_CONFIG, "tox21": TOX21_CONFIG}
    if dataset_name not in registry:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    cfg = registry[dataset_name]
    if loss_type is not None and loss_type != cfg.loss_type:
        cfg = replace(cfg, loss_type=loss_type)
    return cfg
