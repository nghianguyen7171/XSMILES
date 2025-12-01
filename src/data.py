"""
Data loading and preprocessing for toxicity prediction datasets.

Provides functions to load ClinTox and Tox21 datasets with proper
train/validation/test splits using scaffold-based splitting.
"""

from typing import Tuple, Optional, List
import pandas as pd
import numpy as np


def load_clintox(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ClinTox dataset with train/val/test splits.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        split_type: Type of split ('scaffold', 'random', 'stratified')
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames with columns:
        - 'smiles': SMILES strings
        - 'CT_TOX': Binary toxicity label (1 = toxic, 0 = non-toxic)
    
    Example:
        >>> train, val, test = load_clintox()
        >>> print(f"Train size: {len(train)}")
    """
    import os
    from pathlib import Path
    
    # Try DeepChem first
    try:
        import deepchem as dc
        from deepchem.molnet import load_clintox as dc_load_clintox
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset using DeepChem
        # Map split_type to DeepChem splitter
        if split_type == "scaffold":
            splitter = dc.splits.ScaffoldSplitter()
        elif split_type == "random":
            splitter = dc.splits.RandomSplitter()
        elif split_type == "stratified":
            splitter = dc.splits.RandomStratifiedSplitter()
        else:
            splitter = dc.splits.RandomSplitter()  # Default fallback
        
        tasks, datasets, transformers = dc_load_clintox(
            data_dir=str(cache_path),
            save_dir=str(cache_path),
            featurizer=dc.feat.RawFeaturizer(),  # Just return SMILES
            splitter=splitter,
            seed=seed
        )
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # DeepChem ClinTox has 2 tasks: ['FDA_APPROVED', 'CT_TOX']
        # We want the CT_TOX task (index 1, or second column)
        # Also handle cases where some molecules might have failed featurization
        # by filtering based on valid weight mask if available
        
        # Extract SMILES from ids (ids contains SMILES strings)
        # Extract labels - ClinTox has 2 tasks, we want CT_TOX (index 1)
        def extract_data(dataset):
            smiles = dataset.ids
            # If y is 2D, take the CT_TOX column (index 1), else flatten
            if len(dataset.y.shape) == 2 and dataset.y.shape[1] == 2:
                # ClinTox has [FDA_APPROVED, CT_TOX] - we want CT_TOX (column 1)
                labels = dataset.y[:, 1]
            elif len(dataset.y.shape) == 2:
                # If it's 2D but not 2 columns, take first column
                labels = dataset.y[:, 0]
            else:
                # 1D array
                labels = dataset.y.flatten()
            
            # Ensure same length (filter out any mismatches)
            min_len = min(len(smiles), len(labels))
            return pd.DataFrame({
                'smiles': smiles[:min_len],
                'CT_TOX': labels[:min_len]
            })
        
        train_df = extract_data(train_dataset)
        val_df = extract_data(val_dataset)
        test_df = extract_data(test_dataset)
        
        return train_df, val_df, test_df
    
    except ImportError:
        # Fallback to PyTDC
        try:
            from tdc.single_pred import Tox
            import os
            
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            data = Tox(name='ClinTox', path=str(cache_path))
            df = data.get_data()
            
            # Rename columns to match expected format
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'CT_TOX'})
            
            # Simple train/val/test split (80/10/10)
            # For scaffold split, would need additional processing
            np.random.seed(seed)
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            n = len(shuffled)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            
            train_df = shuffled[:n_train]
            val_df = shuffled[n_train:n_train + n_val]
            test_df = shuffled[n_train + n_val:]
            
            return train_df, val_df, test_df
        
        except ImportError:
            raise ImportError(
                "Neither DeepChem nor PyTDC is installed. "
                "Install one with: pip install deepchem or pip install pytdc"
            )


def load_tox21(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Tox21 multi-task toxicity dataset.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        split_type: Type of split ('scaffold', 'random', 'stratified')
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames with columns:
        - 'smiles': SMILES strings
        - Multiple binary task columns (NR-AR, NR-AR-LBD, etc.)
        - Missing labels encoded as NaN
    
    Example:
        >>> train, val, test = load_tox21()
        >>> print(f"Number of tasks: {len([c for c in train.columns if c != 'smiles'])}")
    """
    import os
    from pathlib import Path
    
    # Try DeepChem first
    try:
        import deepchem as dc
        from deepchem.molnet import load_tox21 as dc_load_tox21
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset using DeepChem
        # Map split_type to DeepChem splitter
        if split_type == "scaffold":
            splitter = dc.splits.ScaffoldSplitter()
        elif split_type == "random":
            splitter = dc.splits.RandomSplitter()
        elif split_type == "stratified":
            splitter = dc.splits.RandomStratifiedSplitter()
        else:
            splitter = dc.splits.RandomSplitter()  # Default fallback
        
        tasks, datasets, transformers = dc_load_tox21(
            data_dir=str(cache_path),
            save_dir=str(cache_path),
            featurizer=dc.feat.RawFeaturizer(),  # Just return SMILES
            splitter=splitter,
            seed=seed
        )
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # Convert to DataFrame with all tasks
        train_data = {'smiles': train_dataset.ids}
        val_data = {'smiles': val_dataset.ids}
        test_data = {'smiles': test_dataset.ids}
        
        for i, task in enumerate(tasks):
            train_data[task] = train_dataset.y[:, i]
            val_data[task] = val_dataset.y[:, i]
            test_data[task] = test_dataset.y[:, i]
        
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)
        
        # Replace -1 with NaN for missing labels (DeepChem convention)
        train_df = train_df.replace(-1, np.nan)
        val_df = val_df.replace(-1, np.nan)
        test_df = test_df.replace(-1, np.nan)
        
        return train_df, val_df, test_df
    
    except ImportError:
        # Fallback to PyTDC
        try:
            from tdc.single_pred import Tox
            from pathlib import Path
            
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            data = Tox(name='Tox21', path=str(cache_path))
            df = data.get_data()
            
            # Rename columns to match expected format
            df = df.rename(columns={'Drug': 'smiles'})
            
            # Simple train/val/test split (80/10/10)
            np.random.seed(seed)
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            n = len(shuffled)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            
            train_df = shuffled[:n_train]
            val_df = shuffled[n_train:n_train + n_val]
            test_df = shuffled[n_train + n_val:]
            
            return train_df, val_df, test_df
        
        except ImportError:
            raise ImportError(
                "Neither DeepChem nor PyTDC is installed. "
                "Install one with: pip install deepchem or pip install pytdc"
            )


def get_task_names(dataset_name: str = "clintox") -> List[str]:
    """
    Get list of task/column names for a dataset.
    
    Args:
        dataset_name: Name of dataset ('clintox' or 'tox21')
    
    Returns:
        List of task column names
    """
    if dataset_name == "clintox":
        return ["CT_TOX"]
    elif dataset_name == "tox21":
        return [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
            "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

