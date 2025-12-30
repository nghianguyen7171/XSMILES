#!/usr/bin/env python3
"""
Generate molecular structure visualizations for dataset examples.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from PIL import Image
import io

from src.data import load_clintox
from src.utils import set_seed

def draw_molecule_high_quality(smiles, size=(600, 600), title=""):
    """Draw molecule with high quality rendering."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates
    from rdkit.Chem import AllChem
    AllChem.Compute2DCoords(mol)
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))
    return img

def generate_examples():
    """Generate molecular structure visualizations."""
    set_seed(42)
    
    # Load dataset
    print("Loading dataset...")
    train_df, _, _ = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    # Select examples
    non_toxic_samples = train_df[train_df['CT_TOX'] == 0].head(2)
    toxic_samples = train_df[train_df['CT_TOX'] == 1].head(2)
    
    # Create output directory
    output_dir = project_root / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate non-toxic examples
    print("\nGenerating non-toxic examples...")
    for idx, row in non_toxic_samples.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        
        print(f"  Sample {idx}: {smiles}")
        print(f"    Formula: {formula}, MW: {mw:.2f}")
        
        img = draw_molecule_high_quality(smiles, size=(800, 800))
        if img:
            save_path = output_dir / f"non_toxic_example_{idx}.png"
            img.save(save_path)
            print(f"    Saved: {save_path}")
    
    # Generate toxic examples
    print("\nGenerating toxic examples...")
    for idx, row in toxic_samples.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        
        print(f"  Sample {idx}: {smiles}")
        print(f"    Formula: {formula}, MW: {mw:.2f}")
        
        img = draw_molecule_high_quality(smiles, size=(800, 800))
        if img:
            save_path = output_dir / f"toxic_example_{idx}.png"
            img.save(save_path)
            print(f"    Saved: {save_path}")
    
    # Generate comparison figure
    print("\nGenerating comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    for i, (idx, row) in enumerate(non_toxic_samples.iterrows()):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        img = draw_molecule_high_quality(smiles, size=(600, 600))
        if img:
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Non-toxic Sample {i+1}\n{smiles}\nFormula: {formula}', 
                               fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
    
    for i, (idx, row) in enumerate(toxic_samples.iterrows()):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        img = draw_molecule_high_quality(smiles, size=(600, 600))
        if img:
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'Toxic Sample {i+1}\n{smiles}\nFormula: {formula}', 
                               fontsize=12, fontweight='bold', color='red')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    comparison_path = output_dir / "molecular_examples_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {comparison_path}")
    
    print("\n✓ Examples generated successfully!")

if __name__ == "__main__":
    generate_examples()

