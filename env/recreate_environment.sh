#!/bin/bash
# Script to recreate the conda environment with Python 3.11
# This fixes kernel crashes due to Python version mismatch

set -e  # Exit on error

echo "⚠️  This will remove the existing 'drug-tox-env' conda environment."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo ""
echo "Step 1/4: Removing old environment..."
conda env remove -n drug-tox-env -y

echo ""
echo "Step 2/4: Creating new environment with Python 3.11..."
conda env create -f env/environment.yml

echo ""
echo "Step 3/4: Activating environment and installing PyG extensions..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drug-tox-env

# Get torch version for PyG wheels
TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('.'); print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "2.0")
TORCH_MAJOR_MINOR="${TORCH_VERSION}.0"

echo "Detected PyTorch version: ${TORCH_VERSION}"
echo "Installing torch-geometric and torch-scatter from PyG wheels..."

# Try installing from PyTorch Geometric wheels first
if pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_MAJOR_MINOR}+cpu.html 2>/dev/null; then
    echo "✓ Successfully installed from PyG wheels"
elif pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html 2>/dev/null; then
    echo "✓ Successfully installed from PyG wheels (fallback)"
else
    echo "⚠️  PyG wheels not available, using generic pip install..."
    pip install torch-scatter torch-geometric || {
        echo "Error: Failed to install torch-scatter/torch-geometric"
        exit 1
    }
fi

# Install torch-molecule
echo "Installing torch-molecule..."
pip install torch-molecule || {
    echo "Error: Failed to install torch-molecule"
    exit 1
}

# Install missing dependencies
echo "Installing missing dependencies..."
pip install "huggingface_hub>=0.22.2" "optuna>=4.0.0"

echo ""
echo "Step 4/4: Installing Jupyter kernel..."
python -m ipykernel install --user --name drug-tox-env --display-name "Python (drug-tox-env)" || {
    # Remove old kernel first if it exists
    jupyter kernelspec remove drug-tox-env -f 2>/dev/null || true
    python -m ipykernel install --user --name drug-tox-env --display-name "Python (drug-tox-env)"
}

echo ""
echo "✓ Environment recreated successfully!"
echo ""
echo "Verifying Python version:"
python --version
echo ""
echo "To activate the environment, run:"
echo "  conda activate drug-tox-env"
echo ""
echo "To test torch-molecule:"
echo "  python -c 'import torch_molecule; print(\"✓ torch-molecule works!\")'"

