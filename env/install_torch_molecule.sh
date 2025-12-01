#!/bin/bash
# Comprehensive script to ensure torch-molecule is properly installed
# This script handles PyTorch version compatibility and dependency ordering

set -e  # Exit on error

echo "========================================="
echo "Installing torch-molecule (Complete Fix)"
echo "========================================="
echo ""

# Check if we're in the conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "drug-tox-env" ]]; then
    echo "⚠️  Please activate the conda environment first:"
    echo "   conda activate drug-tox-env"
    echo ""
    echo "Activating now..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate drug-tox-env
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo "✓ Python version: $(python --version)"
echo ""

# Verify we're using Python 3.11
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "❌ Error: Python version is $PYTHON_VERSION, but we need 3.11"
    echo "Please recreate the environment with Python 3.11"
    exit 1
fi

echo "Step 1/5: Checking current PyTorch installation..."
python -c "import torch; print(f'Current PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "PyTorch not found, installing..."
    conda install -y pytorch
}

TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('.'); print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "2.0")

echo ""
echo "Step 2/5: Checking NumPy version (must be < 2.0)..."
NUMPY_VERSION=$(python -c "import numpy; v=numpy.__version__.split('.'); print(f'{v[0]}.{int(v[1])}')" 2>/dev/null || echo "2.0")
if [[ "${NUMPY_VERSION%%.*}" -ge "2" ]]; then
    echo "⚠️  NumPy 2.x detected, downgrading to 1.x for compatibility..."
    pip install "numpy<2.0" --quiet
    echo "✓ NumPy downgraded"
else
    echo "✓ NumPy version OK"
fi

echo ""
echo "Step 3/5: Installing torch-scatter and torch-geometric..."

# Try different PyTorch versions for wheel compatibility
# PyTorch 2.4-2.6 have the best wheel support
if [[ "$TORCH_VERSION" == "2.9" ]] || [[ "$TORCH_VERSION" == "2.8" ]] || [[ "$TORCH_VERSION" == "2.7" ]]; then
    echo "⚠️  PyTorch $TORCH_VERSION is very new - wheels may not be available"
    echo "    Downgrading to PyTorch 2.4.0 for better compatibility..."
    
    pip uninstall -y torch torchvision torchaudio torch-scatter torch-geometric 2>/dev/null || true
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
    
    TORCH_VERSION="2.4"
    echo "✓ PyTorch 2.4.0 installed"
fi

# Remove existing installations
pip uninstall -y torch-scatter torch-geometric 2>/dev/null || true

# Try installing from PyG wheels for the current PyTorch version
echo "Installing from PyG wheels for PyTorch ${TORCH_VERSION}..."
TORCH_MAJOR_MINOR="${TORCH_VERSION}.0"

if pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_MAJOR_MINOR}+cpu.html --quiet 2>/dev/null; then
    echo "✓ Successfully installed from PyG wheels"
elif pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html --quiet 2>/dev/null; then
    echo "✓ Successfully installed from PyG wheels (fallback)"
else
    echo "⚠️  PyG wheels not available, trying standard installation..."
    pip install torch-scatter torch-geometric 2>&1 | tail -5
fi

echo ""
echo "Step 4/5: Verifying torch-scatter and torch-geometric..."
if python -c "import torch_scatter; import torch_geometric; print('✓ Both imported successfully')" 2>/dev/null; then
    echo "✓ Verification passed"
else
    echo "❌ Verification failed - torch-scatter/torch-geometric not working"
    echo "   Trying alternative installation method..."
    
    # Try installing with --no-build-isolation
    pip install torch-scatter torch-geometric --no-build-isolation 2>&1 | tail -5
    
    if ! python -c "import torch_scatter; import torch_geometric" 2>/dev/null; then
        echo "❌ Error: Could not install torch-scatter/torch-geometric"
        echo "   Please see TROUBLESHOOTING.md for alternative solutions"
        exit 1
    fi
fi

echo ""
echo "Step 5/5: Installing torch-molecule and dependencies..."

# Ensure scipy is up to date (torch-molecule requires >= 1.14.1)
echo "Upgrading scipy to meet torch-molecule requirements..."
pip install "scipy>=1.14.1" --upgrade --quiet 2>&1 | tail -3 || {
    # If scipy 1.14.1+ doesn't exist, use latest available
    echo "⚠️  scipy>=1.14.1 not available, installing latest..."
    pip install scipy --upgrade --quiet
}

# Install missing dependencies first
echo "Installing required dependencies..."
pip install "huggingface_hub>=0.22.2" "optuna>=4.0.0" --quiet 2>&1 | tail -3

# Install dataset loading dependencies
echo "Installing dataset dependencies (deepchem, pytdc)..."
pip install deepchem pytdc --quiet 2>&1 | tail -3

# Now install torch-molecule
echo "Installing torch-molecule..."
if pip install torch-molecule --quiet 2>&1 | tail -5; then
    echo "✓ torch-molecule installed"
else
    echo "⚠️  Standard installation failed, trying with verbose output..."
    pip install torch-molecule 2>&1 | tail -10
fi

echo ""
echo "========================================="
echo "Final Verification"
echo "========================================="

# Verify everything works
echo "Testing imports..."

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || exit 1
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || exit 1
python -c "import torch_scatter; print('✓ torch-scatter')" || exit 1
python -c "import torch_geometric; print('✓ torch-geometric')" || exit 1
python -c "import deepchem; print('✓ deepchem')" || exit 1
python -c "from tdc.single_pred import Tox; print('✓ pytdc')" || exit 1

# Test torch-molecule import
if python -c "import torch_molecule; print('✓ torch-molecule')" 2>/dev/null; then
    echo ""
    echo "🎉 SUCCESS! All packages installed and working!"
    echo ""
    echo "You can now run:"
    echo "  python -c 'import torch_molecule; print(\"Working!\")'"
    echo ""
else
    echo ""
    echo "⚠️  torch-molecule installed but import failed"
    echo "   This may be due to binary incompatibilities"
    echo "   Trying to diagnose..."
    
    python -c "import torch_molecule" 2>&1 | head -10
    
    echo ""
    echo "If this persists, see TROUBLESHOOTING.md"
    exit 1
fi

