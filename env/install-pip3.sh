#!/bin/bash
# Installation script for macOS pip3 users
# Handles the torch-scatter dependency issue by installing packages in correct order

# Don't exit on error immediately - we need to handle fallback cases
set +e

echo "Installing dependencies with pip3 (macOS)..."
echo ""

# Step 1: Install base dependencies (including torch)
echo "Step 1/3: Installing base dependencies (torch, numpy, pandas, etc.)..."
pip3 install -r env/requirements-base.txt

# Step 2: Install torch extensions (requires torch to be installed)
echo ""
echo "Step 2/3: Installing torch extensions (torch-scatter, torch-geometric, torch-molecule)..."
echo "Installing from PyTorch Geometric wheel repository for better macOS compatibility..."

# Get torch version to determine correct wheel URL
TORCH_VERSION=$(python3 -c "import torch; v=torch.__version__.split('.'); print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "2.0")
TORCH_MAJOR_MINOR="${TORCH_VERSION}.0"

echo "Detected PyTorch version: ${TORCH_VERSION}"
echo "Attempting to install from PyG wheels for torch ${TORCH_MAJOR_MINOR}..."

# Try installing from PyTorch Geometric wheels first
# Use multiple version attempts in case exact version isn't available
PYG_INSTALLED=0

# Try with .0 suffix (e.g., 2.8.0)
if pip3 install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_MAJOR_MINOR}+cpu.html 2>&1 | grep -q "Successfully installed"; then
    echo "✓ Successfully installed from PyG wheels (torch ${TORCH_MAJOR_MINOR})"
    PYG_INSTALLED=1
# Try with just major.minor (e.g., 2.8)
elif pip3 install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html 2>&1 | grep -q "Successfully installed"; then
    echo "✓ Successfully installed from PyG wheels (torch ${TORCH_VERSION})"
    PYG_INSTALLED=1
fi

# If PyG wheels didn't work, try default installation
if [ $PYG_INSTALLED -eq 0 ]; then
    echo "⚠ PyG wheels not available for torch ${TORCH_VERSION}, trying default installation..."
    echo "  This may take longer and might require compilation tools."
    if ! pip3 install torch-scatter torch-geometric 2>&1 | grep -q "Successfully installed"; then
        echo ""
        echo "❌ Installation failed. torch-scatter doesn't have pre-built wheels for torch ${TORCH_VERSION}."
        echo ""
        echo "Recommended solutions:"
        echo ""
        echo "Option 1: Downgrade to torch 2.4.0 (has PyG wheels):"
        echo "  pip3 uninstall torch -y"
        echo "  pip3 install torch==2.4.0"
        echo "  ./env/install-pip3.sh"
        echo ""
        echo "Option 2: Use conda (recommended for macOS):"
        echo "  conda env create -f env/environment.yml"
        echo "  conda activate drug-tox-env"
        echo ""
        exit 1
    fi
fi

# Then install torch-molecule (which will use the already-installed torch-geometric)
echo ""
echo "Installing torch-molecule..."
pip3 install torch-molecule

# Step 3: Install optional dependencies
echo ""
echo "Step 3/3: Installing optional dependencies..."
pip3 install -r env/requirements-optional.txt

# Set error handling back
set -e

echo ""
echo "✓ All dependencies installed successfully!"
echo ""
echo "To verify installation, run:"
echo "  python3 -c \"import torch; import torch_molecule; import rdkit; print('✓ Core packages installed')\""
