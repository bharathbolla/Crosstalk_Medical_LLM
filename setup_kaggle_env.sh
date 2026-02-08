#!/bin/bash
# Kaggle Environment Setup Script
# Creates isolated venv to avoid dependency conflicts

set -e  # Exit on error

echo "=================================================="
echo "Setting up isolated Python environment for Kaggle"
echo "=================================================="

# Step 1: Create virtual environment
echo ""
echo "1. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "✓ Virtual environment created and activated"

# Step 2: Upgrade pip
echo ""
echo "2. Upgrading pip..."
pip install --upgrade pip -q

echo "✓ pip upgraded"

# Step 3: Install compatible package versions
echo ""
echo "3. Installing compatible packages..."

# Install in specific order to avoid conflicts
pip install -q pyarrow==14.0.0
pip install -q datasets==2.20.0
pip install -q transformers==4.40.0
pip install -q evaluate==0.4.2
pip install -q wandb==0.17.0
pip install -q accelerate==0.30.0
pip install -q scikit-learn==1.5.0
pip install -q pyyaml==6.0.1

echo "✓ All packages installed"

# Step 4: Verify installation
echo ""
echo "4. Verifying installation..."
python -c "import datasets; print(f'datasets version: {datasets.__version__}')"
python -c "import pyarrow; print(f'pyarrow version: {pyarrow.__version__}')"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo ""
echo "=================================================="
echo "✅ Environment setup complete!"
echo "=================================================="
echo ""
echo "To use this environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run test:"
echo "  python test_parsers.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo "=================================================="
