#!/bin/bash
# Activate the medical-mtl virtual environment
# Usage: source activate.sh

echo "Activating medical-mtl environment..."
source medical-mtl/Scripts/activate

echo ""
echo "========================================"
echo "Medical MTL Environment Activated!"
echo "========================================"
echo ""
echo "Python: $(which python)"
echo ""
echo "Next steps:"
echo "  1. Download datasets: python scripts/download_datasets_hf.py --all"
echo "  2. Run validation: python validate_setup.py"
echo "  3. Start experiments!"
echo ""
