#!/bin/bash
# Quick start script for CAAA experiment
set -e

echo "CAAA: Context-Aware Anomaly Attribution"
echo "========================================="

python3 --version

# If --data rcaeval is passed to this script, verify data directory first.
if echo "$@" | grep -q -- "--data rcaeval"; then
    DATA_DIR="${DATA_DIR:-data/raw}"
    if [ ! -d "$DATA_DIR" ]; then
        echo ""
        echo "ERROR: RCAEval data directory '$DATA_DIR' not found."
        echo "Please download the dataset first:"
        echo "  python -m src.data_loader.download_data"
        echo ""
        exit 1
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the main pipeline
echo ""
echo "Running CAAA pipeline..."
python -m src.main \
    --n-fault 50 \
    --n-load 50 \
    --epochs 50 \
    --model caaa \
    --output outputs/results

echo ""
echo "Experiment complete! Check outputs/results/ for results."
