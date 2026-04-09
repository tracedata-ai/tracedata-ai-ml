#!/bin/bash
# Quick Start Script for MLOps Pipeline

set -e

echo "════════════════════════════════════════════════════════════"
echo "          MLOps Training Pipeline - Quick Start"
echo "════════════════════════════════════════════════════════════"
echo ""

# Step 1: Check Python
echo "✓ Checking Python..."
python --version

# Step 2: Install dependencies
echo ""
echo "✓ Checking dependencies..."
python -c "import mlflow; print(f'  MLFlow: {mlflow.__version__}')" 2>/dev/null || {
    echo "  ⚠ Installing MLFlow..."
    pip install mlflow pyyaml -q
}

python -c "import xgboost; print(f'  XGBoost: {xgboost.__version__}')" 2>/dev/null || {
    echo "  ⚠ Installing XGBoost..."
    pip install xgboost -q
}

# Step 3: Create directories
echo ""
echo "✓ Creating directories..."
mkdir -p models reports data logs mlruns

# Step 4: Show configuration
echo ""
echo "✓ MLOps Configuration:"
grep -A 5 "experiment_name:" mlops_config.yaml | head -2

# Step 5: Run training
echo ""
echo "════════════════════════════════════════════════════════════"
echo "                   Starting Training..."
echo "════════════════════════════════════════════════════════════"
echo ""

python -m src.mlops.training_pipeline

# Step 6: Show results
echo ""
echo "════════════════════════════════════════════════════════════"
echo "                   Training Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results:"
echo "  • Model saved to: models/smoothness_model.joblib"
echo "  • Training data: data/train.csv"
echo "  • Feature importance: reports/feature_importance.json"
echo "  • MLFlow runs: mlruns/"
echo ""
echo "🎯 Next steps:"
echo "  1. View results in MLFlow:"
echo "     mlflow ui"
echo ""
echo "  2. Open in browser:"
echo "     http://localhost:5000"
echo ""
echo "  3. Compare experiments:"
echo "     mlflow experiments list"
echo ""
echo "✅ Setup complete!"
