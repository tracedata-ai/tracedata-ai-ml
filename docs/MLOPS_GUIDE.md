# Complete MLOps Setup Guide

> **Beginner path:** follow **[GETTING_STARTED.md](GETTING_STARTED.md)** end-to-end first.  
> **Two training pipelines exist** — do not mix them up:

| Pipeline | Command | Config | Features |
|----------|---------|--------|----------|
| **Production (recommended)** | `uv run python -m src.mlops.production_window_training` | `production_mlops.yaml` | **3** (from `extract_smoothness_features`) |
| **Research / synthetic trips** | `uv run python -m src.mlops.training_pipeline` | `mlops_config.yaml` | **18** (synthetic trip aggregates) |

Install tools with **`uv sync`** at the repo root; avoid relying on a manual `pip install` list unless you know you need it.

---

## Overview

Your project now has a **production-ready MLOps pipeline** with:
- ✅ Reproducible synthetic data generation
- ✅ MLFlow experiment tracking
- ✅ Automated training pipeline
- ✅ Cross-validation strategy
- ✅ Model versioning & registry
- ✅ Complete metrics tracking

Since you don't have real telematics data, we use synthetic data generation that simulates realistic driving behaviors.

---

## Architecture

```
┌─────────────────────────────────────┐
│  MLOps Configuration (YAML)         │
│  • Model hyperparameters            │
│  • Experiment settings              │
│  • Data generation config           │
│  • Reproducibility seeds            │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Synthetic Data Generation          │
│  • Driver profiles (smooth→unsafe)  │
│  • Telematics windows (10-min)      │
│  • 18 aggregated features           │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Data Splits                        │
│  • Train: 60% (vehicle dev)         │
│  • Val:   20%                       │
│  • Test:  20%                       │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Model Training with MLFlow         │
│  • XGBoost (18 features)            │
│  • Cross-validation (5-fold)        │
│  • Early stopping on validation     │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Evaluation & Metrics               │
│  • R², RMSE, MAE on test            │
│  • CV scores for stability          │
│  • Quality gate (R² ≥ 0.85)         │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  MLFlow Logging                     │
│  • Model artifact                   │
│  • Metrics (train/val/test)         │
│  • Hyperparameters                  │
│  • Feature importance               │
│  • Metadata                         │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Model Registry                     │
│  • Save best model                  │
│  • Version tracking                 │
│  • Run comparison                   │
└────────────────────────────────────┘
```

---

## Files Created/Modified

### Configuration
- **`mlops_config.yaml`** (NEW)
  - All hyperparameters, experiment settings, data config
  - Reproducibility seeds, feature ranges, monitoring settings
  - Environment-specific (dev/prod) configurations

### Core MLOps Code
- **`src/mlops/__init__.py`** (NEW)
  - MLOps module entry point

- **`src/mlops/training_pipeline.py`** (NEW)
  - Complete training orchestration with MLFlow
  - Data generation, training, evaluation, logging
  - Cross-validation and quality gates

### Data Generation
- **`src/utils/data_generation_strategy.py`** (NEW)
  - Synthetic telematics generation
  - 4 driver profiles (smooth, normal, jerky, unsafe)
  - 18-feature aggregation matching real format

---

## Quick Start

### 1. Install dependencies
```bash
# From project root
uv sync
```

### 2. Run a training pipeline
**Production (3-feature) model:**
```bash
uv run python -m src.mlops.production_window_training
```

**Research (18-feature) pipeline:**
```bash
uv run python -m src.mlops.training_pipeline
```

The **18-feature** command will, in order:

1. Generate ~300 synthetic trips (defaults: 20 drivers × 15 trips; see `mlops_config.yaml`)
2. Split into train/val/test
3. Train XGBoost on **18** columns
4. Run 5-fold cross-validation
5. Evaluate on all sets
6. Log to MLflow
7. Save a model artifact when the quality gate passes
8. Print metrics

The **production** command (`production_window_training`) instead builds **10-minute ping windows**, extracts **3** features, and logs `model/` plus `serving/` — see **[GETTING_STARTED.md](GETTING_STARTED.md)**.

**Example terminal output (18-feature run):**
```
======================================================================
🚀 STARTING MLOps TRAINING PIPELINE
======================================================================
🔄 Generating synthetic data...
✅ Generated 300 synthetic trips
   Drivers: 20
   Trips per driver: 15
   Profiles: {'normal': 120, 'smooth': 105, 'jerky': 45, 'unsafe': 30}

📊 Data split:
   Train: 180 (60.0%)
   Val:   60 (20.0%)
   Test:  60 (20.0%)

📊 Features: 18
   avg_accel_g, avg_accel_std, max_decel_g, ...

🎯 Labels: generated 300 synthetic labels

🚀 Training XGBoost model...
✅ Training complete

📈 Train Metrics:
   mae: 8.3214
   rmse: 11.5462
   r2_score: 0.8923
   
📈 Validation Metrics:
   mae: 9.1523
   rmse: 12.3141
   r2_score: 0.8712
   
📈 Test Metrics:
   mae: 8.7634
   rmse: 11.8923
   r2_score: 0.8834

🔄 Cross-validation (5 folds)...
   Mean R²: 0.8756 (±0.0234)

✅ Quality gate PASSED: R²=0.8834

📝 Logging to MLFlow...
✅ Logged to MLFlow

======================================================================
✅ MLOps PIPELINE COMPLETE
======================================================================

📊 FINAL RESULTS:
   Test R²: 0.8834
   Test RMSE: 11.8923
   CV R² Mean: 0.8756
   MLFlow Run: abc123def456
```

### 3. View MLFlow Experiments
```bash
# Start MLFlow UI
mlflow ui

# Then open http://localhost:5000
```

---

## Synthetic Data Generation Strategy

### Driver Profiles

| Profile | Smoothness | Aggression | Speed Control | Efficiency |
|---------|-----------|------------|---------------|-----------|
| Smooth | 0.9 | 0.05 | 0.1 | 0.9 |
| Normal | 0.6 | 0.3 | 0.4 | 0.6 |
| Jerky | 0.3 | 0.6 | 0.7 | 0.4 |
| Unsafe | 0.1 | 0.85 | 0.9 | 0.2 |

Each profile affects:
- Jerk patterns (smoothness)
- Harsh event frequency (aggression)
- Speed consistency (speed control)
- RPM management (efficiency)

### Data Generation Parameters (mlops_config.yaml)

```yaml
data:
  generation:
    drivers_to_simulate: 20        # Number of drivers
    trips_per_driver: 15           # Trips each driver takes
    trip_duration_minutes:
      min: 20
      max: 180
    driver_styles:
      smooth: 0.35   # 35% smooth drivers
      normal: 0.40   # 40% average drivers
      jerky: 0.15    # 15% poor drivers
      unsafe: 0.10   # 10% dangerous drivers
```

### Data Format
Each trip generates:
- 12 telematics windows (1 per 10 minutes, 2-hour trip)
- Each window has sensor readings
- Aggregated into 18 trip-level features
- Labels generated with physics-driven formula

---

## MLFlow Features

### 1. Experiment Tracking
Each run logs:
- **Parameters**: model hyperparameters, data config
- **Metrics**: R², RMSE, MAE, CV scores
- **Artifacts**: model, feature importance, metadata

### 2. Model Registry
```bash
# View all experiments
mlflow experiments list

# View runs in experiment
mlflow runs list --experiment-name "smoothness-scoring-production"

# Compare runs
mlflow run compare <run_id_1> <run_id_2>
```

### 3. Model Versioning
Models are saved with:
- Training timestamp
- Data version (synthetic v1.0)
- Full metrics
- Complete configuration
- Feature importance

### 4. Reproducibility
All runs are reproducible using:
- Fixed random seeds (numpy, sklearn, xgboost)
- Serialized data (train/val/test CSVs)
- Configuration snapshots (YAML)
- MLFlow run tracking

---

## Configuration Management

### Environment-Specific Settings

**Development** (mlops_config.yaml):
```yaml
environment:
  dev:
    num_drivers: 20          # Small dataset for fast iteration
    trips_per_driver: 15
    test_size: 0.2
    quick_mode: false
```

**Production** (production settings):
```yaml
environment:
  prod:
    num_drivers: 100         # Larger dataset for stability
    trips_per_driver: 50
    test_size: 0.15
    quick_mode: false
```

Switch with environment variable:
```bash
export MLOPS_ENV=prod && python -m src.mlops.training_pipeline
```

---

## Model Evaluation

### Metrics Tracked

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| R² | 1 - (SS_res / SS_tot) | % variance explained (0-1) |
| RMSE | √(MSE) | Average error magnitude |
| MAE | mean(\|actual - pred\|) | Absolute average error |
| Median AE | median(\|residuals\|) | Robust to outliers |

### Quality Gate
```yaml
model:
  quality_gate:
    min_r2_score: 0.85    # ✅ PASS if R² ≥ 0.85
    min_rmse: 15.0        # ✅ PASS if RMSE ≤ 15
```

If quality gate fails:
- Model is NOT saved
- Run is marked as FAILED
- Review metrics in MLFlow UI

---

## Cross-Validation

5-fold cross-validation provides:
- **Estimate of generalization** - how well model performs on unseen data
- **Stability estimate** - consistency across different data splits
- **Data efficiency** - all data used for training and evaluation

```
Fold 1: Train on 80%, Test on 20%
Fold 2: Train on 80%, Test on 20%
Fold 3: Train on 80%, Test on 20%
Fold 4: Train on 80%, Test on 20%
Fold 5: Train on 80%, Test on 20%
────────────────────────────────────
Average R² ± std: 0.8756 ± 0.0234
```

---

## Data Splits Strategy

### Train (60%)
- Used to train the model
- Model learns from this data
- Larger for stable training

### Validation (20%)
- Used for early stopping
- Helps prevent overfitting
- Model doesn't directly learn from this

### Test (20%)
- Used for final evaluation
- Simulates production performance
- Never seen during training

**Why this split?**
- Train/val prevents overfitting
- Test set is truly independent
- Reflects production scenario where you score new trips

---

## Transitioning to Real Data

When you have real telematics data:

### Step 1: Create data loader
```python
def load_real_trips_from_s3():
    # Load from your data source
    # Must have same 18 features
    return df
```

### Step 2: Update configuration
```yaml
data:
  generation:
    source: "s3"           # Instead of "synthetic"
    s3_bucket: "your-bucket"
    s3_prefix: "trips/"
```

### Step 3: Update pipeline
```python
class MLOpsTrainingPipeline:
    def generate_data(self):
        if self.config['data']['generation']['source'] == 'real':
            return load_real_trips_from_s3()
        else:
            # Use synthetic
            pipeline = SyntheticDataPipeline(...)
```

### Step 4: Re-train
```bash
python -m src.mlops.training_pipeline
```

Model will automatically use real data with same pipeline!

---

## Advanced Features

### Feature Scaling
```yaml
data:
  feature_scaling:
    method: "standardize"   # or "minmax"
    fit_on_split: "train"   # Fit scaler on training data
```

### Data Quality Checks
```yaml
data_quality:
  max_missing_percent: 5
  remove_outliers: false
  feature_ranges:
    avg_accel_g: [0.0, 1.0]
    # ... other ranges
```

### SHAP Analysis (Disabled by Default)
```yaml
evaluation:
  compute_shap_values: true   # Slow but informative
  shap_sample_size: 50
```

Enables advanced feature importance analysis.

---

## Monitoring Deployed Models

### Performance Monitoring
```yaml
monitoring:
  enabled: true
  alerts:
    metric_degradation_threshold: 0.05   # Alert if R² drops 5%
    drift_detection_enabled: true
    prediction_drift_threshold: 0.10
```

### Data Drift Detection
Monitor if new trips have different feature distributions:
```
Normal range: avg_jerk = 0.008 ± 0.003
Alert if: avg_jerk = 0.025 (out of range)
```

---

## Troubleshooting

### Issue: "Quality gate FAILED"
**Cause**: Model R² < 0.85
**Solution**: 
- Increase training data (more drivers/trips)
- Adjust hyperparameters in mlops_config.yaml
- Verify feature generation

### Issue: "ModuleNotFoundError: mlflow"
**Cause**: MLFlow not installed
**Solution**: 
```bash
pip install mlflow
```

### Issue: "YAML parsing error"
**Cause**: Syntax error in mlops_config.yaml
**Solution**:
```bash
python -m yaml mlops_config.yaml  # Validate
```

### Issue: MLFlow UI shows no experiments
**Cause**: Using wrong tracking URI
**Solution**:
```bash
# Check current URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# View runs directory
ls -la mlruns/
```

---

## Next Steps

1. **Run training**: `python -m src.mlops.training_pipeline`
2. **View results**: `mlflow ui` then open http://localhost:5000
3. **Collect real data**: When available, update data source
4. **Deploy model**: Use saved model in API or batch scorer
5. **Monitor in production**: Track drift and performance

---

## Key Files to Remember

| File | Purpose |
|------|---------|
| `mlops_config.yaml` | Central config - edit here for tuning |
| `src/mlops/training_pipeline.py` | Main training orchestration |
| `src/utils/data_generation_strategy.py` | Synthetic data creation |
| `models/smoothness_model.joblib` | Saved best model |
| `data/train.csv` | Training dataset (reproducible) |
| `reports/feature_importance.json` | What features matter most |
| `mlruns/` | MLFlow experiment tracking |
| `mlflow.db` | MLFlow metadata database |

---

## Summary

You now have:
- ✅ **Reproducible synthetic data** - same seed = same data
- ✅ **MLFlow tracking** - all experiments logged
- ✅ **Automated training** - one command to train
- ✅ **Cross-validation** - ensures model stability
- ✅ **Quality gates** - prevents bad models shipping
- ✅ **Production-ready** - ready for real data when available
- ✅ **Monitoring** - detect performance degradation

**Ready to train?** Run: `python -m src.mlops.training_pipeline`
