# Smoothness ML Scoring Engine

Vehicle telematics smoothness scoring with XGBoost and per-feature attribution (XGBoost `pred_contribs`, same additive decomposition as SHAP for trees).

## Core capability

**Score smoothness (0–100)** from a window of pings (e.g. 10 minutes), plus a feature breakdown.

```python
from src.inference import SmoothnessInference
from src.utils.simulator import generate_telemetry

# After training: joblib on disk + serving/ inside the MLflow run (or download both for Docker).
inf = SmoothnessInference.from_run("<mlflow_run_id>", "./mlruns")
# Or: SmoothnessInference.from_local_paths("models/smoothness_model.joblib", Path("path/to/serving"))

pings = generate_telemetry("smooth", duration_minutes=10)
result = inf.score_window(pings)
# result["smoothness_score"], result["features"], result["shap"], result["shap_base_value"]
```

Or load from an MLflow run: `SmoothnessInference.from_run(run_id, tracking_uri="./mlruns")`.

## Layout

```
src/
├── core/
│   ├── features.py              # extract_smoothness_features (ping window → 3 features)
│   ├── model_contract.py        # Feature order + contract version
│   ├── scoring.py               # Trip DB scoring
│   └── explain.py               # Legacy SHAP/LIME helpers
├── inference/
│   └── smoothness_inference.py  # Production-style score_window()
├── mlops/
│   ├── production_window_training.py  # 3-feature model → MLflow (use this for deploy)
│   └── training_pipeline.py           # 18-feature synthetic pipeline (research)
└── utils/
    ├── simulator.py
    └── trainer.py
```

## Quick start

```bash
uv sync
```

**Train the production (3-feature) model and log to MLflow:**

```bash
uv run python -m src.mlops.production_window_training
```

**Optional — 18-feature synthetic experiment:**

```bash
uv run python -m src.mlops.training_pipeline
```

## Features (production model)

Three values from `extract_smoothness_features` (see `src/core/features.py`):

| Feature | Meaning |
|--------|---------|
| `accel_fluidity` | Mean absolute jerk on acceleration |
| `driving_consistency` | Std dev of acceleration |
| `comfort_zone_percent` | % of samples in [-0.5, 0.5] m/s² |

Pings must include `acceleration_ms2` and `speed_kmh` (and optional `timestamp`, `lat`, `lon`).

## Testing

```bash
uv run pytest tests/ -v
```

## Documentation

1. **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — step-by-step (start here)
2. **[docs/README.md](docs/README.md)** — full doc index

## Configuration

- **Production training**: `production_mlops.yaml` (MLflow URI, sample counts, quality gate).
- **18-feature pipeline**: `mlops_config.yaml`.

## Notes

- **MLflow**: `production_window_training` logs `model/` and `serving/` (contract JSON + background rows) for Docker or the main app repo.
- **Attributions**: `SmoothnessInference` uses XGBoost `pred_contribs` (robust with XGBoost 3.x); the `shap` package is still used elsewhere (e.g. `explain.py`) but may need the same pattern if you hit loader issues.
