# Smoothness ML Scoring Engine

Vehicle telematics smoothness scoring with XGBoost and per-feature attribution (XGBoost `pred_contribs`, same additive decomposition as SHAP for trees).

## Two scoring paths (read this first)

| | **Pings → 3 features** | **Device aggregates → 18 features** |
|--|-------------------------|--------------------------------------|
| **When** | Backend has **raw pings** (`acceleration_ms2`, `speed_kmh`) | Device sends **`smoothness_log`**-style JSON (or you train on the same 18 columns) |
| **Train** | `production_window_training` / `tracedata-mlops production` | `training_pipeline` / `tracedata-mlops synthetic` |
| **Score** | `SmoothnessInference.score_trip_from_ping_windows(windows)` | `DeviceAggregateTripScorer.score_trip_at_end(envelopes)` |

Full narrative: **[docs/SCORING_PATHS.md](docs/SCORING_PATHS.md)**.

---

### Example: device `smoothness_log` payload (Path B / 18-feature world)

Illustrative JSON — **not** used by the 3-feature ping model:
```
{
  "batch_id": "BAT-6ba7b815-9dad-11d1-80b4-00c04fd430c8",
  "ping_type": "batch",
  "source": "telematics_device",
  "is_emergency": false,
  "event": {
    "event_id": "EVT-6ba7b816-9dad-11d1-80b4-00c04fd430c8",
    "device_event_id": "TEL-6ba7b817-9dad-11d1-80b4-00c04fd430c8",
    "trip_id": "TRP-2026-a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6",
    "driver_id": "DRV-ANON-7829",
    "truck_id": "T12345",
    "batch_id": "BAT-6ba7b815-9dad-11d1-80b4-00c04fd430c8",
    "event_type": "smoothness_log",
    "category": "normal_operation",
    "priority": "low",
    "timestamp": "2026-03-07T08:20:00Z",
    "offset_seconds": 1200,
    "trip_meter_km": 24.8,
    "odometer_km": 180224.8,
    "location": { "lat": 1.3485, "lon": 103.8380 },
    "schema_version": "event_v1",
    "details": {
      "sample_count": 600,
      "window_seconds": 600,
      "speed": {
        "mean_kmh": 72.3,
        "std_dev": 8.1,
        "max_kmh": 94.0,
        "variance": 65.61
      },
      "longitudinal": {
        "mean_accel_g": 0.04,
        "std_dev": 0.12,
        "max_decel_g": -0.31,
        "harsh_brake_count": 0,
        "harsh_accel_count": 0
      },
      "lateral": {
        "mean_lateral_g": 0.02,
        "max_lateral_g": 0.18,
        "harsh_corner_count": 0
      },
      "jerk": {
        "mean": 0.008,
        "max": 0.041,
        "std_dev": 0.006
      },
      "engine": {
        "mean_rpm": 1820,
        "max_rpm": 2340,
        "idle_seconds": 45,
        "idle_events": 1,
        "longest_idle_seconds": 38,
        "over_rev_count": 0,
        "over_rev_seconds": 0
      },
      "incident_event_ids": [],
      "raw_log_url": "s3://tracedata-sensors/T12345-batch-20260307-0820.bin"
    },
    "evidence": null
  }
}
```

## Core capability (Path A — pings)

**One trip** = a list of **10-minute ping windows** → **one** smoothness score and **one** explanation.

```python
from src.inference import SmoothnessInference
from src.utils.simulator import generate_telemetry

inf = SmoothnessInference.from_local_paths(
    "models/smoothness_model.joblib",
    "path/to/serving",  # from MLflow run artifacts
)
w1 = generate_telemetry("smooth", duration_minutes=10)
w2 = generate_telemetry("smooth", duration_minutes=10)
result = inf.score_trip_from_ping_windows([w1, w2])
# result["trip_smoothness_score"], result["explanation"]
```

CLI-style demo: `uv run python scripts/score_10min_window.py`.

**Path B** (device JSON): `DeviceAggregateTripScorer` — see [docs/SCORING_PATHS.md](docs/SCORING_PATHS.md).

## Layout

```
src/
├── core/
│   ├── features.py              # extract_smoothness_features (ping window → 3 features)
│   ├── model_contract.py        # Feature order + contract version
│   ├── scoring.py               # Trip DB scoring
│   └── explain.py               # Legacy SHAP/LIME helpers
├── inference/
│   ├── smoothness_inference.py   # Pings → 3 features → score_trip_from_ping_windows
│   └── device_trip_scorer.py     # smoothness_log → 18 features → score_trip_at_end
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

1. **[docs/SCORING_PATHS.md](docs/SCORING_PATHS.md)** — **3 vs 18 features** (how to describe the system)
2. **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — step-by-step (start here)
3. **[docs/README.md](docs/README.md)** — full doc index

## Configuration

- **Production training**: `production_mlops.yaml` (MLflow URI, sample counts, quality gate).
- **18-feature pipeline**: `mlops_config.yaml`.

## Notes

- **MLflow**: `production_window_training` logs `model/` and `serving/` (contract JSON + background rows) for Docker or the main app repo.
- **Attributions**: `SmoothnessInference` uses XGBoost `pred_contribs` (robust with XGBoost 3.x); the `shap` package is still used elsewhere (e.g. `explain.py`) but may need the same pattern if you hit loader issues.
