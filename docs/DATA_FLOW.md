# Data flow (short)

> Read **[GETTING_STARTED.md](GETTING_STARTED.md)** to run the project step by step.

## Production path (what this repo trains by default)

This matches **`src.mlops.production_window_training`**, **`src.core.features`**, and **`src.inference.SmoothnessInference`**.

```
┌─────────────────────────────────────────────────────────────┐
│  Pings in a time window (e.g. ~10 min, ~30 s spacing)      │
│  Each ping: speed_kmh, acceleration_ms2, (+ optional GPS)     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  extract_smoothness_features() → 3 numbers                  │
│  • accel_fluidity                                           │
│  • driving_consistency                                      │
│  • comfort_zone_percent                                     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  XGBoost regressor → smoothness_score (0–100)               │
│  + per-feature contributions (XGBoost pred_contribs)       │
└─────────────────────────────────────────────────────────────┘
```

**Safety scoring** (rule-based, separate from the ML model) uses **`detect_safety_events()`** on the same pings. **`ScoringService`** in `src/core/scoring.py` combines smoothness + safety for trip scoring against SQLite.

## Research / alternate path (18 features)

**`src.mlops.training_pipeline`** builds synthetic **trip-level** tables with **18** aggregated columns (see `src/utils/data_generation_strategy.py`). That model is **not** the same as the 3-feature production model. Use it for experiments only unless you align serving code to those 18 columns.

## Older “device event” diagram

Some older docs describe a rich 18-field **device payload** (longitudinal, lateral, RPM, etc.). That is a **design reference** for future ingestion. The **current** training + serving contract for smoothness in this repo is the **3-feature** path above unless you explicitly use the 18-feature pipeline.
