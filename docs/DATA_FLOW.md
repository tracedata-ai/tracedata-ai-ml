# Data flow (short)

> Read **[GETTING_STARTED.md](GETTING_STARTED.md)** to run the project.  
> Read **[SCORING_PATHS.md](SCORING_PATHS.md)** for the full **3 vs 18 feature** story and which class to call.

---

## Path A — Pings → 3 features (default product path)

Matches **`production_window_training`**, **`extract_smoothness_features`**, **`SmoothnessInference`**.

**Trip API:** `score_trip_from_ping_windows(windows)` — each inner list is pings for one ~10-minute bucket; output is **one** `trip_smoothness_score` and **one** `explanation` (aggregated attributions).

```
┌─────────────────────────────────────────────────────────────┐
│  Many windows × (pings in ~10 min, e.g. ~30 s spacing)      │
│  Each ping: speed_kmh, acceleration_ms2 (+ optional GPS)    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Per window: extract_smoothness_features() → 3 numbers        │
│  • accel_fluidity • driving_consistency • comfort_zone_%      │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  XGBoost per window → score; weighted mean → trip score       │
│  + aggregated pred_contribs → single trip explanation         │
└─────────────────────────────────────────────────────────────┘
```

**Safety (rules, not this XGBoost model):** `detect_safety_events()` on pings. **`ScoringService`** (`src/core/scoring.py`) can combine smoothness + safety against SQLite for demos.

---

## Path B — Device `smoothness_log` → 18 features

Matches **`training_pipeline`** (synthetic rows with the same column names), **`parse_telematics_event`**, **`DeviceAggregateTripScorer`**.

**Trip API:** `score_trip_at_end(envelopes)` — one envelope per ~10-minute device aggregate; **one** trip score and aggregated explanation.

```
┌─────────────────────────────────────────────────────────────┐
│  Per window: smoothness_log JSON (details.speed, .jerk, …)   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Row aligned to 18 training columns (device_window_features)  │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  XGBoost per window → weighted trip score + attributions     │
└─────────────────────────────────────────────────────────────┘
```

Example payload shape: see the JSON block at the top of the root **`README.md`** (illustrative device event).

---

## Do not mix paths

A model trained on **3** features only accepts ping-derived inputs via **`SmoothnessInference`**.  
A model trained on **18** aggregate columns only accepts device-style rows via **`DeviceAggregateTripScorer`**.  
Check **`serving/model_contract.json`** `feature_columns` after each MLflow run.
