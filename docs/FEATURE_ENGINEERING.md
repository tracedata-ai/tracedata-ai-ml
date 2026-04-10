# Feature engineering

> New here? Start with **[GETTING_STARTED.md](GETTING_STARTED.md)**.  
> **Which API and training command?** See **[SCORING_PATHS.md](SCORING_PATHS.md)** — same two paths, explained for stakeholders.

This repo has **two different feature worlds** (two models). They correspond to **two input shapes**, not two optional views of the same model.

---

## A) Ping stream → 3 features (default deployment when you have raw pings)

**Code:** `src/core/features.py` → `extract_smoothness_features()`  
**Training:** `uv run python -m src.mlops.production_window_training`  
**Config:** `production_mlops.yaml`

**Input:** A list of ping dicts. Each ping must include:

- `acceleration_ms2`
- `speed_kmh`

Optional: `timestamp`, `lat`, `lon` (used elsewhere; not required for the three numbers below).

**Output (3 features):**

| Feature | Meaning |
|---------|---------|
| `accel_fluidity` | Mean absolute jerk (change in acceleration between consecutive samples). |
| `driving_consistency` | Standard deviation of acceleration across the window. |
| `comfort_zone_percent` | Percent of samples where acceleration is in [-0.5, 0.5] m/s². |

These three values are the **only** inputs to this XGBoost model. Column order is fixed — see `src/core/model_contract.py`.

**Inference:** `SmoothnessInference` — prefer **`score_trip_from_ping_windows([[...], ...])`** for a full trip; one window is `[[pings]]`.

**Safety features (rules, not XGBoost):** `detect_safety_events()` counts harsh brake/accel/speeding thresholds on the same pings.

---

## B) Device aggregates → 18 features (when the device sends `smoothness_log`-style rows)

**Code:** `src/core/device_window_features.py`, `src/core/smoothness_ml_engine.py` (`parse_telematics_event`), `src/mlops/training_pipeline.py`  
**Training:** `uv run python -m src.mlops.training_pipeline` / `tracedata-mlops synthetic`  
**Config:** `mlops_config.yaml`

Data is one **row per ~10 minute window** with columns such as `avg_accel_g`, `total_harsh_brakes`, `avg_jerk`, `avg_rpm`, etc. (full list: `DEVICE_AGGREGATE_FEATURE_COLUMNS` in `device_window_features.py`). Synthetic data uses the same schema; real data should match the **`smoothness_log`** `details` block shape.

**Inference:** `DeviceAggregateTripScorer` — **`score_trip_at_end([envelope, ...])`** for one trip score and aggregated explanation.

**Do not** feed raw pings into this model or device JSON into `SmoothnessInference` without the matching training pipeline and `serving/` contract.

---

## Which should I use?

| You are… | Use |
|-----------|-----|
| Backend stores **raw pings** and batches them into windows | **A) 3-feature** + `SmoothnessInference` |
| Ingestion is **device pre-aggregated** events (or you train on that schema) | **B) 18-feature** + `DeviceAggregateTripScorer` |
| Experimenting with synthetic trip tables only | **B)** unless you are specifically testing Path A |

For deeper product narrative (smoothness vs safety, DB tables), see **[strategy.md](strategy.md)** and **[walkthrough.md](walkthrough.md)**.
