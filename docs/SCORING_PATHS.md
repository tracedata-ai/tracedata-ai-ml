# Two scoring paths (3 vs 18 features)

Use this page when you need to **explain the system** or **choose an API**. The repo ships **two separate models** for **two different kinds of input**. They are not interchangeable: the feature count matches how the data arrives, not an arbitrary choice.

---

## One-sentence summary

- **Pings in the backend** → engineer **3** numbers per 10-minute window → **`SmoothnessInference`** → one trip score from many windows.  
- **Device `smoothness_log` aggregates** (or matching synthetic rows) → **18** numbers per window → **`DeviceAggregateTripScorer`** → one trip score from many windows.

---

## Comparison table

| | **Path A — Pings (production default)** | **Path B — Device aggregates** |
|--|----------------------------------------|--------------------------------|
| **What you pass** | Lists of raw pings: `acceleration_ms2`, `speed_kmh` (optional `timestamp`, GPS, …) | One JSON object per ~10 min: `smoothness_log` / `details.*` (speed, longitudinal, jerk, engine, …) |
| **Feature count** | **3** (`accel_fluidity`, `driving_consistency`, `comfort_zone_percent`) | **18** (e.g. `avg_accel_g`, `total_harsh_brakes`, `avg_jerk`, …) |
| **Where features come from** | `extract_smoothness_features()` in `src/core/features.py` | `parse_telematics_event` + `features_row_from_smoothness_log()` (`src/core/device_window_features.py`) |
| **Training** | `src/mlops/production_window_training.py` / `tracedata-mlops production` | `src/mlops/training_pipeline.py` / `tracedata-mlops synthetic` |
| **Inference** | `SmoothnessInference` — **`score_trip_from_ping_windows(windows)`** (recommended) or legacy `score_window(pings)` for a single bucket | `DeviceAggregateTripScorer` — **`score_trip_at_end(envelopes)`** |
| **Typical product** | Backend collects pings, batches into 10-minute windows, scores end of trip | Device uploads pre-aggregated windows to S3/API; service scores each row then aggregates trip |

---

## How to describe it to others

**Short:**  
“We have two smoothness models: one trained on **raw ping streams** (3 engineered features per window), and one trained on **device-reported 10-minute statistics** (18 features). You use the one that matches your data source.”

**If they only have pings:**  
“Use the **3-feature** model and `SmoothnessInference`; each 10-minute slice is a list of pings, and we return **one** trip score and **one** explanation aggregated across slices.”

**If they only have `smoothness_log` events:**  
“Use the **18-feature** model and `DeviceAggregateTripScorer`; each event is one row, and we return **one** trip score and aggregated attributions at trip end.”

---

## MLflow and `serving/`

Each training run logs its own **`model/`** and **`serving/`** (contract JSON + background rows). The **feature columns in `model_contract.json` must match** the path you use:

- 3-feature contract → load with **`SmoothnessInference`**.
- 18-feature contract → load with **`DeviceAggregateTripScorer`**.

Mixing a 3-feature `serving/` with an 18-feature joblib (or the reverse) will fail or give nonsense.

---

## Sample scripts

- **Pings / trip:** `scripts/score_10min_window.py` — builds or reads `windows` → `score_trip_from_ping_windows`.
- **Device path:** use `DeviceAggregateTripScorer` in Python with envelopes parsed like production ingestion (see `src/inference/device_trip_scorer.py`).

---

## Related docs

| Doc | Role |
|-----|------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Install, train Path A, first prediction |
| [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) | Column meanings for both paths |
| [DATA_FLOW.md](DATA_FLOW.md) | Diagrams for Path A + pointer to Path B |
| [MLOPS_GUIDE.md](MLOPS_GUIDE.md) | MLflow, two training entrypoints |
