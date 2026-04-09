# Feature engineering

> New here? Start with **[GETTING_STARTED.md](GETTING_STARTED.md)**.

This repo has **two different feature worlds**. Keep them separate in your head.

---

## A) Production smoothness model (use for real deployment)

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

These three values are the **only** inputs to the production XGBoost model. Column order is fixed — see `src/core/model_contract.py`.

**Safety features (rules, not XGBoost):** `detect_safety_events()` counts harsh brake/accel/speeding thresholds on the same pings.

---

## B) Research pipeline (18 synthetic trip features)

**Code:** `src/utils/data_generation_strategy.py`, `src/mlops/training_pipeline.py`  
**Training:** `uv run python -m src.mlops.training_pipeline`  
**Config:** `mlops_config.yaml`

Synthetic trips are summarized into **18** columns (longitudinal, lateral, speed, jerk, engine aggregates, etc.). Labels come from `generate_synthetic_labels()` in the training pipeline.

**Do not** assume this model matches `ScoringService` or `SmoothnessInference` without changing those modules to compute the same 18 columns from your data.

---

## Which should I use?

| You are… | Use |
|-----------|-----|
| Building the app / Docker / MLflow `serving/` contract | **A) 3-feature** path |
| Experimenting with richer synthetic trips | **B) 18-feature** path |

For deeper product narrative (smoothness vs safety, DB tables), see **[strategy.md](strategy.md)** and **[walkthrough.md](walkthrough.md)**.
