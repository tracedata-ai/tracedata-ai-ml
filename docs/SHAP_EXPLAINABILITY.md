# Explainability: score breakdown (“SHAP-style”)

> **Start here:** **[GETTING_STARTED.md](GETTING_STARTED.md)** — train once, then run the Step 5 snippet.

Additive explanations answer: *“Which inputs pushed the score up or down, versus a baseline?”*  
For **tree models** (XGBoost), the **`shap` Python library** and **XGBoost `pred_contribs`** both implement related additive decompositions. This repo uses **more than one code path** — pick the one that matches your deployment.

---

## Path 1 — Production inference (recommended for your main app)

**Module:** `src/inference/smoothness_inference.py`  
**Class:** `SmoothnessInference`

- Builds **3 features** from pings via `extract_smoothness_features()`.
- Uses **XGBoost `pred_contribs`** for per-feature contributions (works well with **XGBoost 3.x** and MLflow-loaded models).
- Returns a dict from `score_window(pings)`:
  - `smoothness_score`
  - `features`
  - `shap` — contributions for `accel_fluidity`, `driving_consistency`, `comfort_zone_percent`
  - `shap_base_value` — bias term (last column of `pred_contribs`)

**Load from MLflow:**

```python
from src.inference import SmoothnessInference

inf = SmoothnessInference.from_run("YOUR_RUN_ID", "./mlruns")
```

**Load from disk (e.g. Docker build copied `model.joblib` + `serving/`):**

```python
from pathlib import Path
from src.inference import SmoothnessInference

inf = SmoothnessInference.from_local_paths(
    "models/smoothness_model.joblib",
    Path("path/to/serving"),
)
```

---

## Path 2 — SQLite trip scoring (`TripExplainer`)

**Module:** `src/core/explain.py`  
Uses **`shap.TreeExplainer`** on the XGBoost model and a background sample from the DB. Same **3 features** as production.

**Note:** On some **XGBoost 3.x + shap** combinations, `TreeExplainer` can error on model serialization. If that happens, use **Path 1** (`pred_contribs`) for serving.

---

## Path 3 — `ExplainableScoringEngine` (18-feature engine)

**Module:** `src/core/smoothness_ml_engine.py`  
Used for the **richer synthetic / sample-based** scoring path (different feature set than Path 1). See docstrings there for `score_trip_from_samples_with_explanation`.

---

## How to read contributions (3-feature model)

| Contribution sign | Meaning |
|-------------------|--------|
| Positive | That feature pushed the score **higher** than the baseline for this prediction. |
| Negative | Pushed the score **lower**. |

Approximately:  
`prediction ≈ base_value + sum(per-feature contributions)` (XGBoost `pred_contribs` makes this exact for the boosted tree model up to numerical detail).

**Features:**

| Feature | Plain language |
|---------|----------------|
| `accel_fluidity` | How “choppy” acceleration changes are (jerk). Lower is usually smoother. |
| `driving_consistency` | Spread of acceleration. Lower spread is usually smoother. |
| `comfort_zone_percent` | How often acceleration stays in the comfort band. Higher is usually smoother. |

---

## References

- SHAP paper: [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- XGBoost `pred_contribs`: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
