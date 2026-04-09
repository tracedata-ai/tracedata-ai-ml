# Getting started (step by step)

Follow these steps **in order** from the **project root** (`tracedata-ai-ml/` — the folder that contains `pyproject.toml`).

Commands work in **PowerShell**, **cmd**, or **bash**. Use `Ctrl+C` if you need to stop a running server.

---

## Step 0 — Prerequisites

1. **Python 3.11+** installed.
2. **uv** (fast Python package manager). Install: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
3. This repo cloned or unzipped on your machine.

Open a terminal and go to the project root:

```bash
cd path/to/tracedata-ai-ml
```

---

## Step 1 — Install dependencies

This creates/uses `.venv` and installs packages from `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

**Check:** no errors at the end.

---

## Step 2 — Run the automated tests

```bash
uv run pytest tests/ -q
```

**What you should see:** `13 passed` and **`1 skipped`**.  
The skipped test is the **integration** API test (it needs a server on port 8000). That is expected.

---

## Step 3 — Train the production smoothness model (recommended path)

This pipeline matches how scoring works in code: **10-minute style windows** of pings → **3 features** → XGBoost. It logs to **MLflow** and saves a joblib model.

```bash
uv run python -m src.mlops.production_window_training
```

Or the same pipeline via the CLI:

```bash
uv run tracedata-mlops production
```

**Wait** until it finishes (it may take a minute).

**What to look for in the log:**

- `Done. run_id=...` — copy this **run ID**; you will use it in Step 5.
- `Saved model to ...\models\smoothness_model.joblib` — appears if the quality gate passed.

Config lives in **`production_mlops.yaml`** (experiment name, sample counts, MLflow URI, etc.).

---

## Step 4 (optional) — Open the MLflow UI

In a **second** terminal, from the same project root:

```bash
uv run mlflow ui
```

Open a browser at **http://127.0.0.1:5000** (or the URL printed in the terminal).  
You should see experiment **smoothness-10min-production** and your run.  
Use **Ctrl+C** in that terminal to stop the UI.

---

## Step 5 — Run one prediction in Python

Replace `YOUR_RUN_ID` with the `run_id` from Step 3.

```bash
uv run python -c "
from src.inference import SmoothnessInference
from src.utils.simulator import generate_telemetry

run_id = 'YOUR_RUN_ID'
inf = SmoothnessInference.from_run(run_id, './mlruns')
pings = generate_telemetry('smooth', duration_minutes=10)
print(inf.score_window(pings))
"
```

**What you get:** a dict with `smoothness_score`, `features`, `shap` (per-feature contributions), and `shap_base_value`.

---

## Step 6 (optional) — SQLite playground (trips in a local database)

This is the **older** path: fill SQLite, extract features, train with `trainer.py`. Useful to understand `scoring.py` and the demo DB.

**6a — Create DB and fake trips**

```bash
uv run python -c "from src.utils.simulator import init_db, simulate_data; init_db(); simulate_data(num_drivers=3, trips_per_driver=5)"
```

**6b — Compute features from telemetry into the `trips` table**

```bash
uv run python -m src.utils.processor
```

**6c — Train and save `models/smoothness_model.joblib` (uses trips in DB)**

```bash
uv run python -m src.utils.trainer
```

If you see “Not enough data”, run **6a** again with more drivers/trips, then **6b** and **6c**.

---

## Step 7 — Where to go next

- **Deploy / another repo:** see root `README.md` and `serving/` artifacts logged by Step 3 (`model_contract.json`, `background_features.json`).
- **Second training mode (18 synthetic features):** `uv run python -m src.mlops.training_pipeline` and `mlops_config.yaml` — explained in [MLOPS_GUIDE.md](MLOPS_GUIDE.md).
- **Concepts (long read):** [strategy.md](strategy.md).

---

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| `uv: command not found` | Install uv (Step 0) and reopen the terminal. |
| `ModuleNotFoundError: src` | Run commands from the **project root**, not inside `src/`. |
| `from_run` cannot find run | Use the exact `run_id` from Step 3; ensure `production_mlops.yaml` has `tracking_uri: ./mlruns` and that folder exists after training. |
| pytest fails on API test | Run `uv run pytest tests/ -q` — only the integration test needs a live API; skipped is OK. |
