# MLflow parallel coordinates plot — what it means and how to pick a “better” run

This note explains the **Parallel coordinates** chart in the MLflow UI when you compare multiple training runs (for example several passes of the production smoothness model with different settings).

---

## What you are looking at

- **Each colored polyline (path across vertical axes)** is **one MLflow run** — one full training job with its own logged parameters and metrics.
- **Each vertical axis** is one quantity you chose in the chart builder:
  - Axes on the **left** are usually **parameters** (things you set before or during training, such as `n_samples`, `xgb_learning_rate`, `xgb_max_depth`).
  - Axes on the **right** are usually **metrics** (numbers computed after training on a held-out split, such as `test_mae`, `test_r2`).
- A line **connects one run’s values** across all selected axes. So you can see, at a glance, which combinations of hyperparameters tend to land where on the test metrics.

The chart does **not** train models; it only **visualizes runs you already logged**.

---

## How to read the metrics on the right

For this project’s regression model:

| Metric | Better direction | Plain language |
|--------|------------------|----------------|
| **`test_r2`** | **Higher** | How much of the variance in the test labels the model explains (1.0 = perfect on that split; lower = more error). |
| **`test_mae`** | **Lower** | Average absolute error between predicted and true smoothness on the **test** split. |

So a run is **better** on the test split if **`test_r2` is higher** and **`test_mae` is lower** (all else equal).

---

## What the color scale usually means

MLflow often colors lines by **one metric** (commonly the same one you care most about optimizing). In your setup, the color bar is tied to **`test_r2`**:

- **Warmer / red** → **higher** `test_r2` (better fit on the test split).
- **Cooler / blue** → **lower** `test_r2` (worse fit on the test split).

Use the color bar labels to map exact values. The **best run in the chart** is the one whose line sits at the **top** of the `test_r2` axis and **bottom** of the `test_mae` axis, and typically appears **red** if color = `test_r2`.

---

## Example interpretation (your comparison runs)

When comparing runs that differ in **`n_samples`**, **`xgb_learning_rate`**, and **`xgb_max_depth`**, the chart often shows patterns like:

- The **strongest run** (highest `test_r2`, lowest `test_mae`) may use **more training rows** (`n_samples` toward the high end of what you tried), **lower learning rate** (smaller steps per tree boost), and **deeper trees** (`xgb_max_depth` toward the high end), *for the specific grid of runs you logged*.

- The **weakest run** among those lines may sit at the opposite corner: **fewer samples**, **higher learning rate**, **shallower trees**, with **lower `test_r2`** and **higher `test_mae`**.

That pattern is **useful for intuition and for choosing the next experiment**, but it is **not a guarantee** that those settings are optimal on real fleet data or on a different random seed / data mix.

---

## Which option is “better,” and why?

**Operationally:** the **better option is the run whose test metrics match your goals**, usually **maximize `test_r2` and minimize `test_mae`** on the **same** test protocol (same `test_size`, same feature pipeline, same data generation assumptions).

**Why that run can win:** with more synthetic samples, the model sees more diversity; lower learning rate can reduce noisy updates; deeper trees can capture more nonlinearity — **on this synthetic labeling setup**. Together they can improve the hold-out scores **for these runs only**.

**Important caveats:**

1. **Synthetic data** — Scores reflect the simulator and label definition (`smoothness_label`), not real roads or sensors. A winner here is a winner **in the lab chart**, not automatically in production.
2. **Correlation, not proof** — Parallel coordinates show **what happened across a few runs**. They do not prove causality; another run with different `n_estimators` or seed could change the story.
3. **Overfitting** — Very flexible settings (deep trees, many rounds) can look great on a single test split but generalize worse on new domains. Use multiple seeds, more data, or cross-validation when decisions matter.
4. **Narrow metric scope** — `test_mae` / `test_r2` are only two views. Latency, stability, fairness, and drift are not on this chart.

---

## Practical next steps

- In MLflow, open the **best** run (by `test_r2` / `test_mae`), note its **Parameters**, and align `production_mlops.yaml` if you want that configuration as the default training recipe.
- Re-run training with **new random seeds** or **slightly different** hyperparameters to see if the ranking stays stable.
- For demos, keep using the parallel coordinates view to explain **trade-offs** between sample size, learning rate, and tree depth at a glance.

---

## Related

- Training entrypoints and MLflow storage: [GETTING_STARTED.md](GETTING_STARTED.md) (Steps 3–4).
- Production config: `production_mlops.yaml` at the repo root.
