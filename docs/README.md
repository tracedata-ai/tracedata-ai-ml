# Documentation index

Read in this order if you are new:

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** — Install, run tests, train a model, run your first prediction. **Do this first.**

Then pick what you need:

| Document | What it is for |
|----------|----------------|
| [MLOPS_GUIDE.md](MLOPS_GUIDE.md) | MLflow, two training pipelines, metrics, registry |
| [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) | Production **3** features vs research **18** features |
| [DATA_FLOW.md](DATA_FLOW.md) | Short picture: pings → features → model → score |
| [SHAP_EXPLAINABILITY.md](SHAP_EXPLAINABILITY.md) | Why a score changed (contributions / “SHAP-style” breakdown) |
| [strategy.md](strategy.md) | Long beginner-friendly concepts: smoothness vs safety, steps in plain language |
| [walkthrough.md](walkthrough.md) | Architecture (DB, agents, XAI) in one pass |
| [ml_pipeline_guide.md](ml_pipeline_guide.md) | ML ideas explained using API/database analogies |

If anything disagrees with the code, trust the code and file an issue—the **production** path is `src.mlops.production_window_training` and `src.inference.SmoothnessInference`.
