# Documentation index

Read in this order if you are new:

1. **[SCORING_PATHS.md](SCORING_PATHS.md)** — **Two models:** pings → **3 features** vs device aggregates → **18 features**. Read this so you pick the right API and training command.
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** — Install, run tests, train the default (ping) model, first prediction. **Do this next.**

Then pick what you need:

| Document | What it is for |
|----------|----------------|
| [MLOPS_GUIDE.md](MLOPS_GUIDE.md) | MLflow, two training pipelines, metrics |
| [AGENTIC_SCORING_DEPLOYMENT.md](AGENTIC_SCORING_DEPLOYMENT.md) | Release-to-runtime handoff for deploying model bundles in agentic scoring systems |
| [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) | Column meanings: **3** ping features vs **18** device aggregates |
| [DATA_FLOW.md](DATA_FLOW.md) | Diagrams: pings → score; pointer to device path |
| [SHAP_EXPLAINABILITY.md](SHAP_EXPLAINABILITY.md) | Why a score changed (contributions / “SHAP-style” breakdown) |
| [strategy.md](strategy.md) | Long beginner-friendly concepts: smoothness vs safety, steps in plain language |
| [walkthrough.md](walkthrough.md) | Architecture (DB, agents, XAI) in one pass |
| [ml_pipeline_guide.md](ml_pipeline_guide.md) | ML ideas explained using API/database analogies |

If anything disagrees with the code, trust the code and file an issue. **Default product path (pings):** `production_window_training` + `SmoothnessInference.score_trip_from_ping_windows`. **Device aggregates:** `training_pipeline` + `DeviceAggregateTripScorer` — see [SCORING_PATHS.md](SCORING_PATHS.md).
