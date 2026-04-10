"""
Train the production smoothness model (3 features from 10-minute ping windows),
log to MLflow with a model signature + serving artifacts for downstream Docker builds.

This aligns with deployment: aggregate pings -> extract_smoothness_features -> predict + SHAP.

Run: python -m src.mlops.production_window_training
Config: production_mlops.yaml (repo root)
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.entities import SpanType
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.features import extract_smoothness_features
from src.core.model_contract import (
    CONTRACT_VERSION,
    SMOOTHNESS_FEATURE_COLUMNS,
    smoothness_label_from_features,
)
from src.mlops.mlflow_common import log_serving_artifacts, log_xgboost_model, maybe_register_model
from src.mlops.mlflow_settings import ensure_mlflow_experiment, resolve_tracking_uri
from src.utils.simulator import generate_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STYLES = ("smooth", "jerky", "unsafe")
STYLE_WEIGHTS = (0.42, 0.38, 0.20)


@contextlib.contextmanager
def _maybe_span(
    name: str,
    enabled: bool,
    *,
    span_type: str = SpanType.UNKNOWN,
) -> Generator[Any, None, None]:
    """MLflow trace span, or a no-op when tracing is disabled in config."""
    if not enabled:
        yield None
        return
    with mlflow.start_span(name, span_type=span_type) as span:
        yield span


def _build_synthetic_frame(
    n_samples: int,
    window_minutes: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, float]] = []
    for i in range(n_samples):
        style = rng.choice(STYLES, p=STYLE_WEIGHTS).item()
        row_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(row_seed)
        random.seed(row_seed)
        points = generate_telemetry(style=style, duration_minutes=window_minutes)
        feat = extract_smoothness_features(points)
        rows.append(feat)
    df = pd.DataFrame(rows)
    np.random.seed(random_seed)
    df["smoothness_label"] = smoothness_label_from_features(df)
    return df


class ProductionWindowTrainingPipeline:
    def __init__(self, config_path: str = "production_mlops.yaml"):
        root = Path(__file__).resolve().parents[2]
        cfg_file = root / config_path if not os.path.isabs(config_path) else Path(config_path)
        with open(cfg_file, encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.mlflow_cfg = self.config["mlflow"]
        self.train_cfg = self.config["training"]
        self.model_cfg = self.config["model"]
        self.registry_cfg = self.config.get("registry", {})
        self.output_cfg = self.config["output"]

    def run(self) -> Dict[str, Any]:
        uri = resolve_tracking_uri(self.mlflow_cfg.get("tracking_uri"))
        experiment = os.environ.get(
            "MLFLOW_EXPERIMENT_PRODUCTION", self.mlflow_cfg["experiment_name"]
        )
        mlflow.set_tracking_uri(uri)
        ensure_mlflow_experiment(experiment)

        trace_enabled = bool(self.mlflow_cfg.get("enable_tracing", True))

        seed = int(self.train_cfg["random_seed"])
        n_samples = int(self.train_cfg["n_samples"])
        window_minutes = int(self.train_cfg["window_minutes"])
        test_size = float(self.train_cfg["test_size"])
        bg_n = min(int(self.train_cfg["background_rows"]), n_samples)

        reports = Path(self.output_cfg["reports_dir"])
        reports.mkdir(parents=True, exist_ok=True)
        model_rel = Path(self.train_cfg["model_output_path"])
        model_path = Path(__file__).resolve().parents[2] / model_rel
        model_path.parent.mkdir(parents=True, exist_ok=True)

        run_id: Optional[str] = None
        metrics: Dict[str, float] = {}
        passed = False

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            with _maybe_span(
                "smoothness_production_train",
                trace_enabled,
                span_type=SpanType.CHAIN,
            ) as root_span:
                if root_span is not None:
                    root_span.set_inputs(
                        {
                            "n_samples": n_samples,
                            "window_minutes": window_minutes,
                            "test_size": test_size,
                            "random_seed": seed,
                        }
                    )

                with _maybe_span("synthetic_ping_windows", trace_enabled):
                    logger.info(
                        "Building synthetic %s-min windows (n=%s)...",
                        window_minutes,
                        n_samples,
                    )
                    df = _build_synthetic_frame(n_samples, window_minutes, seed)

                with _maybe_span("train_test_split", trace_enabled):
                    X = df[SMOOTHNESS_FEATURE_COLUMNS]
                    y = df["smoothness_label"]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=seed
                    )

                params = dict(self.model_cfg["xgboost"])
                with _maybe_span("xgboost_fit", trace_enabled):
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train)

                with _maybe_span("evaluate_test_split", trace_enabled):
                    pred_test = model.predict(X_test)
                    metrics = {
                        "test_mae": float(mean_absolute_error(y_test, pred_test)),
                        "test_r2": float(r2_score(y_test, pred_test)),
                    }

                gate = self.model_cfg["quality_gate"]["min_r2_score"]
                passed = metrics["test_r2"] >= gate

                if root_span is not None:
                    root_span.set_outputs(
                        {
                            "test_r2": metrics["test_r2"],
                            "test_mae": metrics["test_mae"],
                            "quality_gate_passed": passed,
                        }
                    )

                mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("window_minutes", window_minutes)
                mlflow.log_param("contract_version", CONTRACT_VERSION)
                mlflow.log_param("quality_gate_min_r2", gate)
                mlflow.log_param("quality_gate_status", "PASSED" if passed else "FAILED")
                mlflow.log_param("tracing_enabled", trace_enabled)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                log_xgboost_model(model, SMOOTHNESS_FEATURE_COLUMNS, X_train)
                root = Path(__file__).resolve().parents[2]
                snaps = [root / "production_mlops.yaml"]
                log_serving_artifacts(
                    run_id,
                    X_train,
                    SMOOTHNESS_FEATURE_COLUMNS,
                    background_rows=bg_n,
                    random_seed=seed,
                    contract_extra={
                        "training_rows": n_samples,
                        "window_minutes": window_minutes,
                    },
                    yaml_snapshots=[p for p in snaps if p.is_file()],
                )

                if passed:
                    joblib.dump(model, model_path)
                    logger.info("Saved model to %s", model_path)
                else:
                    logger.warning("Quality gate failed; model not written to %s", model_path)

                maybe_register_model(
                    run_id,
                    register=bool(passed and self.registry_cfg.get("register")),
                    model_name=(self.registry_cfg.get("model_name") or None),
                )

        assert run_id is not None
        logger.info("Done. run_id=%s metrics=%s", run_id, metrics)
        return {"run_id": run_id, "metrics": metrics, "quality_gate_passed": passed}


def main() -> None:
    ProductionWindowTrainingPipeline().run()


if __name__ == "__main__":
    main()
