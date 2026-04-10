"""
One-off: several production training runs with different XGBoost / data settings
for MLflow comparison. From repo root:

  uv run python tmp/run_production_tune_passes.py
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.mlops.production_window_training import ProductionWindowTrainingPipeline


def _deep_merge_inplace(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_inplace(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "production_mlops.yaml"
    with open(base_path, encoding="utf-8") as f:
        base_cfg: Dict[str, Any] = yaml.safe_load(f)

    # Distinct params/metrics for charts; keep quality gate reachable for synthetic data.
    variants: List[Dict[str, Any]] = [
        {
            "training": {
                "n_samples": 2200,
                "random_seed": 11,
                "model_output_path": "models/smoothness_tune_pass01.joblib",
            },
            "model": {
                "xgboost": {
                    "n_estimators": 80,
                    "learning_rate": 0.12,
                    "max_depth": 4,
                    "random_state": 11,
                }
            },
        },
        {
            "training": {
                "n_samples": 2800,
                "random_seed": 23,
                "model_output_path": "models/smoothness_tune_pass02.joblib",
            },
            "model": {
                "xgboost": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 7,
                    "random_state": 23,
                }
            },
        },
        {
            "training": {
                "n_samples": 2500,
                "random_seed": 42,
                "model_output_path": "models/smoothness_tune_pass03.joblib",
            },
            "model": {
                "xgboost": {
                    "n_estimators": 300,
                    "learning_rate": 0.06,
                    "max_depth": 6,
                    "random_state": 42,
                }
            },
        },
        {
            "training": {
                "n_samples": 1800,
                "random_seed": 77,
                "model_output_path": "models/smoothness_tune_pass04.joblib",
            },
            "model": {
                "xgboost": {
                    "n_estimators": 150,
                    "learning_rate": 0.15,
                    "max_depth": 3,
                    "random_state": 77,
                }
            },
        },
        {
            "training": {
                "n_samples": 3200,
                "random_seed": 99,
                "test_size": 0.25,
                "model_output_path": "models/smoothness_tune_pass05.joblib",
            },
            "model": {
                "xgboost": {
                    "n_estimators": 100,
                    "learning_rate": 0.07,
                    "max_depth": 8,
                    "random_state": 99,
                }
            },
        },
    ]

    for i, overlay in enumerate(variants, start=1):
        cfg = copy.deepcopy(base_cfg)
        _deep_merge_inplace(cfg, overlay)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as tf:
            yaml.safe_dump(cfg, tf, default_flow_style=False, sort_keys=False)
            tmp_path = tf.name
        try:
            print(f"\n=== Pass {i}/{len(variants)} ===")
            result = ProductionWindowTrainingPipeline(tmp_path).run()
            print(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print("\nAll passes finished. Open MLflow UI on experiment smoothness-10min-production.")


if __name__ == "__main__":
    main()
