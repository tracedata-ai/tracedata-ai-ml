"""
Reference implementation for the main application repo:

1. At Docker build: download MLflow artifacts (`model/` + `serving/`).
2. At runtime: load the XGBoost model, call ``score_window(pings)``.

Uses XGBoost ``pred_contribs`` for feature attributions (SHAP-compatible decomposition for
trees). Avoids ``shap.TreeExplainer``, which can fail on XGBoost 3.x base_score serialization.

Serving image deps: xgboost, numpy, pandas (and feature extraction — here
``src.core.features.extract_smoothness_features``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.features import extract_smoothness_features
from src.core.model_contract import SMOOTHNESS_FEATURE_COLUMNS, features_dict_to_frame


class SmoothnessInference:
    """
    XGBoost regressor + per-prediction feature contributions via ``pred_contribs``.
    ``background_features.json`` is still required: it validates the feature contract.
    """

    def __init__(
        self,
        model: Any,
        background: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_columns = list(feature_columns or SMOOTHNESS_FEATURE_COLUMNS)
        if list(self.feature_columns) != list(SMOOTHNESS_FEATURE_COLUMNS):
            raise ValueError(
                f"feature_columns must match contract order {SMOOTHNESS_FEATURE_COLUMNS!r}"
            )
        # Validate artifact schema; contributions do not use a background matrix.
        _ = background[self.feature_columns].astype(float)

    @classmethod
    def from_run(cls, run_id: str, tracking_uri: str) -> SmoothnessInference:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        client = mlflow.tracking.MlflowClient()
        local = client.download_artifacts(run_id, "serving")
        return cls._from_serving_dir(model, Path(local))

    @classmethod
    def from_mlflow_model_uri(
        cls,
        model_uri: str,
        serving_dir: Union[str, Path],
    ) -> SmoothnessInference:
        import mlflow

        model = mlflow.xgboost.load_model(model_uri)
        return cls._from_serving_dir(model, Path(serving_dir))

    @classmethod
    def from_local_paths(
        cls,
        model_path: Union[str, Path],
        serving_dir: Union[str, Path],
    ) -> SmoothnessInference:
        import joblib

        model = joblib.load(model_path)
        return cls._from_serving_dir(model, Path(serving_dir))

    @staticmethod
    def _from_serving_dir(model: Any, root: Path) -> SmoothnessInference:
        with open(root / "model_contract.json", encoding="utf-8") as f:
            contract = json.load(f)
        with open(root / "background_features.json", encoding="utf-8") as f:
            bg_rows = json.load(f)
        cols = contract["feature_columns"]
        bg = pd.DataFrame(bg_rows)
        return SmoothnessInference(model, bg, feature_columns=cols)

    def _pred_contribs_row(self, x: pd.DataFrame) -> np.ndarray:
        """Shape (n_features + 1,): per-feature contribution + bias (last)."""
        dm = xgb.DMatrix(x.values, feature_names=self.feature_columns)
        mat = self.model.get_booster().predict(dm, pred_contribs=True)
        row = np.asarray(mat, dtype=float)
        if row.ndim == 2:
            row = row[0]
        return row

    def predict_from_features(self, features: Mapping[str, float]) -> float:
        x = features_dict_to_frame(features)[self.feature_columns]
        pred = self.model.predict(x)[0]
        return float(np.clip(pred, 0, 100))

    def explain_features(self, features: Mapping[str, float]) -> Dict[str, Any]:
        x = features_dict_to_frame(features)[self.feature_columns]
        contribs = self._pred_contribs_row(x)
        bias = float(contribs[-1])
        out: Dict[str, Any] = {"base_value": bias}
        for i, c in enumerate(self.feature_columns):
            out[c] = float(contribs[i])
        return out

    def score_window(self, pings: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """
        ``pings``: dicts with at least ``acceleration_ms2`` and ``speed_kmh``
        (same as ``extract_smoothness_features``).
        """
        feat = extract_smoothness_features(list(pings))
        score = self.predict_from_features(feat)
        breakdown = self.explain_features(feat)
        shap_breakdown = {k: v for k, v in breakdown.items() if k != "base_value"}
        return {
            "smoothness_score": score,
            "features": feat,
            "shap": shap_breakdown,
            "shap_base_value": breakdown["base_value"],
        }
