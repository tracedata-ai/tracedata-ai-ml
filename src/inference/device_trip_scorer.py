"""
Device ``smoothness_log`` path: score each 10-minute window with the 18-feature XGBoost model,
then produce one end-of-trip outcome (weighted score + aggregated attributions).

Trip-level ``trip_shap`` is a weighted average of per-window ``pred_contribs`` (excluding bias);
it explains the trip aggregate in the same sense as window-level tree SHAP, not an exact SHAP
vector for a hypothetical single-row trip model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.device_window_features import (
    DEVICE_AGGREGATE_FEATURE_COLUMNS,
    features_row_from_smoothness_log,
    window_weight_seconds,
)


class DeviceAggregateTripScorer:
    """
    18-feature aggregate model (synthetic / device training pipeline), not the 3 ping features.
    """

    def __init__(
        self,
        model: Any,
        background: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_columns = list(feature_columns or DEVICE_AGGREGATE_FEATURE_COLUMNS)
        if self.feature_columns != DEVICE_AGGREGATE_FEATURE_COLUMNS:
            raise ValueError(
                "DeviceAggregateTripScorer expects the 18 device aggregate columns in training "
                f"order; got {self.feature_columns!r}. For ping-based scoring use "
                "SmoothnessInference."
            )
        _ = background[self.feature_columns].astype(float)

    @classmethod
    def from_run(cls, run_id: str, tracking_uri: str) -> DeviceAggregateTripScorer:
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
    ) -> DeviceAggregateTripScorer:
        import mlflow

        model = mlflow.xgboost.load_model(model_uri)
        return cls._from_serving_dir(model, Path(serving_dir))

    @classmethod
    def from_local_paths(
        cls,
        model_path: Union[str, Path],
        serving_dir: Union[str, Path],
    ) -> DeviceAggregateTripScorer:
        import joblib

        model = joblib.load(model_path)
        return cls._from_serving_dir(model, Path(serving_dir))

    @staticmethod
    def _from_serving_dir(model: Any, root: Path) -> DeviceAggregateTripScorer:
        with open(root / "model_contract.json", encoding="utf-8") as f:
            contract = json.load(f)
        with open(root / "background_features.json", encoding="utf-8") as f:
            bg_rows = json.load(f)
        cols = contract["feature_columns"]
        bg = pd.DataFrame(bg_rows)
        return DeviceAggregateTripScorer(model, bg, feature_columns=cols)

    def _row_frame(self, row: Mapping[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{c: float(row[c]) for c in self.feature_columns}])

    def _pred_contribs_row(self, x: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(x.values, feature_names=self.feature_columns)
        mat = self.model.get_booster().predict(dm, pred_contribs=True)
        row = np.asarray(mat, dtype=float)
        if row.ndim == 2:
            row = row[0]
        return row

    def score_window_from_envelope(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        row = features_row_from_smoothness_log(envelope)
        x = self._row_frame(row)
        pred = float(np.clip(self.model.predict(x)[0], 0, 100))
        contribs = self._pred_contribs_row(x)
        bias = float(contribs[-1])
        shap = {c: float(contribs[i]) for i, c in enumerate(self.feature_columns)}
        return {
            "smoothness_score": pred,
            "shap": shap,
            "shap_base_value": bias,
            "features": row,
            "window_weight": window_weight_seconds(envelope),
        }

    def score_trip_at_end(
        self,
        envelopes: List[Mapping[str, Any]],
        *,
        include_per_window: bool = False,
    ) -> Dict[str, Any]:
        if not envelopes:
            raise ValueError("at least one window envelope is required")

        scores: List[float] = []
        weights: List[float] = []
        bases: List[float] = []
        feat_matrix: List[np.ndarray] = []
        per_window: List[Dict[str, Any]] = []

        for env in envelopes:
            row = features_row_from_smoothness_log(env)
            x = self._row_frame(row)
            pred = float(np.clip(self.model.predict(x)[0], 0, 100))
            contribs = self._pred_contribs_row(x)
            bias = float(contribs[-1])
            fc = np.array(
                [float(contribs[i]) for i in range(len(self.feature_columns))],
                dtype=float,
            )
            w = window_weight_seconds(env)
            scores.append(pred)
            weights.append(w)
            bases.append(bias)
            feat_matrix.append(fc)
            if include_per_window:
                per_window.append(
                    {
                        "smoothness_score": pred,
                        "shap": {c: float(fc[i]) for i, c in enumerate(self.feature_columns)},
                        "shap_base_value": bias,
                        "features": row,
                        "window_weight": w,
                    }
                )

        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()
        trip_score = float(np.dot(scores, w_arr))
        stack = np.stack(feat_matrix)
        trip_vec = stack.T @ w_arr
        trip_shap = {c: float(trip_vec[i]) for i, c in enumerate(self.feature_columns)}
        trip_base = float(np.dot(bases, w_arr))
        worst_idx = int(np.argmin(scores))

        out: Dict[str, Any] = {
            "trip_smoothness_score": trip_score,
            "trip_shap": trip_shap,
            "trip_shap_base_value": trip_base,
            "window_count": len(envelopes),
            "worst_window_index": worst_idx,
            "worst_window_score": float(scores[worst_idx]),
        }
        if include_per_window:
            out["windows"] = per_window
        return out
