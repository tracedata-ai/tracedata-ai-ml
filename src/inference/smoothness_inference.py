"""
Production-style ping path: **one trip in, one score + explanation out**.

1. At deploy: load MLflow artifacts (``model`` + ``serving/``) or local ``.joblib`` + serving dir.
2. At runtime: call ``score_trip_from_ping_windows(windows)`` where each inner list is the
   pings for one ~10-minute bucket. A short trip is a single window: ``[pings]``.

Uses XGBoost ``pred_contribs`` (tree SHAP-style decomposition). Avoids ``shap.TreeExplainer``
on XGBoost 3.x ``base_score`` issues.

Deps: xgboost, numpy, pandas, ``src.core.features.extract_smoothness_features``.
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

    def score_trip_from_ping_windows(
        self,
        windows: List[List[Mapping[str, Any]]],
        *,
        window_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        End-of-trip API: several 10-minute ping windows → one smoothness score + one explanation.

        ``windows[i]``: pings for window *i*; each ping needs ``acceleration_ms2`` and ``speed_kmh``.

        We score each window with the same 3-feature model, then aggregate with weights
        (default: each window weighted by its ping count, minimum 1).

        ``explanation.feature_attributions`` is a weighted average of per-window ``pred_contribs``
        (same sense as ``DeviceAggregateTripScorer`` for device aggregates).
        """
        if not windows:
            raise ValueError("windows must contain at least one ping window")
        if window_weights is not None and len(window_weights) != len(windows):
            raise ValueError("window_weights length must match windows")

        scores: List[float] = []
        weights: List[float] = []
        bases: List[float] = []
        feat_rows: List[np.ndarray] = []

        for i, pings in enumerate(windows):
            if not pings:
                raise ValueError(
                    f"windows[{i}] is empty; omit the window or pass placeholder pings"
                )
            feat = extract_smoothness_features(list(pings))
            x = features_dict_to_frame(feat)[self.feature_columns]
            pred = float(np.clip(self.model.predict(x)[0], 0, 100))
            contribs = self._pred_contribs_row(x)
            bias = float(contribs[-1])
            fc = np.array(
                [float(contribs[j]) for j in range(len(self.feature_columns))],
                dtype=float,
            )
            w = (
                float(window_weights[i])
                if window_weights is not None
                else max(float(len(pings)), 1.0)
            )
            scores.append(pred)
            weights.append(max(w, 1e-6))
            bases.append(bias)
            feat_rows.append(fc)

        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()
        trip_score = float(np.dot(scores, w_arr))
        stack = np.stack(feat_rows)
        trip_vec = stack.T @ w_arr
        attributions = {c: float(trip_vec[j]) for j, c in enumerate(self.feature_columns)}
        trip_base = float(np.dot(bases, w_arr))
        worst_idx = int(np.argmin(scores))

        return {
            "trip_smoothness_score": trip_score,
            "explanation": {
                "feature_attributions": attributions,
                "base_value": trip_base,
                "window_count": len(windows),
                "worst_window_index": worst_idx,
                "worst_window_score": float(scores[worst_idx]),
                "method": "weighted_mean_of_window_pred_contribs",
            },
        }

    def score_window(self, pings: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """
        Legacy single-window shape (smoothness_score + shap keys). Same math as
        ``score_trip_from_ping_windows([pings])``.
        """
        feat = extract_smoothness_features(list(pings))
        trip = self.score_trip_from_ping_windows([pings])
        exp = trip["explanation"]
        return {
            "smoothness_score": trip["trip_smoothness_score"],
            "features": feat,
            "shap": exp["feature_attributions"],
            "shap_base_value": exp["base_value"],
        }
