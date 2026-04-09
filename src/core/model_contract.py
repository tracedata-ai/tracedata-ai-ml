"""
Production model I/O contract: 10-minute ping aggregates -> 3 smoothness features.

The deployment service should:
1. Collect pings for a window (e.g. 10 minutes, same sampling as training).
2. Call extract_smoothness_features(pings) in src.core.features.
3. Pass the feature dict (or row) to the XGBoost model in feature column order.
4. For attributions in production, use XGBoost ``pred_contribs`` (see
   ``src.inference.smoothness_inference``) — same additive breakdown as SHAP for boosted trees.

CONTRACT_VERSION must bump when feature columns or semantics change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0"
MODEL_NAME = "smoothness_10min_xgb"

# Order is part of the contract — training, MLflow signature, and serving must match.
SMOOTHNESS_FEATURE_COLUMNS: List[str] = [
    "accel_fluidity",
    "driving_consistency",
    "comfort_zone_percent",
]


def smoothness_label_from_features(df: pd.DataFrame) -> np.ndarray:
    """
    Synthetic label (same heuristic as src.utils.trainer.generate_labels).
    Replace with real labels when available.
    """
    score = (
        70
        - (df["accel_fluidity"] * 80)
        - (df["driving_consistency"] * 40)
        + (df["comfort_zone_percent"] * 0.4)
    )
    score = score + np.random.normal(0, 2, len(df))
    return np.clip(score, 0, 100)


def features_dict_to_frame(row: Mapping[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame in contract column order."""
    data = {c: [row[c]] for c in SMOOTHNESS_FEATURE_COLUMNS}
    return pd.DataFrame(data)


def frame_to_dict_list(df: pd.DataFrame, columns: Sequence[str] | None = None) -> List[Dict[str, float]]:
    cols = list(columns) if columns is not None else SMOOTHNESS_FEATURE_COLUMNS
    return df[cols].astype(float).to_dict(orient="records")


def build_contract_payload(**extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model_name": MODEL_NAME,
        "feature_columns": list(SMOOTHNESS_FEATURE_COLUMNS),
        "description": (
            "Regression on 3 features from extract_smoothness_features() over a ping window "
            "(e.g. 10 minutes of ~30s samples)."
        ),
    }
    payload.update(extra)
    return payload


def write_contract_json(path: str | Path, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_contract_payload(**extra), f, indent=2)


def read_contract_json(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_training_contract_json(
    path: str | Path,
    feature_columns: Sequence[str],
    *,
    contract_version: str = "custom",
    model_name: str = "smoothness_xgb",
    description: str = "XGBoost smoothness regressor",
    **extra: Any,
) -> None:
    """Serving bundle contract when feature set is not the standard 3-column production contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "contract_version": contract_version,
        "model_name": model_name,
        "feature_columns": list(feature_columns),
        "description": description,
    }
    payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
