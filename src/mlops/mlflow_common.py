"""
Shared MLflow logging for XGBoost smoothness models (signatures, serving/ artifacts, registry).

Used by production window training, SQLite `trainer`, and optional `smoothness_ml_engine` training.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from src.core.model_contract import (
    SMOOTHNESS_FEATURE_COLUMNS,
    frame_to_dict_list,
    write_contract_json,
    write_training_contract_json,
)

logger = logging.getLogger(__name__)


def regression_signature(
    feature_columns: List[str], output_name: str = "smoothness_score"
) -> ModelSignature:
    inputs = Schema([ColSpec("double", c) for c in feature_columns])
    outputs = Schema([ColSpec("double", output_name)])
    return ModelSignature(inputs=inputs, outputs=outputs)


def log_xgboost_model(
    model: Any,
    feature_columns: List[str],
    input_example: pd.DataFrame,
) -> None:
    cols_df = input_example[feature_columns].head(3)
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=regression_signature(feature_columns),
        input_example=cols_df,
    )


def log_serving_artifacts(
    run_id: str,
    X_train: pd.DataFrame,
    feature_columns: List[str],
    *,
    background_rows: int,
    random_seed: int,
    contract_extra: Optional[Dict[str, Any]] = None,
    yaml_snapshots: Optional[List[Path]] = None,
) -> None:
    """Upload serving/model_contract.json, background_features.json, optional config snapshots."""
    extra = dict(contract_extra or {})
    bg_n = min(max(1, background_rows), len(X_train))
    bg = X_train[feature_columns].sample(n=bg_n, random_state=random_seed)

    with tempfile.TemporaryDirectory() as tmp:
        serving = Path(tmp) / "serving"
        serving.mkdir()
        contract_path = serving / "model_contract.json"

        if list(feature_columns) == list(SMOOTHNESS_FEATURE_COLUMNS):
            write_contract_json(contract_path, mlflow_run_id=run_id, **extra)
        else:
            write_training_contract_json(
                contract_path,
                feature_columns,
                mlflow_run_id=run_id,
                **extra,
            )

        with open(serving / "background_features.json", "w", encoding="utf-8") as f:
            json.dump(frame_to_dict_list(bg, columns=feature_columns), f, indent=2)

        if yaml_snapshots:
            for p in yaml_snapshots:
                if p.is_file():
                    shutil.copy(p, serving / f"config_snapshot_{p.name}")

        mlflow.log_artifacts(str(serving), artifact_path="serving")


def maybe_register_model(
    run_id: str,
    *,
    register: bool,
    model_name: Optional[str],
) -> None:
    if register and model_name:
        res = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        logger.info("Registered model %s version %s", model_name, res.version)
