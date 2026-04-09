import os
import sqlite3

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.config import (
    DB_NAME,
    MLFLOW_EXPERIMENT_SQLITE,
    MLFLOW_TRACKING_URI,
    SMOOTHNESS_MODEL_PATH,
)
from src.core.model_contract import SMOOTHNESS_FEATURE_COLUMNS
from src.mlops.mlflow_common import log_serving_artifacts, log_xgboost_model, maybe_register_model

SQLITE_BACKGROUND_ROWS = int(os.environ.get("MLFLOW_SQLITE_BACKGROUND_ROWS", "96"))
REGISTER_SQLITE = os.environ.get("MLFLOW_REGISTER_SQLITE_MODEL", "").lower() in ("1", "true", "yes")
SQLITE_REGISTRY_NAME = os.environ.get("MLFLOW_SQLITE_REGISTRY_MODEL_NAME", "")


def generate_labels(df):
    """
    Since we don't have human-labeled data, we synthesize labels based on
    physics-driven heuristics. High Smoothness = Low Fluidity, Low Consistency, High Comfort%.
    """
    score = (
        70
        - (df["accel_fluidity"] * 80)
        - (df["driving_consistency"] * 40)
        + (df["comfort_zone_percent"] * 0.4)
    )
    score += np.random.normal(0, 2, len(df))
    return np.clip(score, 0, 100)


def train_model():
    """
    Train the XGBoost smoothness model from SQLite trips (3 features).

    Logs a full MLflow run: params, metrics, ``model/`` (signature + example),
    and ``serving/`` (contract + background rows) for the same workflow as
    ``production_window_training``.
    """
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        """
        SELECT trip_id, accel_fluidity, driving_consistency, comfort_zone_percent
        FROM trips
        WHERE accel_fluidity IS NOT NULL
        """,
        conn,
    )
    conn.close()

    if len(df) < 10:
        print("❌ Not enough data to train. Run simulator.py first.")
        return None

    df["smoothness_label"] = generate_labels(df)
    X = df[["accel_fluidity", "driving_consistency", "comfort_zone_percent"]]
    y = df["smoothness_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "objective": "reg:squarederror",
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_SQLITE)

    model = None
    run_id = None
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("data_source", "sqlite_trips")
        mlflow.set_tag("pipeline", "sqlite_trainer")

        print(f"🚀 Training XGBoost on {len(X_train)} trips...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("test_mae", float(mae))
        mlflow.log_metric("test_r2", float(r2))
        print(f"✅ MAE: {mae:.2f}, R²: {r2:.4f}")

        MIN_R2 = 0.85
        passed = r2 >= MIN_R2
        mlflow.log_param("quality_gate_min_r2", MIN_R2)
        mlflow.log_param("quality_gate_status", "PASSED" if passed else "FAILED")
        mlflow.set_tag("quality_gate", "PASSED" if passed else "FAILED")

        feat_cols = list(SMOOTHNESS_FEATURE_COLUMNS)
        log_xgboost_model(model, feat_cols, X_train)
        log_serving_artifacts(
            run_id,
            X_train,
            feat_cols,
            background_rows=SQLITE_BACKGROUND_ROWS,
            random_seed=42,
            contract_extra={"source": "telemetry.db", "sqlite_rows": len(df)},
            yaml_snapshots=None,
        )

        maybe_register_model(
            run_id,
            register=passed and REGISTER_SQLITE and bool(SQLITE_REGISTRY_NAME),
            model_name=SQLITE_REGISTRY_NAME or None,
        )

        if not passed:
            print(f"❌ Quality gate FAILED. R²={r2:.2f} below threshold {MIN_R2}.")
            return None

        joblib.dump(model, SMOOTHNESS_MODEL_PATH)
        print(f"💾 Model saved to {SMOOTHNESS_MODEL_PATH}")
        print(f"📊 MLflow run_id={run_id} (experiment={MLFLOW_EXPERIMENT_SQLITE})")

    return model


if __name__ == "__main__":
    train_model()
