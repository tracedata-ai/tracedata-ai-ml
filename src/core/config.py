import os

# --- PROJECT PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "telemetry.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "smoothness_model.joblib")

# --- DATABASE CONFIG ---
DB_NAME = DB_PATH  # For sqlite3.connect()

# --- MODEL CONFIG ---
SMOOTHNESS_MODEL_PATH = MODEL_PATH

# --- MLflow (dedicated folder: see src/mlops/mlflow_settings.py) ---
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")  # legacy file-store path only

from src.mlops.mlflow_settings import (  # noqa: E402
    MLFLOW_ROOT,
    ensure_mlflow_experiment,
    resolve_tracking_uri,
)

MLFLOW_TRACKING_URI = resolve_tracking_uri(None)
# Experiments by training entrypoint
MLFLOW_EXPERIMENT_PRODUCTION = os.environ.get(
    "MLFLOW_EXPERIMENT_PRODUCTION", "smoothness-10min-production"
)
MLFLOW_EXPERIMENT_SYNTHETIC_18 = os.environ.get(
    "MLFLOW_EXPERIMENT_SYNTHETIC_18", "smoothness-scoring-synthetic-18f"
)
MLFLOW_EXPERIMENT_SQLITE = os.environ.get(
    "MLFLOW_EXPERIMENT_SQLITE", "smoothness-sqlite-trips"
)
MLFLOW_EXPERIMENT_ML_ENGINE = os.environ.get(
    "MLFLOW_EXPERIMENT_ML_ENGINE", "smoothness-ml-engine-sqlite"
)
