import os

# --- PROJECT PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "telemetry.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "smoothness_model.joblib")

# --- DATABASE CONFIG ---
DB_NAME = DB_PATH  # For sqlite3.connect()

# --- MODEL CONFIG ---
SMOOTHNESS_MODEL_PATH = MODEL_PATH

# --- MLflow (override with env in CI / shared server) ---
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLRUNS_DIR)
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
