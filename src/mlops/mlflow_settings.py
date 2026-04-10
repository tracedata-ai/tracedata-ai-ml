"""
MLflow layout: dedicated project folder (SQLite backend + artifact root).

Default (no env overrides):
  <project>/mlflow/mlflow.db     — tracking backend
  <project>/mlflow/artifacts/   — run artifacts (models, serving/, etc.)

Override:
  MLFLOW_TRACKING_URI       — full URI (e.g. sqlite:///..., file:..., http://...)
  MLFLOW_ROOT               — folder for default sqlite DB (default: <project>/mlflow)
  MLFLOW_ARTIFACT_ROOT      — artifact directory URI target (default: <MLFLOW_ROOT>/artifacts)
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_ROOT = Path(os.environ.get("MLFLOW_ROOT", _PROJECT_ROOT / "mlflow"))

# YAML / code uses this sentinel to mean “use defaults in this module”
_TRACKING_AUTO = frozenset({"auto", "__default__", ""})


def _sqlite_uri_for_db(db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db_path.resolve().as_posix()


def resolve_tracking_uri(yaml_tracking_uri: str | None) -> str:
    """
    Resolve MLflow tracking URI.

    Priority: MLFLOW_TRACKING_URI env > non-auto YAML value > SQLite under MLFLOW_ROOT.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if yaml_tracking_uri and yaml_tracking_uri.strip().lower() not in _TRACKING_AUTO:
        return yaml_tracking_uri
    return _sqlite_uri_for_db(MLFLOW_ROOT / "mlflow.db")


def artifact_root_path() -> Path:
    return Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", MLFLOW_ROOT / "artifacts"))


def ensure_mlflow_experiment(experiment_name: str) -> None:
    """
    Set active experiment, creating it if needed with a file: artifact root under
    MLFLOW_ARTIFACT_ROOT (so artifacts stay inside the dedicated mlflow folder).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    root = artifact_root_path()
    root.mkdir(parents=True, exist_ok=True)
    artifact_uri = root.resolve().as_uri()

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)
