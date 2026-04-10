"""MLflow settings (simple default: local file store at ./mlruns)."""

from __future__ import annotations

import os

# YAML / code uses this sentinel to mean “use defaults in this module”
_TRACKING_AUTO = frozenset({"auto", "__default__", ""})


def resolve_tracking_uri(yaml_tracking_uri: str | None) -> str:
    """
    Resolve MLflow tracking URI.

    Priority: MLFLOW_TRACKING_URI env > non-auto YAML value > ./mlruns.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if yaml_tracking_uri and yaml_tracking_uri.strip().lower() not in _TRACKING_AUTO:
        return yaml_tracking_uri
    return "./mlruns"


def ensure_mlflow_experiment(experiment_name: str) -> None:
    """Set active experiment, creating it if needed with default MLflow behavior."""
    import mlflow

    mlflow.set_experiment(experiment_name)
