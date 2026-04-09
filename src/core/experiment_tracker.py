"""
Legacy JSON run manifests under ``mlruns/<experiment>/<run_id>/run.json``.

Training flows now use **real MLflow** (``src.utils.trainer``, ``src.mlops.*``).
Keep this module only if you still import ``tracker`` from here; new code should use
``import mlflow`` or ``src.mlops.mlflow_common``.
"""

import json
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class ExperimentTracker:
    """
    Writes experiment runs as JSON manifests to the mlruns/ directory.
    Output is compatible with the MLFlow tracking data model.
    """

    MLRUNS_DIR = Path("mlruns")

    def __init__(self):
        self._current_run: Optional[dict] = None
        self._run_dir: Optional[Path] = None

    @contextmanager
    def start_run(self, experiment: str = "default"):
        """Start a new experiment run. Use as a context manager."""
        run_id = str(uuid.uuid4()).replace("-", "")[:20]
        experiment_dir = self.MLRUNS_DIR / experiment
        self._run_dir = experiment_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._current_run = {
            "run_id": run_id,
            "experiment": experiment,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": []
        }

        print(f"🔬 MLFlow-Compatible Run ID: {run_id}")
        print(f"📂 Experiment: {experiment}")

        try:
            yield self
            self._current_run["status"] = "FINISHED"
        except Exception as e:
            self._current_run["status"] = "FAILED"
            self._current_run["error"] = str(e)
            raise
        finally:
            self._current_run["end_time"] = datetime.now(timezone.utc).isoformat()
            self._save_run()

    def log_param(self, key: str, value):
        """Log a single hyperparameter."""
        if self._current_run:
            self._current_run["params"][key] = value

    def log_params(self, params: dict):
        """Log multiple hyperparameters at once."""
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float):
        """Log a named evaluation metric."""
        if self._current_run:
            self._current_run["metrics"][key] = value

    def set_tag(self, key: str, value: str):
        """Set a run tag (e.g., 'quality_gate': 'PASSED')."""
        if self._current_run:
            self._current_run["tags"][key] = value

    def log_artifact(self, local_path: str, artifact_path: str = "artifacts"):
        """Copy an artifact file into the run directory."""
        if self._current_run and self._run_dir:
            src = Path(local_path)
            if src.exists():
                dest_dir = self._run_dir / artifact_path
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_dir / src.name)
                self._current_run["artifacts"].append(str(dest_dir / src.name))

    def _save_run(self):
        """Persist the run manifest as a JSON file."""
        if self._run_dir and self._current_run:
            run_file = self._run_dir / "run.json"
            with open(run_file, "w") as f:
                json.dump(self._current_run, f, indent=2)

            # Print summary
            status_icon = "✅" if self._current_run["status"] == "FINISHED" else "❌"
            print(f"\n{status_icon} Run {self._current_run['run_id'][:8]}... {self._current_run['status']}")
            print(f"   Params: {self._current_run['params']}")
            print(f"   Metrics: {self._current_run['metrics']}")
            print(f"   Tags: {self._current_run['tags']}")
            print(f"   Manifest: {run_file}")


# Singleton tracker — import this everywhere
tracker = ExperimentTracker()
