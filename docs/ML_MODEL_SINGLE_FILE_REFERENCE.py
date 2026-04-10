"""
Single-file reference: load smoothness model and score a trip.

Use this as a handoff example for agentic deployment teams.

Run examples:
  uv run python docs/ML_MODEL_SINGLE_FILE_REFERENCE.py --mode mlmodel-dir --mlmodel-dir ./mlmodel_bundle
  uv run python docs/ML_MODEL_SINGLE_FILE_REFERENCE.py --mode local --model-path models/smoothness_model.joblib --serving-dir serving
  uv run python docs/ML_MODEL_SINGLE_FILE_REFERENCE.py --mode mlflow-run --run-id <RUN_ID> --tracking-uri ./mlruns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
from src.core.features import extract_smoothness_features
from src.core.model_contract import SMOOTHNESS_FEATURE_COLUMNS
from src.inference.smoothness_inference import SmoothnessInference
from src.utils.simulator import generate_telemetry


def _build_demo_windows() -> List[List[Dict[str, Any]]]:
    """
    Demo payload: trip represented as multiple ~10-minute ping windows.
    In production, replace with real windows from your event pipeline.
    """
    return [
        generate_telemetry("smooth", duration_minutes=10),
        generate_telemetry("normal", duration_minutes=10),
        generate_telemetry("jerky", duration_minutes=10),
    ]


def _build_feature_frame_from_windows(windows: List[List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for i, win in enumerate(windows):
        if not win:
            raise ValueError(f"windows[{i}] is empty")
        feat = extract_smoothness_features(win)
        rows.append({c: float(feat[c]) for c in SMOOTHNESS_FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=SMOOTHNESS_FEATURE_COLUMNS)


def _deterministic_heuristic_explanation(
    feature_df: pd.DataFrame,
    window_weights: List[float],
    window_scores: List[float],
) -> Dict[str, Any]:
    """
    Fallback explanation when only MLmodel pyfunc is available.
    Uses fixed coefficients from synthetic label generation, not true SHAP.
    """
    coeffs = {
        "accel_fluidity": -80.0,
        "driving_consistency": -40.0,
        "comfort_zone_percent": 0.4,
    }
    w = np.asarray(window_weights, dtype=float)
    w = w / w.sum()
    trip_attribs: Dict[str, float] = {}
    for c in SMOOTHNESS_FEATURE_COLUMNS:
        vals = feature_df[c].astype(float).to_numpy()
        trip_attribs[c] = float(np.dot(vals * coeffs[c], w))

    worst_idx = int(np.argmin(np.asarray(window_scores, dtype=float)))
    return {
        "feature_attributions": trip_attribs,
        "base_value": 0.0,
        "window_count": int(len(window_scores)),
        "worst_window_index": worst_idx,
        "worst_window_score": float(window_scores[worst_idx]),
        "method": "deterministic_heuristic",
    }


def _score_with_mlmodel_dir(mlmodel_dir: str, windows: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Minimal path for a colleague who only has an exported MLmodel directory.
    Drops explainability and returns score-only response for portability.
    """
    model_root = Path(mlmodel_dir)
    if not (model_root / "MLmodel").is_file():
        raise FileNotFoundError(f"Expected MLmodel at: {model_root / 'MLmodel'}")

    pyfunc_model = mlflow.pyfunc.load_model(str(model_root))
    feature_df = _build_feature_frame_from_windows(windows)
    preds = pyfunc_model.predict(feature_df)

    # Trip score = weighted mean across windows (same default weighting spirit as runtime scorer).
    vals = [float(x) for x in list(preds)]
    weights = [max(float(len(w)), 1.0) for w in windows]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    trip_score = float(np.dot(np.asarray(vals, dtype=float), w))
    explanation = _deterministic_heuristic_explanation(feature_df, weights, vals)

    return {
        "trip_smoothness_score": trip_score,
        "explanation": explanation,
        "explainability_mode": "deterministic_heuristic",
        "method": "mlflow_pyfunc_mean_over_windows",
        "note": "Heuristic attributions only. Use local/run mode for true XGBoost pred_contribs.",
    }


def load_scorer(args: argparse.Namespace) -> SmoothnessInference:
    if args.mode == "local":
        if not args.model_path or not args.serving_dir:
            raise ValueError("--model-path and --serving-dir are required for --mode local")
        return SmoothnessInference.from_local_paths(
            model_path=args.model_path,
            serving_dir=args.serving_dir,
        )

    if args.mode == "mlflow-run":
        if not args.run_id or not args.tracking_uri:
            raise ValueError("--run-id and --tracking-uri are required for --mode mlflow-run")
        return SmoothnessInference.from_run(
            run_id=args.run_id,
            tracking_uri=args.tracking_uri,
        )

    raise ValueError(f"Unsupported mode: {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference scorer integration")
    parser.add_argument(
        "--mode",
        choices=["mlmodel-dir", "local", "mlflow-run"],
        required=True,
        help="Load model from MLmodel directory, local files, or MLflow run artifacts.",
    )
    parser.add_argument(
        "--mlmodel-dir",
        default=None,
        help="Folder containing MLmodel file (mlmodel-dir mode)",
    )
    parser.add_argument("--model-path", default=None, help="Path to model .joblib (local mode)")
    parser.add_argument("--serving-dir", default=None, help="Path to serving/ directory (local mode)")
    parser.add_argument("--run-id", default=None, help="MLflow run ID (mlflow-run mode)")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (mlflow-run mode)")
    args = parser.parse_args()

    windows = _build_demo_windows()
    if args.mode == "mlmodel-dir":
        if not args.mlmodel_dir:
            raise ValueError("--mlmodel-dir is required for --mode mlmodel-dir")
        result = _score_with_mlmodel_dir(args.mlmodel_dir, windows)
        print(json.dumps(result, indent=2))
        return

    scorer = load_scorer(args)

    result = scorer.score_trip_from_ping_windows(windows)
    result["explainability_mode"] = "pred_contribs"
    result["note"] = "Model-derived attributions (XGBoost pred_contribs) per window, aggregated to trip."

    # Output contract for downstream agent: trip score + explainability payload.
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

