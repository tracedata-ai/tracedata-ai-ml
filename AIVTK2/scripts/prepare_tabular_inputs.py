"""
Prepare AIVT2 tabular input bundle for TraceData smoothness regression model.

Outputs (pickle/joblib friendly for tabular flow):
- data/testing_dataset.sav
- data/ground_truth_dataset.sav
- data/background_dataset.sav
- model/smoothness_model.joblib
- bundle_metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.features import extract_smoothness_features
from src.core.model_contract import SMOOTHNESS_FEATURE_COLUMNS, smoothness_label_from_features
from src.utils.simulator import generate_telemetry


def _build_rows(n_samples: int, window_minutes: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)
    styles = ("smooth", "jerky", "unsafe")
    probs = (0.42, 0.38, 0.20)

    rows: list[dict[str, Any]] = []
    for _ in range(n_samples):
        style = rng.choice(styles, p=probs).item()
        age = int(rng.integers(21, 66))
        years_exp = max(0, age - int(rng.integers(20, 28)))
        points = generate_telemetry(style=style, duration_minutes=window_minutes)
        feat = extract_smoothness_features(points)
        rows.append(
            {
                "accel_fluidity": float(feat["accel_fluidity"]),
                "driving_consistency": float(feat["driving_consistency"]),
                "comfort_zone_percent": float(feat["comfort_zone_percent"]),
                "age": age,
                "years_experience": years_exp,
                "style": style,
            }
        )
    df = pd.DataFrame(rows)
    df["smoothness_label"] = smoothness_label_from_features(df)
    return df


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Train first (production pipeline) or pass --model-path."
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIVT2 tabular datasets and model bundle.")
    parser.add_argument("--output-dir", default="AIVTK2/artifacts/tabular")
    parser.add_argument("--bundle-name", default=None)
    parser.add_argument("--model-path", default="models/smoothness_model.joblib")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--background-rows", type=int, default=96)
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_root = root / args.output_dir
    bundle_name = args.bundle_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = output_root / bundle_name
    data_dir = bundle_dir / "data"
    model_dir = bundle_dir / "model"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _build_rows(
        n_samples=int(args.n_samples),
        window_minutes=int(args.window_minutes),
        random_seed=int(args.random_seed),
    )

    feature_cols = list(SMOOTHNESS_FEATURE_COLUMNS)
    # SHAP/explainability plugins require test data schema aligned to model features.
    # Keep an additional fairness-oriented dataset with sensitive attributes separately.
    test_df = df[feature_cols].copy()
    test_with_sensitive_df = df[feature_cols + ["age", "years_experience"]].copy()
    gt_df = df[["smoothness_label"]].copy()
    bg_df = df[feature_cols].sample(
        n=min(int(args.background_rows), len(df)),
        random_state=int(args.random_seed),
    )

    test_path = data_dir / "testing_dataset.sav"
    test_sensitive_path = data_dir / "testing_dataset_with_sensitive.sav"
    gt_path = data_dir / "ground_truth_dataset.sav"
    bg_path = data_dir / "background_dataset.sav"
    test_df.to_pickle(test_path)
    test_with_sensitive_df.to_pickle(test_sensitive_path)
    gt_df.to_pickle(gt_path)
    bg_df.to_pickle(bg_path)

    source_model = _resolve_model_path(str(args.model_path))
    target_model = model_dir / "smoothness_model.joblib"
    shutil.copy2(source_model, target_model)

    metadata = {
        "framework": "AIVT2 tabular input bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_name": bundle_name,
        "model": {
            "path": str(target_model.relative_to(bundle_dir)),
            "serialization": "joblib",
        },
        "datasets": {
            "testing_dataset": str(test_path.relative_to(bundle_dir)),
            "testing_dataset_with_sensitive": str(test_sensitive_path.relative_to(bundle_dir)),
            "ground_truth_dataset": str(gt_path.relative_to(bundle_dir)),
            "background_dataset": str(bg_path.relative_to(bundle_dir)),
            "serialization": "pickle (.sav)",
            "target_column": "smoothness_label",
            "feature_columns": feature_cols,
            "sensitive_columns_available_in": "testing_dataset_with_sensitive.sav",
            "sensitive_columns": ["age", "years_experience"],
        },
        "generation_config": {
            "n_samples": int(args.n_samples),
            "background_rows": int(args.background_rows),
            "window_minutes": int(args.window_minutes),
            "random_seed": int(args.random_seed),
        },
    }
    (bundle_dir / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({"bundle_dir": str(bundle_dir), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()

