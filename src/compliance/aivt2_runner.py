"""
AIVT2-aligned nightly compliance runner for the 3-feature smoothness model.

This script is CI-friendly:
1) resolves an MLflow run (explicit --run-id or latest in experiment),
2) runs fairness/explainability/robustness checks,
3) writes JSON artifacts under reports/aivt2/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import mlflow
import numpy as np

from src.core.features import extract_smoothness_features
from src.core.model_contract import smoothness_label_from_features
from src.inference.smoothness_inference import SmoothnessInference
from src.utils.simulator import generate_telemetry


@dataclass
class ComplianceThresholds:
    fairness_min_disparate_impact: float = 0.80
    fairness_max_abs_statistical_parity_diff: float = 0.20
    explainability_max_mean_abs_additivity_error: float = 1e-4
    robustness_max_mean_abs_score_delta: float = 8.0


def _resolve_run_id(
    tracking_uri: str,
    experiment_name: str,
    explicit_run_id: str | None,
) -> str:
    if explicit_run_id:
        return explicit_run_id

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name!r}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found in experiment: {experiment_name!r}")
    return str(runs.iloc[0]["run_id"])


def _build_eval_rows(n_samples: int, window_minutes: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows: List[Dict[str, Any]] = []
    styles = ("smooth", "jerky", "unsafe")
    probs = (0.42, 0.38, 0.20)

    for _ in range(n_samples):
        style = rng.choice(styles, p=probs).item()
        age = int(rng.integers(21, 66))
        years_exp = max(0, age - int(rng.integers(20, 28)))
        points = generate_telemetry(style=style, duration_minutes=window_minutes)
        features = extract_smoothness_features(points)
        rows.append(
            {
                "style": style,
                "age": age,
                "years_experience": years_exp,
                "features": {k: float(v) for k, v in features.items()},
            }
        )
    return rows


def _safe_rate(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def _run_fairness(
    scorer: SmoothnessInference,
    rows: Sequence[Mapping[str, Any]],
    thresholds: ComplianceThresholds,
) -> Dict[str, Any]:
    pred_scores: List[float] = []
    labels: List[int] = []
    for row in rows:
        pred = scorer.predict_from_features(row["features"])
        pred_scores.append(pred)
        labels.append(1 if pred >= 80.0 else 0)

    old_mask = [1 if int(r["age"]) >= 35 else 0 for r in rows]
    priv = [labels[i] for i in range(len(labels)) if old_mask[i] == 1]
    unpriv = [labels[i] for i in range(len(labels)) if old_mask[i] == 0]

    priv_rate = _safe_rate(sum(priv), len(priv))
    unpriv_rate = _safe_rate(sum(unpriv), len(unpriv))
    disparate_impact = _safe_rate(unpriv_rate, priv_rate) if priv_rate > 0 else 0.0
    stat_parity_diff = float(unpriv_rate - priv_rate)

    passed = (
        disparate_impact >= thresholds.fairness_min_disparate_impact
        and abs(stat_parity_diff) <= thresholds.fairness_max_abs_statistical_parity_diff
    )
    return {
        "status": "pass" if passed else "fail",
        "protected_attribute": "age>=35",
        "favorable_threshold": 80.0,
        "metrics": {
            "disparate_impact": disparate_impact,
            "statistical_parity_difference": stat_parity_diff,
            "privileged_favorable_rate": priv_rate,
            "unprivileged_favorable_rate": unpriv_rate,
            "sample_count": len(rows),
        },
        "thresholds": {
            "min_disparate_impact": thresholds.fairness_min_disparate_impact,
            "max_abs_statistical_parity_difference": (
                thresholds.fairness_max_abs_statistical_parity_diff
            ),
        },
    }


def _run_explainability(
    scorer: SmoothnessInference,
    rows: Sequence[Mapping[str, Any]],
    thresholds: ComplianceThresholds,
) -> Dict[str, Any]:
    additivity_errors: List[float] = []
    global_abs: Dict[str, List[float]] = {}
    for row in rows:
        features = row["features"]
        pred = scorer.predict_from_features(features)
        exp = scorer.explain_features(features)
        base = float(exp["base_value"])
        attrib_sum = 0.0
        for k, v in exp.items():
            if k == "base_value":
                continue
            attrib_sum += float(v)
            global_abs.setdefault(k, []).append(abs(float(v)))
        recon = base + attrib_sum
        additivity_errors.append(abs(recon - pred))

    mean_abs_error = float(np.mean(additivity_errors)) if additivity_errors else 0.0
    global_importance = {k: float(np.mean(vals)) for k, vals in global_abs.items()}
    passed = mean_abs_error <= thresholds.explainability_max_mean_abs_additivity_error
    return {
        "status": "pass" if passed else "fail",
        "metrics": {
            "mean_abs_additivity_error": mean_abs_error,
            "max_abs_additivity_error": float(np.max(additivity_errors))
            if additivity_errors
            else 0.0,
            "sample_count": len(rows),
            "global_mean_abs_attribution": global_importance,
        },
        "thresholds": {
            "max_mean_abs_additivity_error": (
                thresholds.explainability_max_mean_abs_additivity_error
            )
        },
        "method": "xgboost_pred_contribs",
    }


def _perturb_features(
    features: Mapping[str, float],
    rng: np.random.Generator,
    noise: float,
) -> Dict[str, float]:
    out = dict(features)
    out["accel_fluidity"] = max(0.0, float(out["accel_fluidity"]) + float(rng.normal(0.0, noise)))
    out["driving_consistency"] = max(
        0.0, float(out["driving_consistency"]) + float(rng.normal(0.0, noise))
    )
    czp = float(out["comfort_zone_percent"]) + float(rng.normal(0.0, noise * 100.0))
    out["comfort_zone_percent"] = float(np.clip(czp, 0.0, 100.0))
    return out


def _run_robustness(
    scorer: SmoothnessInference,
    rows: Sequence[Mapping[str, Any]],
    thresholds: ComplianceThresholds,
    seed: int,
    perturbation_noise: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    for row in rows:
        base = scorer.predict_from_features(row["features"])
        perturbed_features = _perturb_features(row["features"], rng, perturbation_noise)
        perturbed = scorer.predict_from_features(perturbed_features)
        deltas.append(abs(base - perturbed))
    mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
    passed = mean_abs_delta <= thresholds.robustness_max_mean_abs_score_delta
    return {
        "status": "pass" if passed else "fail",
        "method": "feature_noise_perturbation",
        "metrics": {
            "mean_abs_score_delta": mean_abs_delta,
            "max_abs_score_delta": float(np.max(deltas)) if deltas else 0.0,
            "sample_count": len(rows),
        },
        "thresholds": {
            "max_mean_abs_score_delta": thresholds.robustness_max_mean_abs_score_delta,
            "noise_sigma": perturbation_noise,
        },
    }


def _regression_snapshot(scorer: SmoothnessInference, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    feats = [r["features"] for r in rows]
    frame = np.array(
        [[f["accel_fluidity"], f["driving_consistency"], f["comfort_zone_percent"]] for f in feats]
    )
    y_true = smoothness_label_from_features(
        np_to_frame(frame)
    )
    y_pred = np.array([scorer.predict_from_features(f) for f in feats], dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = 0.0 if math.isclose(denom, 0.0) else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {"mae": mae, "r2": r2, "sample_count": int(len(rows))}


def np_to_frame(frame: np.ndarray) -> Any:
    import pandas as pd

    return pd.DataFrame(
        {
            "accel_fluidity": frame[:, 0],
            "driving_consistency": frame[:, 1],
            "comfort_zone_percent": frame[:, 2],
        }
    )


def run_compliance(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_id: str | None,
    output_dir: Path,
    eval_samples: int,
    window_minutes: int,
    random_seed: int,
    perturbation_noise: float,
    thresholds: ComplianceThresholds,
) -> Dict[str, Any]:
    resolved_run_id = _resolve_run_id(tracking_uri, experiment_name, run_id)
    scorer = SmoothnessInference.from_run(resolved_run_id, tracking_uri)
    rows = _build_eval_rows(eval_samples, window_minutes, random_seed)

    fairness = _run_fairness(scorer, rows, thresholds)
    explainability = _run_explainability(scorer, rows, thresholds)
    robustness = _run_robustness(scorer, rows, thresholds, random_seed + 1, perturbation_noise)
    regression = _regression_snapshot(scorer, rows)

    overall_pass = all(
        section["status"] == "pass" for section in (fairness, explainability, robustness)
    )
    out = {
        "status": "pass" if overall_pass else "fail",
        "framework": "AIVT2-aligned",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_context": {
            "run_id": resolved_run_id,
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
        },
        "sections": {
            "fairness": fairness,
            "explainability": explainability,
            "robustness": robustness,
            "regression_snapshot": regression,
        },
    }

    target = output_dir / resolved_run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (target / "evaluation_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (target / "summary.md").write_text(_build_markdown_summary(out), encoding="utf-8")
    return out


def _build_markdown_summary(result: Mapping[str, Any]) -> str:
    run_ctx = result["run_context"]
    sections = result["sections"]
    fairness = sections["fairness"]
    explainability = sections["explainability"]
    robustness = sections["robustness"]
    regression = sections["regression_snapshot"]

    f_metrics = fairness["metrics"]
    e_metrics = explainability["metrics"]
    r_metrics = robustness["metrics"]

    return (
        "# AIVT2-aligned compliance summary\n\n"
        f"- **Overall status:** `{result['status']}`\n"
        f"- **Run ID:** `{run_ctx['run_id']}`\n"
        f"- **Experiment:** `{run_ctx['experiment_name']}`\n"
        f"- **Generated at (UTC):** `{result['generated_at_utc']}`\n\n"
        "## Section status\n\n"
        f"- Fairness: `{fairness['status']}`\n"
        f"- Explainability: `{explainability['status']}`\n"
        f"- Robustness: `{robustness['status']}`\n\n"
        "## Fairness\n\n"
        f"- Disparate impact: `{f_metrics['disparate_impact']:.6f}`\n"
        f"- Statistical parity difference: `{f_metrics['statistical_parity_difference']:.6f}`\n"
        f"- Privileged favorable rate: `{f_metrics['privileged_favorable_rate']:.6f}`\n"
        f"- Unprivileged favorable rate: `{f_metrics['unprivileged_favorable_rate']:.6f}`\n"
        f"- Sample count: `{f_metrics['sample_count']}`\n\n"
        "## Explainability\n\n"
        f"- Mean abs additivity error: `{e_metrics['mean_abs_additivity_error']:.10f}`\n"
        f"- Max abs additivity error: `{e_metrics['max_abs_additivity_error']:.10f}`\n"
        f"- Sample count: `{e_metrics['sample_count']}`\n\n"
        "## Robustness\n\n"
        f"- Mean abs score delta: `{r_metrics['mean_abs_score_delta']:.6f}`\n"
        f"- Max abs score delta: `{r_metrics['max_abs_score_delta']:.6f}`\n"
        f"- Sample count: `{r_metrics['sample_count']}`\n\n"
        "## Regression snapshot\n\n"
        f"- MAE: `{regression['mae']:.6f}`\n"
        f"- R2: `{regression['r2']:.6f}`\n"
        f"- Sample count: `{regression['sample_count']}`\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIVT2-aligned nightly compliance checks.")
    parser.add_argument("--tracking-uri", default="./mlruns")
    parser.add_argument("--experiment-name", default="smoothness-10min-production")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default="reports/aivt2")
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--perturbation-noise", type=float, default=0.05)
    parser.add_argument("--fairness-min-di", type=float, default=0.80)
    parser.add_argument("--fairness-max-spd", type=float, default=0.20)
    parser.add_argument("--explainability-max-mean-additivity-error", type=float, default=1e-4)
    parser.add_argument("--robustness-max-mean-delta", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = ComplianceThresholds(
        fairness_min_disparate_impact=float(args.fairness_min_di),
        fairness_max_abs_statistical_parity_diff=float(args.fairness_max_spd),
        explainability_max_mean_abs_additivity_error=float(
            args.explainability_max_mean_additivity_error
        ),
        robustness_max_mean_abs_score_delta=float(args.robustness_max_mean_delta),
    )
    result = run_compliance(
        tracking_uri=str(args.tracking_uri),
        experiment_name=str(args.experiment_name),
        run_id=args.run_id,
        output_dir=Path(args.output_dir),
        eval_samples=int(args.eval_samples),
        window_minutes=int(args.window_minutes),
        random_seed=int(args.random_seed),
        perturbation_noise=float(args.perturbation_noise),
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
