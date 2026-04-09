"""
TARGET SMOOTHNESS DATA - ONCE EVERY 10 MINUTES
{
  "batch_id": "BAT-6ba7b815-9dad-11d1-80b4-00c04fd430c8",
  "ping_type": "batch",
  "source": "telematics_device",
  "is_emergency": false,
  "event": {
    "event_id": "EVT-6ba7b816-9dad-11d1-80b4-00c04fd430c8",
    "device_event_id": "TEL-6ba7b817-9dad-11d1-80b4-00c04fd430c8",
    "trip_id": "TRP-2026-a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6",
    "driver_id": "DRV-ANON-7829",
    "truck_id": "T12345",
    "batch_id": "BAT-6ba7b815-9dad-11d1-80b4-00c04fd430c8",
    "event_type": "smoothness_log",
    "category": "normal_operation",
    "priority": "low",
    "timestamp": "2026-03-07T08:20:00Z",
    "offset_seconds": 1200,
    "trip_meter_km": 24.8,
    "odometer_km": 180224.8,
    "location": { "lat": 1.3485, "lon": 103.8380 },
    "schema_version": "event_v1",
    "details": {
      "sample_count": 600,
      "window_seconds": 600,
      "speed": {
        "mean_kmh": 72.3,
        "std_dev": 8.1,
        "max_kmh": 94.0,
        "variance": 65.61
      },
      "longitudinal": {
        "mean_accel_g": 0.04,
        "std_dev": 0.12,
        "max_decel_g": -0.31,
        "harsh_brake_count": 0,
        "harsh_accel_count": 0
      },
      "lateral": {
        "mean_lateral_g": 0.02,
        "max_lateral_g": 0.18,
        "harsh_corner_count": 0
      },
      "jerk": {
        "mean": 0.008,
        "max": 0.041,
        "std_dev": 0.006
      },
      "engine": {
        "mean_rpm": 1820,
        "max_rpm": 2340,
        "idle_seconds": 45,
        "idle_events": 1,
        "longest_idle_seconds": 38,
        "over_rev_count": 0,
        "over_rev_seconds": 0
      },
      "incident_event_ids": [],
      "raw_log_url": "s3://tracedata-sensors/T12345-batch-20260307-0820.bin"
    },
    "evidence": null
  }
}


=============================================================================
SMOOTHNESS ML ENGINE - Unified Machine Learning Pipeline for Driver Scoring
=============================================================================

OVERVIEW:
    This module consolidates all machine learning components for calculating
    smoothness and safety rewards. It works with windowed telematics data
    (e.g., 12 samples for a 2-hour trip, 1 ping every 10 minutes).

COMPONENTS:
    1. Telematics Event Parsing     - Parse device smoothness_log events
    2. Trip Aggregation             - Combine multiple samples into trip features
    3. Label Generation             - Create training labels from features
    4. Model Training               - Train XGBoost regressor
    5. Scoring Service              - Generate trip and driver scores
    6. Experiment Tracking          - MLFlow-compatible run logging

DATA FLOW:
    Device sends 10-minute windows:
        Event 1 (0-10 min):  jerk.mean=0.008, harsh_brakes=0, ...
        Event 2 (10-20 min): jerk.mean=0.010, harsh_brakes=1, ...
        ...
        Event 12 (110-120 min): jerk.mean=0.009, harsh_brakes=0, ...
                ↓
            AGGREGATE ALL SAMPLES
                ↓
        Trip Features:
            avg_jerk=0.0085
            total_harsh_brakes=2
            max_speed_observed=94.0
                ↓
            PREDICT SMOOTHNESS SCORE (XGBoost)
                ↓
        Output: smoothness=87.5, safety=92.0, overall=89.75

WORKFLOW:
    Scoring:
        1. Collect all telemetry samples for a trip (e.g., 12 events)
        2. Aggregate features across samples
        3. Predict smoothness score (XGBoost)
        4. Calculate safety score (rule-based)
        5. Compute overall trip score

USAGE EXAMPLE:
    # Scoring a trip from telematics samples
    from src.core.smoothness_ml_engine import ScoringEngine

    engine = ScoringEngine()

    # Collect all samples for a trip
    trip_samples = [event_1, event_2, ..., event_12]

    scores = engine.score_trip_from_samples(trip_samples)
    print(f"Smoothness: {scores['smoothness']}")
    print(f"Safety: {scores['safety']}")
    print(f"Overall: {scores['overall']}")

=============================================================================
"""

import os
import sqlite3
from typing import Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.config import (
    DB_NAME,
    MLFLOW_EXPERIMENT_ML_ENGINE,
    MLFLOW_TRACKING_URI,
    SMOOTHNESS_MODEL_PATH,
)
from src.core.explain import TripExplainer
from src.mlops.mlflow_common import log_serving_artifacts, log_xgboost_model

_ML_ENGINE_BG_ROWS = int(os.environ.get("MLFLOW_ML_ENGINE_BACKGROUND_ROWS", "64"))

# =============================================================================
# SECTION 1: TELEMATICS EVENT PARSING & AGGREGATION
# =============================================================================


def parse_telematics_event(event: Dict) -> Dict[str, float]:
    """
    Parse a single 10-minute telematics smoothness_log event.

    Extracts ALL smoothness and safety metrics from device-generated event.
    Includes longitudinal, lateral, speed, jerk, and engine behavior.

    INPUT FORMAT (from telematics device):
        {
            "details": {
                "jerk": {"mean": 0.008, "std_dev": 0.006, "max": 0.041},
                "longitudinal": {
                    "mean_accel_g": 0.04,
                    "std_dev": 0.12,
                    "max_decel_g": -0.31,
                    "harsh_brake_count": 0,
                    "harsh_accel_count": 0
                },
                "lateral": {
                    "mean_lateral_g": 0.02,
                    "max_lateral_g": 0.18,
                    "harsh_corner_count": 0
                },
                "speed": {
                    "mean_kmh": 72.3,
                    "std_dev": 8.1,
                    "max_kmh": 94.0,
                    "variance": 65.61
                },
                "engine": {
                    "mean_rpm": 1820,
                    "max_rpm": 2340,
                    "idle_seconds": 45,
                    "idle_events": 1,
                    "longest_idle_seconds": 38,
                    "over_rev_count": 0,
                    "over_rev_seconds": 0
                }
            }
        }

    RETURNS:
        Dict with 16 features for complete telematics analysis
    """
    details = event.get("details", {})

    # Longitudinal acceleration (braking/acceleration smoothness)
    longi = details.get("longitudinal", {})
    mean_accel_g = float(longi.get("mean_accel_g", 0.0))
    accel_std_g = float(longi.get("std_dev", 0.0))
    max_decel_g = abs(float(longi.get("max_decel_g", 0.0)))  # Always positive
    harsh_brake = int(longi.get("harsh_brake_count", 0))
    harsh_accel = int(longi.get("harsh_accel_count", 0))

    # Lateral acceleration (turning/cornering smoothness)
    lateral = details.get("lateral", {})
    mean_lateral_g = float(lateral.get("mean_lateral_g", 0.0))
    max_lateral_g = float(lateral.get("max_lateral_g", 0.0))
    harsh_corner = int(lateral.get("harsh_corner_count", 0))

    # Speed variability & consistency
    speed = details.get("speed", {})
    mean_speed_kmh = float(speed.get("mean_kmh", 0.0))
    speed_std = float(speed.get("std_dev", 0.0))
    max_speed_kmh = float(speed.get("max_kmh", 0.0))

    # Jerk (smoothness of acceleration transitions)
    jerk = details.get("jerk", {})
    jerk_mean = float(jerk.get("mean", 0.0))
    jerk_std = float(jerk.get("std_dev", 0.0))
    jerk_max = float(jerk.get("max", 0.0))

    # Engine behavior (RPM consistency, over-revving, idling patterns)
    engine = details.get("engine", {})
    mean_rpm = float(engine.get("mean_rpm", 0.0))
    max_rpm = float(engine.get("max_rpm", 0.0))
    idle_seconds = int(engine.get("idle_seconds", 0))
    over_rev_count = int(engine.get("over_rev_count", 0))

    return {
        # Longitudinal (primary)
        "mean_accel_g": round(mean_accel_g, 4),
        "accel_std_g": round(accel_std_g, 4),
        "max_decel_g": round(max_decel_g, 4),
        "harsh_brake_count": harsh_brake,
        "harsh_accel_count": harsh_accel,
        # Lateral
        "mean_lateral_g": round(mean_lateral_g, 4),
        "max_lateral_g": round(max_lateral_g, 4),
        "harsh_corner_count": harsh_corner,
        # Speed & variability
        "mean_speed_kmh": round(mean_speed_kmh, 2),
        "speed_std": round(speed_std, 2),
        "max_speed_kmh": round(max_speed_kmh, 2),
        # Jerk (acceleration smoothness)
        "jerk_mean": round(jerk_mean, 4),
        "jerk_std": round(jerk_std, 4),
        "jerk_max": round(jerk_max, 4),
        # Engine behavior
        "mean_rpm": round(mean_rpm, 1),
        "max_rpm": round(max_rpm, 1),
        "idle_seconds": idle_seconds,
        "over_rev_count": over_rev_count,
    }


def aggregate_trip_samples(events: List[Dict]) -> Dict[str, float]:
    """
    Aggregate multiple telematics samples (10-min windows) into trip-level features.

    For a 2-hour trip: aggregates 12 samples into 1 trip feature set.

    AGGREGATION STRATEGY:
        - Means: Average across all samples
        - Maximums: Highest value observed
        - Counts: Sum all occurrences
        - Durations: Sum idle/over-rev time

    RETURNS:
        Dict with 18 trip-wide aggregate features
    """
    if not events:
        return {
            # Longitudinal
            "avg_accel_g": 0.0,
            "avg_accel_std": 0.0,
            "max_decel_g": 0.0,
            "total_harsh_brakes": 0,
            "total_harsh_accels": 0,
            # Lateral
            "avg_lateral_g": 0.0,
            "max_lateral_g": 0.0,
            "total_harsh_corners": 0,
            # Speed
            "avg_speed_kmh": 0.0,
            "avg_speed_std": 0.0,
            "max_speed_kmh": 0.0,
            # Jerk
            "avg_jerk": 0.0,
            "avg_jerk_std": 0.0,
            "max_jerk": 0.0,
            # Engine
            "avg_rpm": 0.0,
            "max_rpm": 0.0,
            "total_idle_seconds": 0,
            "total_over_revs": 0,
            "sample_count": 0,
        }

    parsed_samples = [parse_telematics_event(event) for event in events]

    # Extract feature lists for aggregation
    accel_g = [s["mean_accel_g"] for s in parsed_samples]
    accel_stds = [s["accel_std_g"] for s in parsed_samples]
    decel_g = [s["max_decel_g"] for s in parsed_samples]
    harsh_brakes = [s["harsh_brake_count"] for s in parsed_samples]
    harsh_accels = [s["harsh_accel_count"] for s in parsed_samples]

    lateral_g = [s["mean_lateral_g"] for s in parsed_samples]
    max_laterals = [s["max_lateral_g"] for s in parsed_samples]
    harsh_corners = [s["harsh_corner_count"] for s in parsed_samples]

    speeds = [s["mean_speed_kmh"] for s in parsed_samples]
    speed_stds = [s["speed_std"] for s in parsed_samples]
    max_speeds = [s["max_speed_kmh"] for s in parsed_samples]

    jerks = [s["jerk_mean"] for s in parsed_samples]
    jerk_stds = [s["jerk_std"] for s in parsed_samples]
    jerk_maxes = [s["jerk_max"] for s in parsed_samples]

    rpms = [s["mean_rpm"] for s in parsed_samples]
    max_rpms = [s["max_rpm"] for s in parsed_samples]
    idle_times = [s["idle_seconds"] for s in parsed_samples]
    over_revs = [s["over_rev_count"] for s in parsed_samples]

    return {
        # Longitudinal aggregates
        "avg_accel_g": round(float(np.mean(accel_g)), 4),
        "avg_accel_std": round(float(np.mean(accel_stds)), 4),
        "max_decel_g": round(float(np.max(decel_g)), 4),
        "total_harsh_brakes": int(sum(harsh_brakes)),
        "total_harsh_accels": int(sum(harsh_accels)),
        # Lateral aggregates
        "avg_lateral_g": round(float(np.mean(lateral_g)), 4),
        "max_lateral_g": round(float(np.max(max_laterals)), 4),
        "total_harsh_corners": int(sum(harsh_corners)),
        # Speed aggregates
        "avg_speed_kmh": round(float(np.mean(speeds)), 2),
        "avg_speed_std": round(float(np.mean(speed_stds)), 2),
        "max_speed_kmh": round(float(np.max(max_speeds)), 2),
        # Jerk aggregates
        "avg_jerk": round(float(np.mean(jerks)), 4),
        "avg_jerk_std": round(float(np.mean(jerk_stds)), 4),
        "max_jerk": round(float(np.max(jerk_maxes)), 4),
        # Engine aggregates
        "avg_rpm": round(float(np.mean(rpms)), 1),
        "max_rpm": round(float(np.max(max_rpms)), 1),
        "total_idle_seconds": int(sum(idle_times)),
        "total_over_revs": int(sum(over_revs)),
        "sample_count": len(events),
    }


# =============================================================================
# SECTION 3: LABEL GENERATION FOR TRAINING
# =============================================================================


def generate_synthetic_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Generate reward-based smoothness scores (0-100).

    PHILOSOPHY: Reward smooth driving behaviors. Baseline 50 represents normal/average
    driving. Score increases with evidence of smooth, efficient operation.

    SCORING SYSTEM:

        BASELINE: 50 (neutral/normal driving)

        REWARDS for smooth behaviors (+points):
            ✓ Low jerk (< 0.008)              +8 pts
            ✓ Smooth acceleration (std < 0.10) +6 pts
            ✓ Gentle braking (< 0.3g)         +5 pts
            ✓ No harsh brakes (count = 0)     +3 pts
            ✓ No harsh accels (count = 0)     +3 pts
            ✓ Minimal lateral forces (< 0.02g) +4 pts
            ✓ No harsh corners (count = 0)    +3 pts
            ✓ Consistent speed (std < 8 km/h) +6 pts
            ✓ Controlled speed (max < 95 km/h) +4 pts
            ✓ Efficient RPM (< 2000 avg)      +5 pts
            ✓ No over-revving (count = 0)     +3 pts
            ✓ Minimal idling (< 50s per 600s) +3 pts

        Range: 0–100
            90+  : Excellent → Eligible for rewards/bonuses
            70–89: Good      → Solid driving
            50–69: Average   → Normal operation, room to improve
            <50  : Poor      → Aggressive driving (not rewarded)

    REWARD CALCULATION:
        Each criterion earns points by how well it meets the smooth driving threshold.
        Multiple good behaviors stack additively (cumulative reward).
    """
    score = 50.0  # Baseline: neutral/normal driving

    # Longitudinal smoothness (most important for safety)
    score += np.where(df["avg_jerk"] < 0.008, 8, 0)  # Low jerk
    score += np.where(df["avg_accel_std"] < 0.10, 6, 0)  # Smooth acceleration
    score += np.where(df["max_decel_g"] < 0.3, 5, 0)  # Gentle braking
    score += np.where(df["total_harsh_brakes"] == 0, 3, 0)  # No harsh braking
    score += np.where(df["total_harsh_accels"] == 0, 3, 0)  # No harsh acceleration

    # Lateral smoothness (cornering - important for vehicle stability)
    score += np.where(df["avg_lateral_g"] < 0.02, 4, 0)  # Minimal lateral forces
    score += np.where(df["total_harsh_corners"] == 0, 3, 0)  # No harsh corners

    # Speed consistency (predictable driving)
    score += np.where(df["avg_speed_std"] < 8.0, 6, 0)  # Consistent speed
    score += np.where(df["max_speed_kmh"] < 95, 4, 0)  # Controlled speed

    # Engine efficiency (vehicle wear and fuel economy)
    score += np.where(df["avg_rpm"] < 2000, 5, 0)  # Efficient RPM
    score += np.where(df["total_over_revs"] == 0, 3, 0)  # No over-revving
    score += np.where(df["total_idle_seconds"] < 50, 3, 0)  # Minimal idling

    # Add small realistic variation (±5 pts)
    score += np.random.normal(0, 2.5, len(df))

    # Clip to valid range [0, 100]
    return np.clip(score, 0, 100)


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================


def train_smoothness_model(
    min_r2_threshold: float = 0.85, test_size: float = 0.2, random_state: int = 42
) -> Optional[xgb.XGBRegressor]:
    """
    Train XGBoost model to predict smoothness from aggregated trip features.

    WORKFLOW:
        1. Load historical trips with aggregated features from DB
        2. Generate synthetic smoothness labels
        3. Split data (80% train, 20% test)
        4. Train XGBoost regressor
        5. Evaluate on test set (MAE, R²)
        6. Validate quality gate (R² >= threshold)
        7. Save model artifact

    INPUT FEATURES (18 comprehensive telematics):
        LONGITUDINAL: avg_accel_g, avg_accel_std, max_decel_g,
                      total_harsh_brakes, total_harsh_accels
        LATERAL: avg_lateral_g, max_lateral_g, total_harsh_corners
        SPEED: avg_speed_kmh, avg_speed_std, max_speed_kmh
        JERK: avg_jerk, avg_jerk_std, max_jerk
        ENGINE: avg_rpm, max_rpm, total_idle_seconds, total_over_revs

    RETURNS:
        Optional[xgb.XGBRegressor]: Trained model if quality gate passed, else None
    """
    feature_columns = [
        # Longitudinal
        "avg_accel_g",
        "avg_accel_std",
        "max_decel_g",
        "total_harsh_brakes",
        "total_harsh_accels",
        # Lateral
        "avg_lateral_g",
        "max_lateral_g",
        "total_harsh_corners",
        # Speed
        "avg_speed_kmh",
        "avg_speed_std",
        "max_speed_kmh",
        # Jerk
        "avg_jerk",
        "avg_jerk_std",
        "max_jerk",
        # Engine
        "avg_rpm",
        "max_rpm",
        "total_idle_seconds",
        "total_over_revs",
    ]

    conn = sqlite3.connect(DB_NAME)
    # Try to load all features; fill missing with defaults
    feature_selector = ", ".join(feature_columns)
    query = f"""
        SELECT trip_id, {feature_selector}
        FROM trips 
        WHERE avg_jerk IS NOT NULL
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"⚠️  Some columns missing from trips table: {e}")
        print("   Attempting fallback with available columns...")
        df = pd.read_sql_query(
            """
            SELECT * FROM trips WHERE avg_jerk IS NOT NULL LIMIT 1
            """,
            conn,
        )
        available_cols = [c for c in feature_columns if c in df.columns]
        print(
            f"   Found {len(available_cols)}/{len(feature_columns)} features: {available_cols}"
        )
        feature_columns = available_cols
        feature_selector = ", ".join(feature_columns)
        df = pd.read_sql_query(
            f"SELECT trip_id, {feature_selector} FROM trips WHERE avg_jerk IS NOT NULL",
            conn,
        )

    conn.close()

    if len(df) < 10:
        print("❌ Not enough data to train. Ensure simulator has generated trips.")
        return None

    # Use available features for label generation
    available_features = [f for f in feature_columns if f in df.columns]
    df["smoothness_label"] = generate_synthetic_labels(df[available_features])

    # Use available features for training
    X = df[available_features]
    y = df["smoothness_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_ML_ENGINE)

    model: Optional[xgb.XGBRegressor] = None
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("num_features", len(available_features))
        mlflow.log_param("features_used", ",".join(available_features))
        mlflow.set_tag("pipeline", "smoothness_ml_engine_sqlite")

        print(
            f"🚀 Training XGBoost on {len(X_train)} trips with {len(available_features)} features..."
        )
        print(f"   Features: {', '.join(available_features[:5])}...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("test_mae", float(mae))
        mlflow.log_metric("test_r2", float(r2))
        mlflow.log_param("quality_gate_min_r2", min_r2_threshold)
        passed = r2 >= min_r2_threshold
        mlflow.log_param("quality_gate_status", "PASSED" if passed else "FAILED")
        mlflow.set_tag("quality_gate", "PASSED" if passed else "FAILED")
        print(f"✅ MAE: {mae:.2f}, R²: {r2:.4f}")

        log_xgboost_model(model, available_features, X_train)
        log_serving_artifacts(
            run_id,
            X_train,
            available_features,
            background_rows=_ML_ENGINE_BG_ROWS,
            random_seed=random_state,
            contract_extra={
                "contract_version": "ml-engine-sqlite",
                "model_name": "smoothness_ml_engine",
                "description": "18f-style columns from trips table when present",
            },
        )

        if not passed:
            print(f"❌ Quality gate FAILED. R²={r2:.2f} < {min_r2_threshold}")
            return None

        joblib.dump(model, SMOOTHNESS_MODEL_PATH)
        print(f"💾 Model saved to {SMOOTHNESS_MODEL_PATH}")
        print(f"📊 MLflow run_id={run_id} ({MLFLOW_EXPERIMENT_ML_ENGINE})")

    return model


# =============================================================================
# SECTION 5: SCORING ENGINE
# =============================================================================


class ScoringEngine:
    """
    Complete scoring pipeline for driver trips from telematics samples.

    Loads trained model and generates smoothness/safety scores for trips.
    """

    def __init__(self):
        """Initialize by loading trained XGBoost model."""
        self.model = joblib.load(SMOOTHNESS_MODEL_PATH)
        self.explainer = TripExplainer()

    def calculate_safety_score(
        self,
        total_harsh_brakes: int,
        total_harsh_accels: int,
        total_harsh_corners: int = 0,
        total_over_revs: int = 0,
    ) -> float:
        """
        Calculate comprehensive safety score from all harsh event counts.

        FORMULA:
            safety = 100
                   - (harsh_brakes × 3)      # Most severe
                   - (harsh_accels × 2)      # Moderate
                   - (harsh_corners × 2)     # Moderate
                   - (over_revs × 1)         # Minor
            safety = max(0, safety)
        """
        penalty = (
            (total_harsh_brakes * 3)
            + (total_harsh_accels * 2)
            + (total_harsh_corners * 2)
            + (total_over_revs * 1)
        )
        score = 100 - penalty
        return float(max(0, score))

    def predict_smoothness_score(self, trip_features: Dict[str, float]) -> float:
        """
        Predict smoothness score using trained XGBoost model.

        Automatically handles available features - works with 4 features or 18.

        PARAMETERS:
            trip_features: Dict with aggregated trip features from aggregate_trip_samples()
                          Model auto-selects features it was trained on
        """
        # Get feature names from model
        expected_features = self.model.get_booster().feature_names

        # Select only features the model expects
        available_features = [f for f in expected_features if f in trip_features]
        missing_features = [f for f in expected_features if f not in available_features]

        if missing_features:
            print(f"⚠️  Missing features: {missing_features}. Using defaults.")
            # Fill missing with defaults
            for f in missing_features:
                trip_features[f] = 0.0

        df = pd.DataFrame([trip_features])[expected_features]
        prediction = self.model.predict(df)[0]
        return float(np.clip(prediction, 0, 100))

    def score_trip_from_samples(self, events: List[Dict]) -> Dict:
        """
        Complete scoring pipeline: samples → aggregated features → scores.

        WORKFLOW:
            1. Aggregate all samples into trip features
            2. Predict smoothness using XGBoost (18 features)
            3. Calculate comprehensive safety score
            4. Compute overall score

        PARAMETERS:
            events: List of telematics smoothness_log events
                    Example: 12 events for 2-hour trip (1 per 10 min)

        RETURNS:
            Dict with keys: smoothness, safety, overall, sample_count,
                           and detailed breakdowns per category
        """
        trip_features = aggregate_trip_samples(events)

        smoothness_score = self.predict_smoothness_score(trip_features)

        safety_score = self.calculate_safety_score(
            trip_features["total_harsh_brakes"],
            trip_features["total_harsh_accels"],
            trip_features.get("total_harsh_corners", 0),
            trip_features.get("total_over_revs", 0),
        )

        overall_score = (smoothness_score + safety_score) / 2

        return {
            # Primary scores
            "smoothness": round(smoothness_score, 2),
            "safety": round(safety_score, 2),
            "overall": round(overall_score, 2),
            # Trip metadata
            "sample_count": trip_features["sample_count"],
            # Longitudinal safety
            "harsh_brakes": trip_features["total_harsh_brakes"],
            "harsh_accels": trip_features["total_harsh_accels"],
            # Lateral safety
            "harsh_corners": trip_features.get("total_harsh_corners", 0),
            # Speed info
            "max_speed_kmh": trip_features["max_speed_kmh"],
            "avg_speed_kmh": trip_features["avg_speed_kmh"],
            # Engine info
            "max_rpm": trip_features["max_rpm"],
            "over_revs": trip_features.get("total_over_revs", 0),
            "idle_seconds": trip_features.get("total_idle_seconds", 0),
            # Raw features for detailed analysis
            "raw_features": trip_features,
        }


# =============================================================================
# SECTION 6: EXPLAINABLE SCORING ENGINE (SHAP/LIME INTEGRATION)
# =============================================================================


class ExplainableScoringEngine(ScoringEngine):
    """
    Extended ScoringEngine with SHAP-based feature importance explanations.

    Provides detailed breakdown of:
    - Which features contributed most to smoothness prediction
    - Feature contributions (positive = better, negative = worse)
    - Waterfall view of prediction (base + contributions)
    - Global importance across all trips (driver signature)

    USAGE:
        >>> engine = ExplainableScoringEngine()
        >>> result = engine.score_trip_from_samples_with_explanation(trip_events)
        >>> print(result['smoothness_explanation']['explanation_text'])
    """

    def __init__(self):
        """Initialize with trained model and SHAP explainer."""
        super().__init__()

        # All possible feature columns
        all_features = [
            # Longitudinal
            "avg_accel_g",
            "avg_accel_std",
            "max_decel_g",
            "total_harsh_brakes",
            "total_harsh_accels",
            # Lateral
            "avg_lateral_g",
            "max_lateral_g",
            "total_harsh_corners",
            # Speed
            "avg_speed_kmh",
            "avg_speed_std",
            "max_speed_kmh",
            # Jerk
            "avg_jerk",
            "avg_jerk_std",
            "max_jerk",
            # Engine
            "avg_rpm",
            "max_rpm",
            "total_idle_seconds",
            "total_over_revs",
        ]

        # Load training data for SHAP background
        conn = sqlite3.connect(DB_NAME)
        # Try to load all features, fall back to what's available
        feature_selector = ", ".join(all_features)
        try:
            training_df = pd.read_sql_query(
                f"""
                SELECT {feature_selector}
                FROM trips 
                WHERE avg_jerk IS NOT NULL
                LIMIT 100
            """,
                conn,
            )
            self.feature_names = list(training_df.columns)
        except Exception:
            # Fall back to available columns
            available = pd.read_sql_query(
                "SELECT * FROM trips LIMIT 1", conn
            ).columns.tolist()
            available_features = [f for f in all_features if f in available]
            feature_selector = (
                ", ".join(available_features) if available_features else "*"
            )
            training_df = pd.read_sql_query(
                f"""
                SELECT {feature_selector}
                FROM trips 
                WHERE avg_jerk IS NOT NULL
                LIMIT 100
            """,
                conn,
            )
            self.feature_names = list(training_df.columns)

        conn.close()

        # Initialize SHAP TreeExplainer for XGBoost
        if len(training_df) > 0:
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.training_data = training_df
        else:
            self.shap_explainer = None
            self.training_data = None
            self.feature_names = all_features

    def explain_smoothness_prediction(self, trip_features: Dict[str, float]) -> Dict:
        """
        Generate SHAP explanation for smoothness prediction.

        Breaks down the model's prediction into individual feature contributions.

        INTERPRETATION:
            Base Value: Model's average prediction across training trips
            Feature Contributions: How each feature pushed score up/down
            Positive contribution: Better than average (good)
            Negative contribution: Worse than average (bad)

        EXAMPLE OUTPUT:
            Smoothness prediction: 87.5/100
            Baseline (avg driver): 75.0/100
            Your result: 87.5/100 (+12.5)

            ✅ POSITIVE FACTORS:
              • avg_jerk: 0.008 [+8.5 pts] (smooth acceleration)
              • avg_accel_std: 0.10 [+4.0 pts] (consistent driving)

            ❌ NEGATIVE FACTORS:
              • total_harsh_brakes: 2 [-5.0 pts] (harsh events)

        RETURNS:
            Dict with:
            - prediction: Predicted smoothness score
            - base_value: Model's baseline
            - feature_contributions: Dict of feature → SHAP value
            - waterfall: Ordered list for waterfall visualization
            - top_positive: Best contributing features
            - top_negative: Worst contributing features
            - explanation_text: Human-readable breakdown
        """
        if self.shap_explainer is None:
            return {
                "prediction": self.predict_smoothness_score(trip_features),
                "error": "SHAP explainer not available (insufficient training data)",
            }

        # Create DataFrame with correct feature order
        features_df = pd.DataFrame([trip_features])[self.feature_names]

        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(features_df)
        base_value = self.shap_explainer.expected_value

        # Get prediction
        prediction = self.predict_smoothness_score(trip_features)

        # Build feature contributions
        contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            contributions[feature_name] = float(shap_values[0, i])

        # Sort by absolute impact
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Separate positive and negative impacts
        positive = [(f, v) for f, v in sorted_contribs if v > 0]
        negative = [(f, v) for f, v in sorted_contribs if v < 0]

        # Build waterfall explanation (base + contributions)
        waterfall = [("base_value", float(base_value))]
        waterfall.extend(sorted_contribs)

        # Generate human-readable summary
        explanation_lines = [
            f"SMOOTHNESS PREDICTION: {prediction:.1f}/100",
            f"Baseline (avg driver): {base_value:.1f}/100",
            f"Your result: {prediction:.1f}/100 ({prediction - base_value:+.1f})",
            "",
        ]

        if positive:
            explanation_lines.append("✅ POSITIVE FACTORS (improving smoothness):")
            for feature, impact in positive:
                value = trip_features.get(feature, "N/A")
                explanation_lines.append(f"  • {feature}: {value} [{impact:+.2f} pts]")

        if negative:
            explanation_lines.append("❌ NEGATIVE FACTORS (reducing smoothness):")
            for feature, impact in negative:
                value = trip_features.get(feature, "N/A")
                explanation_lines.append(f"  • {feature}: {value} [{impact:+.2f} pts]")

        explanation_text = "\n".join(explanation_lines)

        return {
            "prediction": round(prediction, 2),
            "base_value": round(float(base_value), 2),
            "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
            "waterfall": [(f, round(v, 4)) for f, v in waterfall],
            "top_positive": positive,
            "top_negative": negative,
            "explanation_text": explanation_text,
        }

    def score_trip_from_samples_with_explanation(self, events: List[Dict]) -> Dict:
        """
        Score trip AND provide detailed SHAP explanations.

        Combines regular scoring with feature importance breakdown.

        RETURNS:
            {
                "scores": {
                    "smoothness", "safety", "overall",
                    "sample_count", "harsh_brakes", "harsh_accels", "max_speed"
                },
                "smoothness_explanation": {
                    "prediction", "base_value", "feature_contributions",
                    "waterfall", "top_positive", "top_negative", "explanation_text"
                },
                "raw_features": aggregated trip features
            }
        """
        trip_features = aggregate_trip_samples(events)

        # Get scores
        smoothness_score = self.predict_smoothness_score(trip_features)
        safety_score = self.calculate_safety_score(
            trip_features["total_harsh_brakes"], trip_features["total_harsh_accels"]
        )
        overall_score = (smoothness_score + safety_score) / 2

        # Get explanation
        smoothness_explanation = self.explain_smoothness_prediction(trip_features)

        return {
            "scores": {
                "smoothness": round(smoothness_score, 2),
                "safety": round(safety_score, 2),
                "overall": round(overall_score, 2),
                "sample_count": trip_features["sample_count"],
                "harsh_brakes": trip_features["total_harsh_brakes"],
                "harsh_accels": trip_features["total_harsh_accels"],
                "max_speed": trip_features["max_speed_observed"],
            },
            "smoothness_explanation": smoothness_explanation,
            "raw_features": trip_features,
        }

    def get_global_feature_importance(self) -> Dict:
        """
        Get global feature importance across all training trips.

        Shows which features are most important for smoothness prediction.
        Useful for understanding what drives smoothness at the fleet level.

        EXAMPLE OUTPUT:
            GLOBAL FEATURE IMPORTANCE (Fleet-wide)
            =======================================

            1. total_harsh_brakes   │████████░░░░░░░░░░░░│ 35.2%
            2. avg_jerk            │██████░░░░░░░░░░░░░░│ 25.1%
            3. avg_accel_std       │█████░░░░░░░░░░░░░░░│ 21.8%
            4. total_harsh_accels  │████░░░░░░░░░░░░░░░░│ 17.9%

        RETURNS:
            Dict with:
            - feature_importance: Dict of feature → importance score
            - ranking: Ranked list of (feature, importance)
            - interpretation: Visual bar chart of importance
        """
        if self.shap_explainer is None or self.training_data is None:
            return {"error": "SHAP explainer not available"}

        # Calculate SHAP values for training set
        shap_values = self.shap_explainer.shap_values(self.training_data)

        # Mean absolute SHAP values = feature importance
        importance_scores = np.mean(np.abs(shap_values), axis=0)

        # Build importance dict
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = float(importance_scores[i])

        # Rank features
        ranked = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Generate interpretation
        total = sum(importance_scores)
        interpretation_lines = [
            "GLOBAL FEATURE IMPORTANCE (Fleet-wide)",
            "=======================================",
            "",
        ]
        for i, (feature, importance) in enumerate(ranked, 1):
            percentage = (importance / total) * 100
            bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            interpretation_lines.append(f"{i}. {feature:20} │{bar}│ {percentage:5.1f}%")

        interpretation_text = "\n".join(interpretation_lines)

        return {
            "feature_importance": feature_importance,
            "ranking": ranked,
            "interpretation": interpretation_text,
        }


# =============================================================================
# ENTRY POINTS
# =============================================================================

if __name__ == "__main__":
    """
    Example usage: python -m src.core.smoothness_ml_engine
    Demonstrates basic scoring and SHAP-based explainability
    """
    print("=" * 80)
    print("SMOOTHNESS ML ENGINE - TELEMATICS EVENT WORKFLOW WITH SHAP EXPLANATIONS")
    print("=" * 80)

    # Example telematics event (10-minute window)
    example_event = {
        "trip_id": "TRP-2026-a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6",
        "details": {
            "jerk": {"mean": 0.008, "std_dev": 0.006},
            "longitudinal": {
                "mean_accel_g": 0.04,
                "std_dev": 0.12,
                "harsh_brake_count": 0,
                "harsh_accel_count": 0,
            },
            "speed": {"mean_kmh": 72.3, "std_dev": 8.1, "max_kmh": 94.0},
        },
    }

    print("\n[STEP 1] Parsing and aggregating 2-hour trip (12 samples)...")
    trip_events = [example_event for _ in range(12)]
    trip_features = aggregate_trip_samples(trip_events)
    print("✅ Aggregated trip features:")
    for key, value in trip_features.items():
        print(f"   {key}: {value}")

    print("\n[STEP 2] Basic scoring (without explanations)...")
    try:
        engine = ScoringEngine()
        scores = engine.score_trip_from_samples(trip_events)
        print("✅ Trip scores:")
        print(f"   Smoothness: {scores['smoothness']}/100")
        print(f"   Safety: {scores['safety']}/100")
        print(f"   Overall: {scores['overall']}/100")
        print(f"   Samples: {scores['sample_count']}")
    except Exception as e:
        print(f"❌ Scoring failed (model may not be trained): {e}")

    print("\n[STEP 3] Scoring with SHAP feature explanations...")
    try:
        explainer_engine = ExplainableScoringEngine()

        # Get scores + SHAP explanations
        result = explainer_engine.score_trip_from_samples_with_explanation(trip_events)

        print("✅ Trip scores with explainability:")
        print(f"   Smoothness: {result['scores']['smoothness']}/100")
        print(f"   Safety: {result['scores']['safety']}/100")
        print(f"   Overall: {result['scores']['overall']}/100")

        print("\n📊 FEATURE CONTRIBUTION BREAKDOWN (SHAP Analysis):")
        print("-" * 80)
        print(result["smoothness_explanation"]["explanation_text"])
        print("-" * 80)

        print("\n🔍 Aggregated Features:")
        for key, value in result["raw_features"].items():
            print(f"   {key}: {value}")

        print("\n📈 Feature Importance Details:")
        print("   Contributions for individual features:")
        for feature, contrib in result["smoothness_explanation"][
            "feature_contributions"
        ].items():
            direction = "↑" if contrib > 0 else "↓"
            print(f"   {direction} {feature}: {contrib:+.4f}")

    except Exception as e:
        print(f"❌ Explainable scoring failed: {e}")

    print("\n[STEP 4] Global feature importance (fleet-wide SHAP)...")
    try:
        explainer_engine = ExplainableScoringEngine()
        importance = explainer_engine.get_global_feature_importance()

        if "error" not in importance:
            print("✅ Fleet-wide feature ranking:")
            print(importance["interpretation"])
        else:
            print(f"⚠️  {importance['error']}")

    except Exception as e:
        print(f"⚠️  Could not calculate global importance: {e}")

    print("\n" + "=" * 80)
