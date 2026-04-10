"""
Map device ``smoothness_log`` envelopes (10-minute aggregates) to model feature rows.

Training target schema matches ``src.mlops.training_pipeline`` / synthetic 18-feature trips:
one row per 10-minute window; trip-level UX aggregates scores at end of trip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Union

from src.core.smoothness_ml_engine import parse_telematics_event

# Column order for XGBoost / MLflow — must match 18-feature training pipeline
DEVICE_AGGREGATE_FEATURE_COLUMNS: List[str] = [
    "avg_accel_g",
    "avg_accel_std",
    "max_decel_g",
    "total_harsh_brakes",
    "total_harsh_accels",
    "avg_lateral_g",
    "max_lateral_g",
    "total_harsh_corners",
    "avg_speed_kmh",
    "avg_speed_std",
    "max_speed_kmh",
    "avg_jerk",
    "avg_jerk_std",
    "max_jerk",
    "avg_rpm",
    "max_rpm",
    "total_idle_seconds",
    "total_over_revs",
]

Envelope = Union[Mapping[str, Any], MutableMapping[str, Any]]


def unwrap_smoothness_envelope(payload: Envelope) -> Dict[str, Any]:
    """
    Accept full batch wrapper or inner event dict.

    Supports:
    - ``{"event": {...}}`` (with ``event.details``)
    - ``{"details": ...}`` (inner event only)
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    ev = payload.get("event")
    if isinstance(ev, Mapping) and "details" in ev:
        return dict(ev)
    if "details" in payload:
        return dict(payload)
    return dict(payload)


def window_weight_seconds(payload: Envelope) -> float:
    """Prefer ``details.window_seconds``, then ``details.sample_count``, else 1.0."""
    inner = unwrap_smoothness_envelope(payload)
    details = inner.get("details") or {}
    if details.get("window_seconds") is not None:
        return max(float(details["window_seconds"]), 1e-6)
    if details.get("sample_count") is not None:
        return max(float(details["sample_count"]), 1.0)
    return 1.0


def features_row_from_smoothness_log(payload: Envelope) -> Dict[str, float]:
    """
    One 10-minute device aggregate → one dict aligned with ``DEVICE_AGGREGATE_FEATURE_COLUMNS``.

    Per-window counts (harsh brakes, etc.) use the same column names as trip-level training
    data; for a single window they are the counts *in that window only*.
    """
    inner = unwrap_smoothness_envelope(payload)
    p = parse_telematics_event(inner)
    return {
        "avg_accel_g": float(p["mean_accel_g"]),
        "avg_accel_std": float(p["accel_std_g"]),
        "max_decel_g": float(p["max_decel_g"]),
        "total_harsh_brakes": float(p["harsh_brake_count"]),
        "total_harsh_accels": float(p["harsh_accel_count"]),
        "avg_lateral_g": float(p["mean_lateral_g"]),
        "max_lateral_g": float(p["max_lateral_g"]),
        "total_harsh_corners": float(p["harsh_corner_count"]),
        "avg_speed_kmh": float(p["mean_speed_kmh"]),
        "avg_speed_std": float(p["speed_std"]),
        "max_speed_kmh": float(p["max_speed_kmh"]),
        "avg_jerk": float(p["jerk_mean"]),
        "avg_jerk_std": float(p["jerk_std"]),
        "max_jerk": float(p["jerk_max"]),
        "avg_rpm": float(p["mean_rpm"]),
        "max_rpm": float(p["max_rpm"]),
        "total_idle_seconds": float(p["idle_seconds"]),
        "total_over_revs": float(p["over_rev_count"]),
    }
