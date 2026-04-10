"""Device envelope → 18 training columns."""

from src.core.device_window_features import (
    DEVICE_AGGREGATE_FEATURE_COLUMNS,
    features_row_from_smoothness_log,
    unwrap_smoothness_envelope,
    window_weight_seconds,
)


def _sample_envelope():
    return {
        "batch_id": "BAT-test",
        "event": {
            "event_type": "smoothness_log",
            "details": {
                "sample_count": 600,
                "window_seconds": 600,
                "speed": {
                    "mean_kmh": 72.3,
                    "std_dev": 8.1,
                    "max_kmh": 94.0,
                    "variance": 65.61,
                },
                "longitudinal": {
                    "mean_accel_g": 0.04,
                    "std_dev": 0.12,
                    "max_decel_g": -0.31,
                    "harsh_brake_count": 0,
                    "harsh_accel_count": 0,
                },
                "lateral": {
                    "mean_lateral_g": 0.02,
                    "max_lateral_g": 0.18,
                    "harsh_corner_count": 0,
                },
                "jerk": {"mean": 0.008, "max": 0.041, "std_dev": 0.006},
                "engine": {
                    "mean_rpm": 1820,
                    "max_rpm": 2340,
                    "idle_seconds": 45,
                    "over_rev_count": 0,
                },
            },
        },
    }


def test_unwrap_outer_event():
    raw = _sample_envelope()
    inner = unwrap_smoothness_envelope(raw)
    assert inner["event_type"] == "smoothness_log"
    assert "details" in inner


def test_unwrap_inner_only():
    raw = _sample_envelope()["event"]
    inner = unwrap_smoothness_envelope(raw)
    assert inner == raw


def test_features_row_keys_and_order():
    row = features_row_from_smoothness_log(_sample_envelope())
    assert list(row.keys()) == DEVICE_AGGREGATE_FEATURE_COLUMNS
    assert row["avg_speed_kmh"] == 72.3
    assert row["max_decel_g"] == 0.31


def test_window_weight_prefers_seconds():
    assert window_weight_seconds(_sample_envelope()) == 600.0


def test_window_weight_fallback_sample_count():
    env = _sample_envelope()
    del env["event"]["details"]["window_seconds"]
    assert window_weight_seconds(env) == 600.0
