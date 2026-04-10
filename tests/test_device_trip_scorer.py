"""End-of-trip aggregation (weighted windows)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core.device_window_features import DEVICE_AGGREGATE_FEATURE_COLUMNS
from src.inference.device_trip_scorer import DeviceAggregateTripScorer


def _minimal_envelope(score_hint: float = 0.0):
    # Values chosen so parse succeeds; model is mocked.
    return {
        "event": {
            "event_type": "smoothness_log",
            "details": {
                "window_seconds": 100.0,
                "sample_count": 50,
                "speed": {"mean_kmh": 50.0, "std_dev": 1.0, "max_kmh": 60.0, "variance": 1.0},
                "longitudinal": {
                    "mean_accel_g": 0.01,
                    "std_dev": 0.02,
                    "max_decel_g": -0.1,
                    "harsh_brake_count": 0,
                    "harsh_accel_count": 0,
                },
                "lateral": {
                    "mean_lateral_g": 0.01,
                    "max_lateral_g": 0.05,
                    "harsh_corner_count": 0,
                },
                "jerk": {"mean": 0.001 + score_hint, "max": 0.01, "std_dev": 0.001},
                "engine": {
                    "mean_rpm": 1500,
                    "max_rpm": 2000,
                    "idle_seconds": 0,
                    "over_rev_count": 0,
                },
            },
        }
    }


@pytest.fixture
def mock_scorer():
    n = len(DEVICE_AGGREGATE_FEATURE_COLUMNS)
    booster = MagicMock()
    # pred_contribs: one row, features + bias
    booster.predict = MagicMock(return_value=np.array([[0.1] * n + [0.5]], dtype=float))
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda x: np.array([70.0]))
    model.get_booster.return_value = booster
    bg = pd.DataFrame([{c: 0.0 for c in DEVICE_AGGREGATE_FEATURE_COLUMNS}])
    return DeviceAggregateTripScorer(model, bg)


def test_score_trip_weighted_average_two_windows(mock_scorer):
    a = _minimal_envelope(0.0)
    b = _minimal_envelope(0.0)
    a["event"]["details"]["window_seconds"] = 300.0
    b["event"]["details"]["window_seconds"] = 100.0
    mock_scorer.model.predict.side_effect = [np.array([80.0]), np.array([60.0])]

    out = mock_scorer.score_trip_at_end([a, b])
    # (80*300 + 60*100) / 400 = 75
    assert out["trip_smoothness_score"] == pytest.approx(75.0)
    assert out["window_count"] == 2
    assert out["worst_window_index"] == 1
    assert out["worst_window_score"] == pytest.approx(60.0)


def test_score_trip_requires_windows(mock_scorer):
    with pytest.raises(ValueError, match="at least one"):
        mock_scorer.score_trip_at_end([])
