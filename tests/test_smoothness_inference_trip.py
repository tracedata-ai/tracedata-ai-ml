"""Trip-level aggregation from multiple ping windows."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core.model_contract import SMOOTHNESS_FEATURE_COLUMNS
from src.inference.smoothness_inference import SmoothnessInference


def _pings(n: int, accel: float) -> list[dict]:
    return [{"acceleration_ms2": accel, "speed_kmh": 50.0} for _ in range(n)]


@pytest.fixture
def mock_scorer():
    n_feat = len(SMOOTHNESS_FEATURE_COLUMNS)
    booster = MagicMock()
    booster.predict = MagicMock(return_value=np.array([[0.1] * n_feat + [0.5]], dtype=float))
    model = MagicMock()
    model.predict = MagicMock(side_effect=[np.array([80.0]), np.array([60.0])])
    model.get_booster.return_value = booster
    bg = pd.DataFrame([{c: 0.0 for c in SMOOTHNESS_FEATURE_COLUMNS}])
    return SmoothnessInference(model, bg)


def test_score_trip_weighted_by_ping_count(mock_scorer):
    w0 = _pings(10, 0.1)
    w1 = _pings(30, 0.1)
    mock_scorer.model.predict.side_effect = [np.array([80.0]), np.array([60.0])]
    out = mock_scorer.score_trip_from_ping_windows([w0, w1])
    # (80*10 + 60*30) / 40 = 65
    assert out["trip_smoothness_score"] == pytest.approx(65.0)
    assert out["explanation"]["window_count"] == 2
    assert out["explanation"]["worst_window_index"] == 1


def test_single_window_same_as_two_identical(mock_scorer):
    p = _pings(5, 0.0)
    mock_scorer.model.predict.side_effect = [np.array([77.0])]
    one = mock_scorer.score_trip_from_ping_windows([p])
    mock_scorer.model.predict.side_effect = [np.array([77.0]), np.array([77.0])]
    two = mock_scorer.score_trip_from_ping_windows([p, p])
    assert one["trip_smoothness_score"] == pytest.approx(77.0)
    assert two["trip_smoothness_score"] == pytest.approx(77.0)


def test_score_window_legacy_keys(mock_scorer):
    mock_scorer.model.predict.side_effect = [np.array([70.0])]
    out = mock_scorer.score_window(_pings(3, 0.0))
    assert out["smoothness_score"] == pytest.approx(70.0)
    assert "shap" in out and "features" in out
