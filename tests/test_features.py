"""
Unit tests for src/core/features.py

These tests validate the feature extraction functions against known inputs.
The goal is to confirm that:
1. Smooth driving (gentle acceleration) produces LOW accel_fluidity and HIGH comfort_zone_percent
2. Jerky driving (sudden acceleration changes) produces HIGH accel_fluidity
3. Edge cases (empty input, single point) are handled safely
"""

from src.core.features import extract_smoothness_features, detect_safety_events

# ─── Fixtures ────────────────────────────────────────────────────────────────

SMOOTH_TRIP = [
    {"acceleration_ms2": 0.1, "speed_kmh": 40},
    {"acceleration_ms2": 0.15, "speed_kmh": 42},
    {"acceleration_ms2": 0.2, "speed_kmh": 45},
    {"acceleration_ms2": 0.1, "speed_kmh": 43},
    {"acceleration_ms2": 0.05, "speed_kmh": 40},
]

JERKY_TRIP = [
    {"acceleration_ms2": 2.0, "speed_kmh": 60},
    {"acceleration_ms2": -1.8, "speed_kmh": 40},
    {"acceleration_ms2": 2.5, "speed_kmh": 70},
    {"acceleration_ms2": -2.2, "speed_kmh": 35},
    {"acceleration_ms2": 1.9, "speed_kmh": 65},
]

UNSAFE_TRIP = [
    {"acceleration_ms2": 0.1, "speed_kmh": 100},  # Speeding
    {"acceleration_ms2": -0.9, "speed_kmh": 85},  # Harsh brake
    {"acceleration_ms2": 0.8, "speed_kmh": 80},  # Harsh acceleration
    {"acceleration_ms2": 0.1, "speed_kmh": 75},
]


# ─── Smoothness Feature Tests ─────────────────────────────────────────────────


class TestExtractSmoothnessFeatures:

    def test_smooth_driver_has_low_fluidity(self):
        """A driver with gentle, consistent acceleration should have low jerk (fluidity)."""
        features = extract_smoothness_features(SMOOTH_TRIP)
        assert (
            features["accel_fluidity"] < 0.5
        ), f"Expected low fluidity for smooth driver, got {features['accel_fluidity']}"

    def test_jerky_driver_has_high_fluidity(self):
        """A driver with sudden acceleration changes should have high fluidity (jerk)."""
        features = extract_smoothness_features(JERKY_TRIP)
        assert (
            features["accel_fluidity"] > 1.0
        ), f"Expected high fluidity for jerky driver, got {features['accel_fluidity']}"

    def test_smooth_driver_high_comfort_zone(self):
        """A smooth driver stays mostly in the comfort band [-0.5, +0.5]."""
        features = extract_smoothness_features(SMOOTH_TRIP)
        assert (
            features["comfort_zone_percent"] > 80.0
        ), f"Expected high comfort zone for smooth driver, got {features['comfort_zone_percent']}"

    def test_jerky_driver_low_comfort_zone(self):
        """A jerky driver spends most time outside the comfort band."""
        features = extract_smoothness_features(JERKY_TRIP)
        assert (
            features["comfort_zone_percent"] == 0.0
        ), f"Expected 0% comfort zone for fully jerky driver, got {features['comfort_zone_percent']}"

    def test_smooth_has_low_consistency(self):
        """A consistent driver should have low std_dev in acceleration."""
        features = extract_smoothness_features(SMOOTH_TRIP)
        assert (
            features["driving_consistency"] < 0.1
        ), f"Expected low consistency for smooth driver, got {features['driving_consistency']}"

    def test_all_features_present(self):
        """The output must always contain exactly the 3 expected feature keys."""
        features = extract_smoothness_features(SMOOTH_TRIP)
        assert "accel_fluidity" in features
        assert "driving_consistency" in features
        assert "comfort_zone_percent" in features

    def test_empty_input_returns_zeros(self):
        """Empty telemetry should return zero values, not raise an error."""
        features = extract_smoothness_features([])
        assert features["accel_fluidity"] == 0.0
        assert features["driving_consistency"] == 0.0
        assert features["comfort_zone_percent"] == 0.0

    def test_single_point_returns_safely(self):
        """A single telemetry point cannot compute jerk — should handle gracefully."""
        features = extract_smoothness_features([{"acceleration_ms2": 0.5, "speed_kmh": 50}])
        assert features["accel_fluidity"] == 0.0  # Can't compute jerk with 1 point
        assert features["comfort_zone_percent"] == 100.0  # 0.5 is on the edge, included


# ─── Safety Event Detection Tests ────────────────────────────────────────────


class TestDetectSafetyEvents:

    def test_speeding_detected(self):
        """Trips exceeding 95 km/h should be flagged as speeding events."""
        events = detect_safety_events(UNSAFE_TRIP)
        assert events["speeding_events"] >= 1, "Expected at least 1 speeding event"

    def test_harsh_braking_detected(self):
        """Acceleration below -0.8 m/s² should be flagged as harsh braking."""
        events = detect_safety_events(UNSAFE_TRIP)
        assert events["harsh_braking_count"] >= 1, "Expected at least 1 harsh braking event"

    def test_harsh_acceleration_detected(self):
        """Acceleration above +0.7 m/s² should be flagged as harsh acceleration."""
        events = detect_safety_events(UNSAFE_TRIP)
        assert events["harsh_acceleration_count"] >= 1, "Expected at least 1 harsh acceleration"

    def test_smooth_trip_no_safety_events(self):
        """A smooth, law-abiding trip should produce zero safety events."""
        events = detect_safety_events(SMOOTH_TRIP)
        assert events["speeding_events"] == 0
        assert events["harsh_braking_count"] == 0
        assert events["harsh_acceleration_count"] == 0

    def test_all_event_keys_present(self):
        """The output must always contain the 3 expected safety event keys."""
        events = detect_safety_events(SMOOTH_TRIP)
        assert "speeding_events" in events
        assert "harsh_braking_count" in events
        assert "harsh_acceleration_count" in events
