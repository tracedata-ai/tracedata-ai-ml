import numpy as np


def extract_smoothness_features(telemetry_points):
    """
    Extracts 3 features for Smoothness Scoring:
    1. Accel Fluidity (Mean Absolute Jerk)
    2. Driving Consistency (Standard Deviation of Acceleration)
    3. Comfort Zone % (Percentage of points in [-0.5, 0.5] range)
    """
    if not telemetry_points:
        return {"accel_fluidity": 0.0, "driving_consistency": 0.0, "comfort_zone_percent": 0.0}

    # Extract all acceleration values
    accelerations = np.array([p["acceleration_ms2"] for p in telemetry_points])

    # Feature 1: Accel Fluidity (Jerk)
    # Jerk is the change in acceleration over time.
    # We take the mean of the absolute differences between consecutive points.
    if len(accelerations) > 1:
        jerk = np.diff(accelerations)
        accel_fluidity = float(np.mean(np.abs(jerk)))
    else:
        accel_fluidity = 0.0

    # Feature 2: Driving Consistency (Variance/StdDev)
    # How much does the acceleration vary?
    driving_consistency = float(np.std(accelerations))

    # Feature 3: Comfort Zone %
    # What percentage of the trip is within the comfortable acceleration band?
    comfort_band = (-0.5, 0.5)
    in_comfort = np.sum((accelerations >= comfort_band[0]) & (accelerations <= comfort_band[1]))
    comfort_zone_percent = float((in_comfort / len(accelerations)) * 100)

    return {
        "accel_fluidity": round(accel_fluidity, 4),
        "driving_consistency": round(driving_consistency, 4),
        "comfort_zone_percent": round(comfort_zone_percent, 2),
    }


def detect_safety_events(telemetry_points):
    """
    Detects 3 types of safety events:
    1. Harsh Braking (accel < -0.8)
    2. Harsh Acceleration (accel > 0.7)
    3. Speeding (speed > 95 km/h) - Assuming 90km/h limit + 5km/h buffer
    """
    harsh_braking_count = 0
    harsh_acceleration_count = 0
    speeding_events = 0

    # Simple thresholding
    for p in telemetry_points:
        accel = p["acceleration_ms2"]
        speed = p["speed_kmh"]

        if accel < -0.8:
            harsh_braking_count += 1
        elif accel > 0.7:
            harsh_acceleration_count += 1

        if speed > 95:
            speeding_events += 1

    return {
        "harsh_braking_count": harsh_braking_count,
        "harsh_acceleration_count": harsh_acceleration_count,
        "speeding_events": speeding_events,
    }


if __name__ == "__main__":
    # Test with a dummy trip
    test_points = [
        {"acceleration_ms2": 0.1, "speed_kmh": 20},
        {"acceleration_ms2": 0.2, "speed_kmh": 25},
        {"acceleration_ms2": 0.8, "speed_kmh": 35},  # Harsh accel
        {"acceleration_ms2": -0.9, "speed_kmh": 25},  # Harsh brake
        {"acceleration_ms2": 0.0, "speed_kmh": 96},  # Speeding
    ]

    features = extract_smoothness_features(test_points)
    events = detect_safety_events(test_points)

    print("Smoothness Features:", features)
    print("Safety Events:", events)
