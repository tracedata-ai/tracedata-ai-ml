import sqlite3
from src.core.features import extract_smoothness_features, detect_safety_events

from src.core.config import DB_NAME


def process_trips():
    """Reads raw telemetry for all trips and updates the trips table with features."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get all trips
    cursor.execute("SELECT trip_id FROM trips")
    trip_ids = [r[0] for r in cursor.fetchall()]

    for trip_id in trip_ids:
        # Get individual points for this trip
        cursor.execute(
            """
            SELECT timestamp, speed_kmh, acceleration_ms2, lat, lon 
            FROM telemetry_points 
            WHERE trip_id = ?
        """,
            (trip_id,),
        )
        points_data = cursor.fetchall()

        # Convert to list of dicts for existing extraction logic
        points = [
            {
                "timestamp": p[0],
                "speed_kmh": p[1],
                "acceleration_ms2": p[2],
                "lat": p[3],
                "lon": p[4],
            }
            for p in points_data
        ]

        # Extract features
        smoothness_feats = extract_smoothness_features(points)
        safety_events = detect_safety_events(points)

        # Update trips table
        cursor.execute(
            """
            UPDATE trips 
            SET accel_fluidity = ?, 
                driving_consistency = ?, 
                comfort_zone_percent = ?,
                harsh_braking_count = ?,
                harsh_acceleration_count = ?,
                speeding_events = ?
            WHERE trip_id = ?
        """,
            (
                smoothness_feats["accel_fluidity"],
                smoothness_feats["driving_consistency"],
                smoothness_feats["comfort_zone_percent"],
                safety_events["harsh_braking_count"],
                safety_events["harsh_acceleration_count"],
                safety_events["speeding_events"],
                trip_id,
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ Processed and updated features for {len(trip_ids)} trips.")


if __name__ == "__main__":
    process_trips()
