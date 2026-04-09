import joblib
import sqlite3
import json
import pandas as pd
import numpy as np
from src.core.features import extract_smoothness_features, detect_safety_events
from src.core.explain import TripExplainer
from src.core.config import DB_NAME, SMOOTHNESS_MODEL_PATH

class ScoringService:
    def __init__(self):
        self.model = joblib.load(SMOOTHNESS_MODEL_PATH)
        self.explainer = TripExplainer()

    def calculate_safety_score(self, events):
        """
        Safety Score = 100 - (Total Events * 2)
        Clipping to 0.
        """
        total_events = (
            events["harsh_braking_count"] + 
            events["harsh_acceleration_count"] + 
            events["speeding_events"]
        )
        penalty = total_events * 2
        score = 100 - penalty
        return float(max(0, score))

    def predict_smoothness_score(self, features):
        """Uses the XGBoost model to predict smoothness score."""
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)[0]
        return float(np.clip(prediction, 0, 100))

    def score_trip(self, trip_id):
        """Processes a trip, computes scores, and updates the database."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Get raw telemetry
        cursor.execute("""
            SELECT timestamp, speed_kmh, acceleration_ms2, lat, lon 
            FROM telemetry_points 
            WHERE trip_id = ?
        """, (trip_id,))
        results = cursor.fetchall()
        if not results:
            return None
        
        points = [
            {"timestamp": r[0], "speed_kmh": r[1], "acceleration_ms2": r[2], "lat": r[3], "lon": r[4]}
            for r in results
        ]

        # 1. Feature Extraction
        features = extract_smoothness_features(points)
        events = detect_safety_events(points)

        # 2. Score Calculation
        smoothness_score = self.predict_smoothness_score(features)
        safety_score = self.calculate_safety_score(events)
        overall_score = (smoothness_score + safety_score) / 2
        
        # 3. Generate Explanation
        explanation = self.explainer.explain_trip_shap(features)

        # 4. Update Trip
        cursor.execute("""
            UPDATE trips 
            SET smoothness_score = ?, safety_score = ?, overall_score = ?, explanation_json = ?
            WHERE trip_id = ?
        """, (smoothness_score, safety_score, overall_score, json.dumps(explanation), trip_id))

        # 5. Update Driver Aggregates
        cursor.execute("SELECT driver_id FROM trips WHERE trip_id = ?", (trip_id,))
        driver_id = cursor.fetchone()[0]
        self.update_driver_stats(driver_id, cursor)

        conn.commit()
        conn.close()

        return {
            "smoothness": round(smoothness_score, 2),
            "safety": round(safety_score, 2),
            "overall": round(overall_score, 2),
            "explanation": explanation
        }

    def update_driver_stats(self, driver_id, cursor):
        """Recalculates lifetime averages and aggregate XAI for a driver."""
        cursor.execute("""
            SELECT 
                smoothness_score, safety_score, overall_score,
                accel_fluidity, driving_consistency, comfort_zone_percent
            FROM trips 
            WHERE driver_id = ? AND overall_score IS NOT NULL
        """, (driver_id,))
        rows = cursor.fetchall()
        
        if not rows:
            return

        df = pd.DataFrame(rows, columns=[
            "smoothness", "safety", "overall", 
            "accel_fluidity", "driving_consistency", "comfort_zone_percent"
        ])
        
        # 1. Averages
        avg_smoothness = float(df["smoothness"].mean())
        avg_safety = float(df["safety"].mean())
        avg_overall = float(df["overall"].mean())
        count = len(df)

        # 2. Calculate average features for "Driving Signature"
        avg_feats = {
            "accel_fluidity": float(df["accel_fluidity"].mean()),
            "driving_consistency": float(df["driving_consistency"].mean()),
            "comfort_zone_percent": float(df["comfort_zone_percent"].mean())
        }
        
        # 3. Generate Aggregate Explanation (Driving Signature)
        signature = self.explainer.explain_trip_shap(avg_feats)

        # 4. Update Driver
        cursor.execute("""
            UPDATE drivers SET
                smoothness_avg = ?,
                safety_avg = ?,
                overall_avg = ?,
                trip_count = ?,
                explanation_json = ?
            WHERE driver_id = ?
        """, (
            avg_smoothness,
            avg_safety,
            avg_overall,
            count,
            json.dumps(signature),
            driver_id
        ))

if __name__ == "__main__":
    service = ScoringService()
    # Score all existing trips in the DB
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT trip_id FROM trips")
    trip_ids = [r[0] for r in cursor.fetchall()]
    conn.close()

    print(f"🎯 Scoring {len(trip_ids)} trips...")
    for tid in trip_ids:
        service.score_trip(tid)
    print("✅ All trips scored and driver stats updated.")
