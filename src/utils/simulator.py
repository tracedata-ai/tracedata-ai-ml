import sqlite3
import json
import random
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
from src.core.config import DB_NAME

def init_db():
    """Initializes the SQLite database with the required schema."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Base tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            years_experience INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            trip_id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER,
            start_time TEXT,
            end_time TEXT,
            distance_km REAL,
            accel_fluidity REAL,
            driving_consistency REAL,
            comfort_zone_percent REAL,
            harsh_braking_count INTEGER,
            harsh_acceleration_count INTEGER,
            speeding_events INTEGER,
            smoothness_score REAL,
            safety_score REAL,
            overall_score REAL,
            FOREIGN KEY (driver_id) REFERENCES drivers (driver_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_points (
            point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id INTEGER,
            timestamp TEXT,
            speed_kmh REAL,
            acceleration_ms2 REAL,
            lat REAL,
            lon REAL,
            FOREIGN KEY (trip_id) REFERENCES trips (trip_id)
        )
    """)

    def try_add_column(table, column, definition):
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError:
            pass # Already exists

    try_add_column("drivers", "smoothness_avg", "REAL DEFAULT 0")
    try_add_column("drivers", "safety_avg", "REAL DEFAULT 0")
    try_add_column("drivers", "overall_avg", "REAL DEFAULT 0")
    try_add_column("drivers", "trip_count", "INTEGER DEFAULT 0")
    try_add_column("drivers", "explanation_json", "TEXT")
    try_add_column("drivers", "fairness_metadata_json", "TEXT")
    
    try_add_column("trips", "explanation_json", "TEXT")
    try_add_column("trips", "fairness_metadata_json", "TEXT")

    conn.commit()
    conn.close()
    print(f"✅ Database {DB_NAME} initialized and migrated.")

def generate_telemetry(style="smooth", duration_minutes=30):
    """
    Generates synthetic telemetry points for a trip.
    Styles: 'smooth', 'jerky', 'unsafe'
    """
    points = []
    start_time = datetime.now()
    
    # Base parameters based on style
    if style == "smooth":
        accel_range = (-0.4, 0.4)
        jerk_std = 0.05
    elif style == "jerky":
        accel_range = (-0.9, 0.9)
        jerk_std = 0.3
    else: # unsafe
        accel_range = (-1.2, 1.2)
        jerk_std = 0.5

    current_speed = 0.0 # km/h
    current_accel = 0.0 # m/s^2
    
    # Generate point every 30 seconds
    num_points = duration_minutes * 2
    
    for i in range(num_points):
        timestamp = (start_time + timedelta(seconds=i*30)).isoformat()
        
        # Physics simulation (very simplified)
        # Jerk is the change in acceleration
        jerk = np.random.normal(0, jerk_std)
        current_accel = np.clip(current_accel + jerk, accel_range[0], accel_range[1])
        
        # Change in speed (v = u + at)
        # 30 seconds interval
        speed_delta = (current_accel * 30) * 3.6 # convert m/s to km/h
        current_speed = max(0, min(100, current_speed + speed_delta))
        
        points.append({
            "timestamp": timestamp,
            "speed_kmh": round(current_speed, 2),
            "acceleration_ms2": round(current_accel, 3),
            "lat": 1.35 + (i * 0.0001), # Dummy movement
            "lon": 103.8 + (i * 0.0001)
        })
        
    return points

def simulate_data(num_drivers=5, trips_per_driver=10):
    """Generates and stores synthetic data for multiple drivers and trips."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    driver_names = ["Ahmad", "Siti", "Chen", "Raj", "Linda"]
    
    for i in range(num_drivers):
        name = driver_names[i % len(driver_names)]
        age = random.randint(22, 65)
        # Experience is roughly age - distance
        exp = max(0, age - random.randint(20, 25))
        
        cursor.execute("""
            INSERT INTO drivers (name, age, years_experience) 
            VALUES (?, ?, ?)
        """, (name, age, exp))
        driver_id = cursor.lastrowid
        
        for t in range(trips_per_driver):
            # Choose a style randomly but biased for some drivers
            if i == 0: style = "smooth" # Ahmad is a pro
            elif i == 1: style = "jerky" # Siti is learning
            else: style = random.choice(["smooth", "jerky", "unsafe"])
            
            duration = random.randint(20, 120)
            telemetry = generate_telemetry(style, duration)
            
            # Insert trip record (initially with NULL scores)
            start_t = telemetry[0]["timestamp"]
            end_t = telemetry[-1]["timestamp"]
            dist = (len(telemetry) * 30 * 50) / 3600 # rough estimate at 50km/h
            
            cursor.execute("""
                INSERT INTO trips (driver_id, start_time, end_time, distance_km)
                VALUES (?, ?, ?, ?)
            """, (driver_id, start_t, end_t, dist))
            trip_id = cursor.lastrowid
            
            # Insert individual telemetry points
            cursor.executemany("""
                INSERT INTO telemetry_points (trip_id, timestamp, speed_kmh, acceleration_ms2, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (trip_id, p["timestamp"], p["speed_kmh"], p["acceleration_ms2"], p["lat"], p["lon"])
                for p in telemetry
            ])
            
    conn.commit()
    conn.close()
    print(f"✅ Simulated data for {num_drivers} drivers and {num_drivers * trips_per_driver} trips.")

if __name__ == "__main__":
    init_db()
    simulate_data()
