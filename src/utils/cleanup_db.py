import sqlite3

from src.core.config import DB_NAME


def cleanup():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    print(f"🧹 Cleaning up {DB_NAME}...")

    # 1. Check for old tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Current tables: {tables}")

    if "trip_telemetry" in tables:
        print("🗑️ Dropping legacy table 'trip_telemetry'...")
        cursor.execute("DROP TABLE trip_telemetry;")

    # 2. Cleanup orphaned data if any
    # (Optional: ensuring trips points only belong to valid trips)
    # cursor.execute("DELETE FROM telemetry_points WHERE trip_id NOT IN (SELECT trip_id FROM trips)")

    # 3. Vacuum - Optmizes database file size after dropping tables
    print("✨ Vacuuming database...")
    cursor.execute("VACUUM;")

    conn.commit()
    conn.close()
    print("✅ Database cleanup complete.")


if __name__ == "__main__":
    cleanup()
