import psycopg2
from datetime import datetime

class DBLogger:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="iqms",
            user="postgres",
            password="0000"
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        print("[DB] Connected to PostgreSQL ✓")

    def insert_entrance(self, track_id, gender, age, confidence, camera_id):
        try:
            self.cursor.execute("""
                INSERT INTO entrance_events 
                    (track_id, gender, age_estimate, confidence, camera_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (track_id, gender, age, confidence, camera_id))
        except Exception as e:
            print(f"[DB] Insert error: {e}")

    def update_dwell(self, track_id, dwell_seconds):
        try:
            self.cursor.execute("""
                UPDATE entrance_events 
                SET dwell_seconds = %s
                WHERE track_id = %s 
                AND id = (
                    SELECT id FROM entrance_events 
                    WHERE track_id = %s 
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (dwell_seconds, track_id, track_id))
        except Exception as e:
            print(f"[DB] Update error: {e}")

    def close(self):
        self.cursor.close()
        self.conn.close()