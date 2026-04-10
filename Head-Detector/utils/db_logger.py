import psycopg2
from datetime import datetime


class DBLogger:
    def __init__(self):
        self.enabled = False
        try:
            self.conn = psycopg2.connect(
                host="localhost",
                port=5432,
                dbname="iqms",
                user="postgres",
                password="0000"
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            self._create_table()
            self.enabled = True
            print("[DB] Connected to PostgreSQL ✓")
        except Exception as e:
            print(f"[DB] WARNING: Could not connect to PostgreSQL — running without DB. Error: {e}")

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS queue_events (
                id            SERIAL PRIMARY KEY,
                timestamp     TIMESTAMP DEFAULT NOW(),
                track_id      INT,
                camera_id     VARCHAR(100),
                confidence    FLOAT,
                dwell_seconds FLOAT DEFAULT 0
            )
        """)

    def insert_queue_event(self, track_id, confidence, camera_id):
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                INSERT INTO queue_events
                    (track_id, camera_id, confidence)
                VALUES (%s, %s, %s)
            """, (track_id, camera_id, confidence))
        except Exception as e:
            print(f"[DB] Insert error: {e}")

    def update_dwell(self, track_id, dwell_seconds):
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                UPDATE queue_events
                SET dwell_seconds = %s
                WHERE track_id = %s
                AND id = (
                    SELECT id FROM queue_events
                    WHERE track_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (dwell_seconds, track_id, track_id))
        except Exception as e:
            print(f"[DB] Update error: {e}")

    def close(self):
        if not self.enabled:
            return
        try:
            self.cursor.close()
            self.conn.close()
        except Exception:
            pass
