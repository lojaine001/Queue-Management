import os

import psycopg2
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# entry_time is passed in as a naive local datetime.now()-based value (see
# queue_management_v2.py). Without pinning the session timezone, Postgres
# interprets that naive value using its own default session timezone instead
# of real local time, silently shifting every stored timestamp -- the same
# bug class already fixed in ensemble_predict.py's _connect().
STORE_TZ = os.getenv("STORE_TZ", "Europe/Paris")


class MissingTables(RuntimeError):
    """The database was never migrated: a setup error, so the camera stops instead of
    counting people without saving them (unlike a database outage)."""


class DBLogger:
    def __init__(self):
        self.enabled = False
        password = os.getenv("DB_PASSWORD", "")
        if not password or password == "change-me":
            # A missing setting is a configuration error, not a database outage:
            # stop instead of silently counting people without saving them.
            raise RuntimeError("DB_PASSWORD is not set. Add it to .env (see .env.example).")
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST",     "localhost"),
                port=int(os.getenv("DB_PORT", 5432)),
                dbname=os.getenv("DB_NAME",   "iqms"),
                user=os.getenv("DB_USER",     "postgres"),
                password=password,
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            self.cursor.execute("SET timezone = %s", (STORE_TZ,))
            self._require_tables()
            self.enabled = True
            print(f"[DB] Connected to PostgreSQL [OK] (timezone={STORE_TZ})")
        except MissingTables:
            raise
        except Exception as e:
            print(f"[DB] WARNING: Could not connect to PostgreSQL -- running without DB. Error: {e}")

    # The tables come from the database migrations (db/); see db/README.md.
    REQUIRED_TABLES = ("entrance_events", "queue_state_snapshots", "service_events", "lane_events")

    def _require_tables(self):
        self.cursor.execute(
            "SELECT t FROM unnest(%s::text[]) AS t WHERE to_regclass('public.' || t) IS NULL",
            (list(self.REQUIRED_TABLES),),
        )
        missing = [row[0] for row in self.cursor.fetchall()]
        if missing:
            raise MissingTables(
                f"Database table(s) missing: {', '.join(missing)}; run "
                "`alembic -c db/alembic.ini upgrade head` from the project root "
                "(on Docker the migrate service does it)."
            )

    def insert_entrance(self, track_id, gender, age, confidence, camera_id,
                        dwell_seconds=None, entry_time=None, has_bag=False):
        if not self.enabled:
            return
        try:
            if entry_time is not None:
                self.cursor.execute("""
                    INSERT INTO entrance_events
                        (timestamp, track_id, gender, age_estimate, confidence,
                         camera_id, dwell_seconds, has_bag)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (entry_time, track_id, gender, age, confidence,
                      camera_id, dwell_seconds, has_bag))
            else:
                self.cursor.execute("""
                    INSERT INTO entrance_events
                        (track_id, gender, age_estimate, confidence,
                         camera_id, dwell_seconds, has_bag)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (track_id, gender, age, confidence,
                      camera_id, dwell_seconds, has_bag))
        except Exception as e:
            print(f"[DB] Insert error: {e}")

    def update_dwell(self, track_id, dwell_seconds):
        if not self.enabled:
            return
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

    def finalize_entrance(self, track_id, gender, age, confidence,
                          dwell_seconds=None, has_bag=False):
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                UPDATE entrance_events
                SET gender = %s,
                    age_estimate = %s,
                    confidence = %s,
                    dwell_seconds = %s,
                    has_bag = %s
                WHERE track_id = %s
                AND id = (
                    SELECT id FROM entrance_events
                    WHERE track_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (gender, age, confidence, dwell_seconds, has_bag, track_id, track_id))
        except Exception as e:
            print(f"[DB] Finalize error: {e}")

    def log_queue_snapshot(self, camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes=2):
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                INSERT INTO queue_state_snapshots
                    (camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes)
                VALUES (%s, %s, %s, %s, %s)
            """, (camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes))
        except Exception as e:
            print(f"[DB] Snapshot error: {e}")

    def get_today_entry_timestamps(self, camera_id: str = None):
        """Return today's entrance event timestamps as Unix epoch floats for counted_entry_times."""
        if not self.enabled:
            return []
        try:
            if camera_id:
                self.cursor.execute("""
                    SELECT EXTRACT(EPOCH FROM timestamp)
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE::TIMESTAMPTZ
                      AND camera_id = %s
                    ORDER BY timestamp ASC
                """, (camera_id,))
            else:
                self.cursor.execute("""
                    SELECT EXTRACT(EPOCH FROM timestamp)
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE::TIMESTAMPTZ
                    ORDER BY timestamp ASC
                """)
            return [float(row[0]) for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"[DB] get_today_entry_timestamps error: {e}")
            return []

    def get_current_lanes(self, camera_id, default=2):
        if not self.enabled:
            return default
        try:
            self.cursor.execute("""
                SELECT active_lanes FROM lane_events
                WHERE camera_id = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (camera_id,))
            row = self.cursor.fetchone()
            return row[0] if row else default
        except Exception:
            return default

    def log_service_event(self, camera_id, track_id, total_dwell_sec):
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                INSERT INTO service_events
                    (camera_id, track_id, total_dwell_sec)
                VALUES (%s, %s, %s)
            """, (camera_id, track_id, total_dwell_sec))
        except Exception as e:
            print(f"[DB] Service event error: {e}")

    def close(self):
        if not self.enabled:
            return
        try:
            self.cursor.close()
            self.conn.close()
        except Exception:
            pass
