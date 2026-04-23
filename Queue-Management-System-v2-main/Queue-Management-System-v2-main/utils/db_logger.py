import os

import psycopg2
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


class DBLogger:
    def __init__(self):
        self.enabled = False
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST",     "localhost"),
                port=int(os.getenv("DB_PORT", 5432)),
                dbname=os.getenv("DB_NAME",   "iqms"),
                user=os.getenv("DB_USER",     "postgres"),
                password=os.getenv("DB_PASSWORD", "0000"),
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            self._create_table()
            self.enabled = True
            print("[DB] Connected to PostgreSQL ✓")
        except Exception as e:
            print(f"[DB] WARNING: Could not connect to PostgreSQL — running without DB. Error: {e}")

    def _create_table(self):
        # Enable TimescaleDB (no-op if already enabled)
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

        # ── entrance_events ──────────────────────────────────────────────────
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS entrance_events (
                id            BIGSERIAL,
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                track_id      INT,
                gender        VARCHAR(20),
                age_estimate  FLOAT,
                confidence    FLOAT,
                camera_id     VARCHAR(100),
                dwell_seconds FLOAT DEFAULT 0
            )
        """)

        # Migrate existing column from TIMESTAMP to TIMESTAMPTZ if needed
        self.cursor.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'entrance_events'
                    AND column_name = 'timestamp'
                    AND data_type = 'timestamp without time zone'
                ) THEN
                    ALTER TABLE entrance_events
                    ALTER COLUMN timestamp TYPE TIMESTAMPTZ
                    USING timestamp AT TIME ZONE 'UTC';
                END IF;
            END $$;
        """)

        # Add has_bag column if missing (existing deployments)
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'entrance_events'
                    AND column_name = 'has_bag'
                ) THEN
                    ALTER TABLE entrance_events ADD COLUMN has_bag BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            SELECT create_hypertable('entrance_events', 'timestamp',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

        # ── queue_state_snapshots ────────────────────────────────────────────
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS queue_state_snapshots (
                id             BIGSERIAL,
                timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                camera_id      VARCHAR(100),
                queue_count    INT          NOT NULL DEFAULT 0,
                avg_dwell_sec  FLOAT        DEFAULT 0,
                max_dwell_sec  FLOAT        DEFAULT 0,
                active_lanes   INT          DEFAULT 2
            )
        """)

        self.cursor.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'queue_state_snapshots'
                    AND column_name = 'timestamp'
                    AND data_type = 'timestamp without time zone'
                ) THEN
                    ALTER TABLE queue_state_snapshots
                    ALTER COLUMN timestamp TYPE TIMESTAMPTZ
                    USING timestamp AT TIME ZONE 'UTC';
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            SELECT create_hypertable('queue_state_snapshots', 'timestamp',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

        # ── service_events ───────────────────────────────────────────────────
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_events (
                id             BIGSERIAL,
                timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                camera_id      VARCHAR(100),
                track_id       INT,
                total_dwell_sec FLOAT       DEFAULT 0
            )
        """)

        self.cursor.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'service_events'
                    AND column_name = 'timestamp'
                    AND data_type = 'timestamp without time zone'
                ) THEN
                    ALTER TABLE service_events
                    ALTER COLUMN timestamp TYPE TIMESTAMPTZ
                    USING timestamp AT TIME ZONE 'UTC';
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            SELECT create_hypertable('service_events', 'timestamp',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

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
        """Insert a periodic queue-state snapshot used by ensemble_predict for dynamic wait estimation."""
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

    def log_service_event(self, camera_id, track_id, total_dwell_sec):
        """Record a service completion (customer leaves). Used for future exit-ROI integration."""
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
