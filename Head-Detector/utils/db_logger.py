import os

try:
    import psycopg2
except ImportError:
    psycopg2 = None
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# entry_time is passed in as a naive local datetime-based value (see
# main.py's use of datetime.fromtimestamp()). Without pinning the session
# timezone, Postgres interprets that naive value using its own default
# session timezone instead of real local time, silently shifting every
# stored timestamp — the same bug class already fixed in
# ensemble_predict.py's _connect().
STORE_TZ = os.getenv("STORE_TZ", "Europe/Paris")


class DBLogger:
    def __init__(self):
        self.enabled = False
        if psycopg2 is None:
            print("[DB] WARNING: psycopg2 is not installed — running without DB.")
            return
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
            self.cursor.execute("SET timezone = %s", (STORE_TZ,))
            self._create_table()
            self.enabled = True
            print(f"[DB] Connected to PostgreSQL ✓ (timezone={STORE_TZ})")
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
                dwell_seconds FLOAT DEFAULT 0,
                has_bag       BOOLEAN DEFAULT FALSE,
                active_head_tracks_in_lane INT DEFAULT 0
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

        # Add active_head_tracks_in_lane column if missing (existing deployments)
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'entrance_events'
                    AND column_name = 'active_head_tracks_in_lane'
                ) THEN
                    ALTER TABLE entrance_events
                    ADD COLUMN active_head_tracks_in_lane INT DEFAULT 0;
                END IF;
            END $$;
        """)

        # Add equipment_type column if missing (existing deployments)
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'entrance_events'
                    AND column_name = 'equipment_type'
                ) THEN
                    ALTER TABLE entrance_events
                    ADD COLUMN equipment_type VARCHAR(30) DEFAULT 'none';
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            SELECT create_hypertable('entrance_events', 'timestamp',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

        # ── queue_state_snapshots ────────────────────────────────────────────
        # Periodic snapshot of live queue state (every ~10 s from the pipeline)
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

        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'queue_state_snapshots'
                    AND column_name = 'lane_counts'
                ) THEN
                    ALTER TABLE queue_state_snapshots
                    ADD COLUMN lane_counts JSONB DEFAULT NULL;
                END IF;
            END $$;
        """)

        # ── service_events ───────────────────────────────────────────────────
        # Reserved for exit-ROI events (track lost / customer served)
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
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'service_events'
                    AND column_name = 'equipment_type'
                ) THEN
                    ALTER TABLE service_events
                    ADD COLUMN equipment_type VARCHAR(30) DEFAULT 'none';
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'service_events'
                    AND column_name = 'active_head_tracks_in_lane'
                ) THEN
                    ALTER TABLE service_events
                    ADD COLUMN active_head_tracks_in_lane INT DEFAULT NULL;
                END IF;
            END $$;
        """)

        self.cursor.execute("""
            SELECT create_hypertable('service_events', 'timestamp',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

    def insert_entrance(self, track_id, gender, age, confidence, camera_id,
                        dwell_seconds=None, entry_time=None, has_bag=False,
                        active_head_tracks_in_lane=0, equipment_type="none"):
        if not self.enabled:
            return
        has_bag = equipment_type != "none"
        try:
            if entry_time is not None:
                self.cursor.execute("""
                    INSERT INTO entrance_events
                        (timestamp, track_id, gender, age_estimate, confidence,
                         camera_id, dwell_seconds, has_bag, active_head_tracks_in_lane,
                         equipment_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (entry_time, track_id, gender, age, confidence,
                      camera_id, dwell_seconds, has_bag, active_head_tracks_in_lane,
                      equipment_type))
            else:
                self.cursor.execute("""
                    INSERT INTO entrance_events
                        (track_id, gender, age_estimate, confidence,
                         camera_id, dwell_seconds, has_bag, active_head_tracks_in_lane,
                         equipment_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (track_id, gender, age, confidence,
                      camera_id, dwell_seconds, has_bag, active_head_tracks_in_lane,
                      equipment_type))
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

    def log_queue_snapshot(self, camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes=2, lane_counts=None):
        """Insert a periodic queue-state snapshot used by ensemble_predict for dynamic wait estimation."""
        if not self.enabled:
            print("[DB] Snapshot skipped — db not enabled")
            return
        try:
            import json
            self.cursor.execute("""
                INSERT INTO queue_state_snapshots
                    (camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes, lane_counts)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes,
                  json.dumps(lane_counts) if lane_counts is not None else None))
            print(f"[DB] Snapshot saved | cam={camera_id} | queue_count={queue_count} | lane_counts={lane_counts}")
        except Exception as e:
            print(f"[DB] Snapshot error: {e}")

    def log_service_event(self, camera_id, track_id, total_dwell_sec, lane_id=None,
                          equipment_type="none", active_head_tracks_in_lane=None):
        """Record a service completion (customer leaves).

        active_head_tracks_in_lane: number of OTHER confirmed tracks in the same lane
        at the moment this track died. 0 means the person was alone → dwell ≈ pure
        service time. NULL for historical rows predating this column.
        """
        if not self.enabled:
            return
        try:
            self.cursor.execute("""
                INSERT INTO service_events
                    (camera_id, track_id, total_dwell_sec, lane_id, equipment_type,
                     active_head_tracks_in_lane)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (camera_id, track_id, total_dwell_sec, lane_id, equipment_type,
                  active_head_tracks_in_lane))
        except Exception as e:
            print(f"[DB] Service event error: {e}")

    def get_period_summary(self, camera_id, interval_minutes=15):
        """Return (track_count, avg_dwell_sec) for tracks inserted in the last interval_minutes."""
        if not self.enabled:
            return None, None
        try:
            self.cursor.execute("""
                SELECT COUNT(*), AVG(dwell_seconds)
                FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= NOW() - INTERVAL '%s minutes'
            """, (camera_id, interval_minutes))
            row = self.cursor.fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
            avg_dwell = float(row[1]) if row and row[1] is not None else 0.0
            return count, avg_dwell
        except Exception as e:
            print(f"[DB] Summary query error: {e}")
            return None, None

    def close(self):
        if not self.enabled:
            return
        try:
            self.cursor.close()
            self.conn.close()
        except Exception:
            pass
