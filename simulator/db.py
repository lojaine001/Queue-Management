import os

import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME",     "iqms"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
}

class SimDB:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.camera_id = f"SIM_{scenario_name}"
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            self._init_db()
            print(f"[SimDB] Connected to PostgreSQL and initialized tables for scenario {scenario_name} ✓")
        except Exception as e:
            print(f"[SimDB] Error connecting to DB: {e}")
            try:
                if self.conn:
                    self.conn.close()
            except Exception:
                pass
            self.conn = None

    def _create_hypertable(self, table_name: str, time_column: str = "timestamp"):
        self.cursor.execute(f"""
            SELECT create_hypertable('{table_name}', '{time_column}',
                if_not_exists => TRUE,
                migrate_data  => TRUE);
        """)

    def _drop_legacy_primary_key(self, table_name: str):
        self.cursor.execute(f"""
            DO $$
            DECLARE
                pk_name text;
            BEGIN
                SELECT conname INTO pk_name
                FROM pg_constraint
                WHERE conrelid = '{table_name}'::regclass
                  AND contype = 'p'
                LIMIT 1;

                IF pk_name IS NOT NULL THEN
                    EXECUTE format('ALTER TABLE {table_name} DROP CONSTRAINT %I', pk_name);
                END IF;
            END $$;
        """)

    def _migrate_timestamp_if_needed(self, table_name: str, time_column: str = "timestamp"):
        self.cursor.execute(f"""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    AND column_name = '{time_column}'
                    AND data_type = 'timestamp without time zone'
                ) THEN
                    ALTER TABLE {table_name}
                    ALTER COLUMN {time_column} TYPE TIMESTAMPTZ
                    USING {time_column} AT TIME ZONE 'UTC';
                END IF;
            END $$;
        """)

    def _init_db(self):
        # Ensure TimescaleDB is enabled
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        
        # entrance_events
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS entrance_events (
                id            BIGSERIAL,
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                track_id      INT,
                gender        VARCHAR(20),
                age_estimate  FLOAT,
                has_bag       BOOLEAN      DEFAULT FALSE,
                has_caddy     BOOLEAN      DEFAULT FALSE,
                is_group      BOOLEAN      DEFAULT FALSE,
                group_id      VARCHAR(100),
                confidence    FLOAT,
                camera_id     VARCHAR(100),
                dwell_seconds FLOAT DEFAULT 0
            )
        """)
        self._drop_legacy_primary_key("entrance_events")
        self._migrate_timestamp_if_needed("entrance_events")
        self._create_hypertable("entrance_events")
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'entrance_events' AND column_name = 'has_bag'
                ) THEN
                    ALTER TABLE entrance_events
                        ADD COLUMN has_bag BOOLEAN DEFAULT FALSE,
                        ADD COLUMN has_caddy BOOLEAN DEFAULT FALSE,
                        ADD COLUMN is_group BOOLEAN DEFAULT FALSE,
                        ADD COLUMN group_id VARCHAR(100);
                END IF;
            END $$;
        """)

        # queue_state_snapshots
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
        self._drop_legacy_primary_key("queue_state_snapshots")
        self._migrate_timestamp_if_needed("queue_state_snapshots")
        self._create_hypertable("queue_state_snapshots")
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'queue_state_snapshots' AND column_name = 'max_dwell_sec'
                ) THEN
                    ALTER TABLE queue_state_snapshots
                        ADD COLUMN max_dwell_sec FLOAT DEFAULT 0;
                END IF;
            END $$;
        """)

        # service_events
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_events (
                id             BIGSERIAL,
                timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                camera_id      VARCHAR(100),
                track_id       INT,
                lane_id        INT,
                total_dwell_sec FLOAT       DEFAULT 0
            )
        """)
        self._drop_legacy_primary_key("service_events")
        self._migrate_timestamp_if_needed("service_events")
        self._create_hypertable("service_events")
        self.cursor.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'service_events' AND column_name = 'lane_id'
                ) THEN
                    ALTER TABLE service_events
                        ADD COLUMN lane_id INT;
                END IF;
            END $$;
        """)

    def insert_entrance(self, person):
        if not self.conn: return
        try:
            self.cursor.execute("""
                INSERT INTO entrance_events
                    (timestamp, track_id, gender, age_estimate, has_bag, has_caddy,
                     is_group, group_id, camera_id, dwell_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                person.entry_time,
                person.track_id,
                person.gender,
                person.age,
                person.has_bag,
                person.has_caddy,
                person.is_group,
                person.group_id,
                self.camera_id,
                0
            ))
        except Exception as e:
            print(f"[SimDB] Insert entrance error: {e}")

    def update_dwell(self, person):
        if not self.conn: return
        try:
            self.cursor.execute("""
                UPDATE entrance_events
                SET dwell_seconds = %s
                WHERE track_id = %s AND camera_id = %s
                AND id = (
                    SELECT id FROM entrance_events
                    WHERE track_id = %s AND camera_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (person.dwell_seconds, person.track_id, self.camera_id, person.track_id, self.camera_id))
        except Exception as e:
            print(f"[SimDB] Update dwell error: {e}")

    def log_snapshot(self, timestamp, queue_count, avg_dwell, max_dwell, active_lanes):
        if not self.conn: return
        try:
            self.cursor.execute("""
                INSERT INTO queue_state_snapshots
                    (timestamp, camera_id, queue_count, avg_dwell_sec, max_dwell_sec, active_lanes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (timestamp, self.camera_id, queue_count, avg_dwell, max_dwell, active_lanes))
        except Exception as e:
            print(f"[SimDB] Snapshot error: {e}")

    def log_service_event(self, person):
        if not self.conn: return
        try:
            self.cursor.execute("""
                INSERT INTO service_events
                    (timestamp, camera_id, track_id, lane_id, total_dwell_sec)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                person.service_end_time,
                self.camera_id,
                person.track_id,
                person.lane_id,
                person.dwell_seconds
            ))
        except Exception as e:
            print(f"[SimDB] Service event error: {e}")

    def close(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()
