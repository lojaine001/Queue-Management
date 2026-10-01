import os
import random
import sys
from datetime import date, timedelta

import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

TABLE_NAME = "entrance_events"

# Set to False (or pass "commit" as an arg) to actually write to the DB.
DRY_RUN = "--commit" not in sys.argv

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

DATE_MAP = [
    ("2026-09-08", "2026-09-15"),
    ("2026-09-09", "2026-09-16"),
    ("2026-09-10", "2026-09-17"),
]

COLUMNS = [
    "track_id", "gender", "age_estimate", "confidence", "camera_id",
    "timestamp", "dwell_seconds", "has_bag", "has_caddy", "is_group",
    "group_id", "active_head_tracks_in_lane", "equipment_type",
]

def backfill_day(cur, source_date, target_date):
    cur.execute(
        f'SELECT {", ".join(COLUMNS)} FROM {TABLE_NAME} WHERE "timestamp"::date = %s',
        (source_date,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"  No source rows found for {source_date}, skipping.")
        return 0

    day_shift = timedelta(days=(date.fromisoformat(target_date) - date.fromisoformat(source_date)).days)

    new_rows = []
    for r in rows:
        r = dict(r)
        jitter = timedelta(seconds=random.uniform(-60, 60))
        r["timestamp"] = r["timestamp"] + day_shift + jitter
        new_rows.append(tuple(r[c] for c in COLUMNS))

    if DRY_RUN:
        print(f"  [DRY RUN] Would insert {len(new_rows)} rows for {target_date} (from {source_date}).")
        print(f"  Example row: {new_rows[0]}")
    else:
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} ({', '.join(COLUMNS)})
            VALUES ({', '.join(['%s'] * len(COLUMNS))})
        """
        psycopg2.extras.execute_batch(cur, insert_sql, new_rows)
        print(f"  Inserted {len(new_rows)} rows for {target_date} (from {source_date}).")

    return len(new_rows)

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    total = 0
    for source_date, target_date in DATE_MAP:
        print(f"Processing {source_date} -> {target_date}")
        total += backfill_day(cur, source_date, target_date)

    if DRY_RUN:
        print(f"\nDRY RUN complete. {total} rows would be inserted total.")
        print("Re-run with --commit to actually write to the database.")
    else:
        conn.commit()
        print(f"\nCommitted. {total} rows inserted total.")

conn.close()
