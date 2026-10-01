import os
import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

def check_day(cur, label, date_str):
    cur.execute("""
        WITH ordered AS (
            SELECT camera_id, "timestamp",
                   LAG("timestamp") OVER (PARTITION BY camera_id ORDER BY "timestamp") AS prev_ts
            FROM entrance_events
            WHERE "timestamp"::date = %s
        )
        SELECT camera_id, COUNT(*) AS total_rows,
               COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("timestamp" - prev_ts)) < 1.5) AS near_dupe_pairs
        FROM ordered
        GROUP BY camera_id
        ORDER BY camera_id;
    """, (date_str,))
    rows = cur.fetchall()
    print(f"\n{label} ({date_str}):")
    if not rows:
        print("  No rows at all for this date.")
    for r in rows:
        pct = (r["near_dupe_pairs"] / r["total_rows"] * 100) if r["total_rows"] else 0
        print(f"  {r['camera_id']}: {r['total_rows']} rows, {r['near_dupe_pairs']} within 1.5s of the previous one ({pct:.1f}%)")

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    check_day(cur, "SUSPECT DAY", "2026-09-13")
    check_day(cur, "Day before (baseline)", "2026-09-12")
    check_day(cur, "Day after (baseline)", "2026-09-14")

conn.close()
