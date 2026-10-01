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

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute("""
        WITH ordered AS (
            SELECT camera_id, "timestamp",
                   LAG("timestamp") OVER (PARTITION BY camera_id ORDER BY "timestamp") AS prev_ts
            FROM queue_state_snapshots
            WHERE "timestamp" >= NOW() - INTERVAL '15 minutes'
        )
        SELECT camera_id, "timestamp", prev_ts,
               EXTRACT(EPOCH FROM ("timestamp" - prev_ts)) AS gap_sec
        FROM ordered
        WHERE prev_ts IS NOT NULL
        ORDER BY camera_id, "timestamp";
    """)
    rows = cur.fetchall()

if not rows:
    print("No rows in the last 15 minutes -- store closed, camera likely idle. "
          "Widen the interval (e.g. to last 6 hours) and rerun to check business-hours data instead.")
else:
    print(f"{len(rows)} rows. gap_sec distribution per camera:\n")
    by_cam = {}
    for r in rows:
        by_cam.setdefault(r["camera_id"], []).append(r["gap_sec"])
    for cam, gaps in by_cam.items():
        under_1s = sum(1 for g in gaps if g < 1.0)
        print(f"  {cam}: n={len(gaps)}  min={min(gaps):.2f}s  max={max(gaps):.2f}s  "
              f"gaps<1s={under_1s} ({100*under_1s/len(gaps):.0f}%)")
        print(f"    sample gaps: {[round(g,2) for g in gaps[:15]]}")

conn.close()
