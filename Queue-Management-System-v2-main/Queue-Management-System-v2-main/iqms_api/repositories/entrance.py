"""entrance_events: one row per person seen at the entrance."""
from __future__ import annotations


def entries_per_3_minutes(cur, since: str) -> list[dict]:
    """Customers (stayed at least 10 s) per 3-minute slot from `since` ago until now.
    `since` is a PostgreSQL interval such as '6 hours' or '2 days'."""
    cur.execute("""
        SELECT
            time_bucket('3 minutes', timestamp) AS slot,
            COUNT(*) AS entries
        FROM entrance_events
        WHERE timestamp >= NOW() - %s::interval
          AND timestamp <= NOW()
          AND camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds >= 10
        GROUP BY 1
        ORDER BY 1 ASC
    """, (since,))
    return cur.fetchall()
