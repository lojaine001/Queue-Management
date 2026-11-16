"""service_events: one row per customer served at a checkout lane (time spent, lane).
Simulator cameras (SIM_*) are left out."""
from __future__ import annotations

from .day_filter import on_day_or_since_today


def average_wait_min(cur, day):
    """Average time spent at checkout on `day`, in minutes (None when nobody was served)."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_min
        FROM service_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
    """, params)
    row = cur.fetchone()
    return row["avg_min"] if row else None


def lanes_used(cur, day) -> int:
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT COUNT(DISTINCT lane_id) AS lanes_count
        FROM service_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
    """, params)
    row = cur.fetchone()
    return int(row["lanes_count"]) if row and row["lanes_count"] else 0


def busiest_lane(cur, day):
    """The lane that served the most customers on `day` (or None)."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT lane_id, COUNT(*) AS cnt
        FROM service_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
        GROUP BY lane_id ORDER BY cnt DESC LIMIT 1
    """, params)
    row = cur.fetchone()
    return row["lane_id"] if row else None
