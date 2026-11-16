"""entrance_events: one row per person seen at the entrance. A customer is a person
who stayed at least 10 s; simulator cameras (SIM_*) are left out."""
from __future__ import annotations

from .day_filter import on_day_or_since_today


def latest_day(cur):
    """The (store-local) day of the newest real entrance event, or None when there is none."""
    cur.execute("""
        SELECT DATE(timestamp) AS day
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%%'
        ORDER BY timestamp DESC LIMIT 1
    """)
    row = cur.fetchone()
    return row["day"] if row else None


def customers_on(cur, day) -> int:
    """Customers on `day` (None: since today 00:00)."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT COUNT(*) AS total
        FROM entrance_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds >= 10
    """, params)
    return int((cur.fetchone() or {}).get("total") or 0)


def customers_per_day(cur, first_day, last_day) -> list[dict]:
    """Customers per day from first_day to last_day (both included); days without any are absent."""
    cur.execute("""
        SELECT DATE(timestamp) AS day, COUNT(*) AS cnt
        FROM entrance_events
        WHERE DATE(timestamp) BETWEEN %s AND %s
          AND camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds >= 10
        GROUP BY 1 ORDER BY 1 ASC
    """, (first_day, last_day))
    return cur.fetchall()


def busiest_hour(cur, day):
    """The hour (Paris time) with the most customers on `day`, and its count (or None)."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour, COUNT(*) AS cnt
        FROM entrance_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds >= 10
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """, params)
    return cur.fetchone()


def customers_per_hour(cur, day) -> list[dict]:
    """Customers per hour (Paris time) on `day`; hours without any are absent."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour,
               COUNT(*) AS cnt
        FROM entrance_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
          AND dwell_seconds >= 10
        GROUP BY 1 ORDER BY 1 ASC
    """, params)
    return cur.fetchall()


def equipment_counts(cur, day) -> list[dict]:
    """People per equipment type (trolley, store_basket, ...) on `day`, any dwell time."""
    condition, params = on_day_or_since_today("timestamp", day)
    cur.execute(f"""
        SELECT equipment_type, COUNT(*) AS cnt
        FROM entrance_events
        WHERE {condition}
          AND camera_id NOT LIKE 'SIM_%%'
          AND equipment_type IS NOT NULL
          AND equipment_type != 'none'
        GROUP BY equipment_type
    """, params)
    return cur.fetchall()


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
