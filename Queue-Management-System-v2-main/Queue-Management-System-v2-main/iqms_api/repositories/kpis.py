"""KPI reads for the export and the day recap. Ranges are store-local dates: start
inclusive, end exclusive. `trunc_unit` is 'hour' or 'day'. Simulator cameras (SIM_*)
are left out where several cameras are counted."""
from __future__ import annotations

from ..settings import CHECKOUT_CAM_ID, STORE_TZ


# ── Checkout wait (service_events of the checkout camera) ─────────────────────

def checkout_wait_by_bucket(cur, trunc_unit: str, start_date, end_date) -> list[dict]:
    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()


def checkout_wait_overall(cur, start_date, end_date) -> dict:
    cur.execute("""
        SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchone() or {}


# ── People waiting (queue_state_snapshots of the checkout camera) ─────────────

def checkout_queue_by_bucket(cur, trunc_unit: str, start_date, end_date) -> list[dict]:
    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting,
               MAX(queue_count) AS max_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()


def checkout_queue_overall(cur, start_date, end_date) -> dict:
    cur.execute("""
        SELECT ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting, MAX(queue_count) AS peak_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchone() or {}


def first_time_queue_reached(cur, start_date, end_date, queue_count: int):
    """The first snapshot in the range with exactly `queue_count` people waiting (or None)."""
    cur.execute("""
        SELECT timestamp FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
          AND queue_count = %s
        ORDER BY timestamp ASC LIMIT 1
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date, queue_count))
    return cur.fetchone()


# ── Entrance, lanes, alerts (export) ──────────────────────────────────────────

def entrance_counts_by_bucket(cur, trunc_unit: str, start_date, end_date) -> list[dict]:
    """Customers (stayed at least 10 s) and their equipment per bucket."""
    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               COUNT(*) AS customers,
               COUNT(*) FILTER (WHERE equipment_type = 'trolley') AS trolley,
               COUNT(*) FILTER (WHERE equipment_type = 'store_basket') AS store_basket
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%%' AND dwell_seconds >= 10
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()


def entrance_counts_by_day_and_hour(cur, start_date, end_date) -> list[dict]:
    cur.execute("""
        SELECT DATE_TRUNC('day', timestamp AT TIME ZONE %s) AS day,
               DATE_TRUNC('hour', timestamp AT TIME ZONE %s) AS hour,
               COUNT(*) AS cnt
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%%' AND dwell_seconds >= 10
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1, 2
    """, (STORE_TZ, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()


def lanes_used_by_bucket(cur, trunc_unit: str, start_date, end_date) -> list[dict]:
    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket, COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()


def lanes_used_total(cur, start_date, end_date):
    """Distinct lanes over the whole range (not a sum of per-bucket counts)."""
    cur.execute("""
        SELECT COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchone()


def alert_slots_by_bucket(cur, trunc_unit: str, start_date, end_date) -> list[dict]:
    """Forecast slots in ALERT per bucket (one slot = BUCKET_MIN minutes)."""
    cur.execute("""
        SELECT DATE_TRUNC(%s, prediction_for AT TIME ZONE %s) AS bucket, COUNT(*) AS alert_slots
        FROM queue_predictions
        WHERE status = 'ALERT'
          AND prediction_for AT TIME ZONE %s >= %s AND prediction_for AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    return cur.fetchall()
