"""KPI queries shared by the day recap and the export: checkout wait and queue length."""
from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from .settings import CHECKOUT_CAM_ID, STORE_TZ


def kpi_wait(cur, start_date, end_date, bucket="hour"):
    """Average/max checkout wait (minutes), from service_events.
    start_date inclusive, end_date exclusive (both dates, store TZ).
    bucket='hour' (used for a single day) or 'day' (used for week/month).
    by_hour/by_day always covers every hour/day in range, zero-filled --
    matches what _export_rows() already does, so a sparse day doesn't
    render as a single bar stretched across the whole chart."""
    key = "by_hour" if bucket == "hour" else "by_day"
    if start_date is None:
        return {"avg_wait_min": None, "max_wait_min": None, "wait_source": "service_events", key: []}
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"
    label_key = "hour" if bucket == "hour" else "date"

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    by_label = {r["bucket"].strftime(fmt): r for r in cur.fetchall()}

    cur.execute("""
        SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    overall = cur.fetchone() or {}

    if bucket == "hour":
        labels = [f"{h:02d}:00" for h in range(24)]
    else:
        labels = []
        d = start_date
        while d < end_date:
            labels.append(d.strftime(fmt))
            d += timedelta(days=1)

    series = []
    for label in labels:
        r = by_label.get(label)
        series.append({
            label_key: label,
            "avg_wait_min": float(r["avg_wait_min"]) if r and r["avg_wait_min"] is not None else None,
            "max_wait_min": float(r["max_wait_min"]) if r and r["max_wait_min"] is not None else None,
        })

    return {
        "avg_wait_min": float(overall["avg_wait_min"]) if overall.get("avg_wait_min") is not None else None,
        "max_wait_min": float(overall["max_wait_min"]) if overall.get("max_wait_min") is not None else None,
        "wait_source": "service_events",
        key: series,
    }


def kpi_queue(cur, start_date, end_date, bucket="hour"):
    """Average/peak people waiting at checkout, from queue_state_snapshots.
    Same start/end/bucket contract as kpi_wait, same zero-filled series."""
    key = "by_hour" if bucket == "hour" else "by_day"
    if start_date is None:
        return {"avg_waiting": None, "peak_waiting": None, "peak_time": None, key: []}
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"
    label_key = "hour" if bucket == "hour" else "date"

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting,
               MAX(queue_count) AS max_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    by_label = {r["bucket"].strftime(fmt): r for r in cur.fetchall()}

    cur.execute("""
        SELECT ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting, MAX(queue_count) AS peak_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    overall = cur.fetchone() or {}

    peak_time = None
    if overall.get("peak_waiting") is not None:
        cur.execute("""
            SELECT timestamp FROM queue_state_snapshots
            WHERE camera_id = %s
              AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
              AND queue_count = %s
            ORDER BY timestamp ASC LIMIT 1
        """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date, overall["peak_waiting"]))
        peak_row = cur.fetchone()
        if peak_row:
            peak_fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d %H:%M"
            peak_time = peak_row["timestamp"].astimezone(ZoneInfo(STORE_TZ)).strftime(peak_fmt)

    if bucket == "hour":
        labels = [f"{h:02d}:00" for h in range(24)]
    else:
        labels = []
        d = start_date
        while d < end_date:
            labels.append(d.strftime(fmt))
            d += timedelta(days=1)

    series = []
    for label in labels:
        r = by_label.get(label)
        series.append({
            label_key: label,
            "avg_waiting": float(r["avg_waiting"]) if r and r["avg_waiting"] is not None else None,
            "max_waiting": int(r["max_waiting"]) if r and r["max_waiting"] is not None else None,
        })

    return {
        "avg_waiting": float(overall["avg_waiting"]) if overall.get("avg_waiting") is not None else None,
        "peak_waiting": int(overall["peak_waiting"]) if overall.get("peak_waiting") is not None else None,
        "peak_time": peak_time,
        key: series,
    }
