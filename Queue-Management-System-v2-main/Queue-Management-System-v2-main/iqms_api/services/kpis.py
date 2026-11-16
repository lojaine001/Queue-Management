"""KPIs shared by the day recap and the export: checkout wait and queue length,
as an overall value plus a series with every hour/day of the range (zero-filled)."""
from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from ..repositories import kpis as repo
from ..settings import STORE_TZ


def _series_labels(bucket: str, start_date, end_date) -> list[str]:
    if bucket == "hour":
        return [f"{h:02d}:00" for h in range(24)]
    labels = []
    d = start_date
    while d < end_date:
        labels.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return labels


def kpi_wait(cur, start_date, end_date, bucket="hour"):
    """Average/max checkout wait (minutes), from service_events.
    start_date inclusive, end_date exclusive (both dates, store TZ).
    bucket='hour' (used for a single day) or 'day' (used for week/month).
    by_hour/by_day always covers every hour/day in range, zero-filled --
    matches what the export already does, so a sparse day doesn't
    render as a single bar stretched across the whole chart."""
    key = "by_hour" if bucket == "hour" else "by_day"
    if start_date is None:
        return {"avg_wait_min": None, "max_wait_min": None, "wait_source": "service_events", key: []}
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"
    label_key = "hour" if bucket == "hour" else "date"

    by_label = {r["bucket"].strftime(fmt): r for r in repo.checkout_wait_by_bucket(cur, trunc_unit, start_date, end_date)}
    overall = repo.checkout_wait_overall(cur, start_date, end_date)

    series = []
    for label in _series_labels(bucket, start_date, end_date):
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

    by_label = {r["bucket"].strftime(fmt): r for r in repo.checkout_queue_by_bucket(cur, trunc_unit, start_date, end_date)}
    overall = repo.checkout_queue_overall(cur, start_date, end_date)

    peak_time = None
    if overall.get("peak_waiting") is not None:
        peak_row = repo.first_time_queue_reached(cur, start_date, end_date, overall["peak_waiting"])
        if peak_row:
            peak_fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d %H:%M"
            peak_time = peak_row["timestamp"].astimezone(ZoneInfo(STORE_TZ)).strftime(peak_fmt)

    series = []
    for label in _series_labels(bucket, start_date, end_date):
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
