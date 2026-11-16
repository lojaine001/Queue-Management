"""Day recap (customers, trend, peak hour, waits, lanes, alerts, equipment) and the
background traffic / demographic statistics."""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

from prediction.forecast_state import freshness

from ..db import _conn
from ..errors import BadRequest
from ..repositories import dashboard, entrance, predictions, service
from ..settings import BUCKET_MIN
from .kpis import kpi_queue, kpi_wait

EQUIPMENT_ORDER = ["trolley", "store_basket"]
EQUIPMENT_COLORS = {"trolley": "#06b6d4", "store_basket": "#a855f7"}
EQUIPMENT_LABELS = {"trolley": "Trolley", "store_basket": "Store basket"}


def day_recap(date) -> dict:
    """
    The summary for a given day (date = 'YYYY-MM-DD'), or for the most recent
    day with data when no date is given.
    """
    requested_date = None
    if date:
        try:
            requested_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise BadRequest("date must be in YYYY-MM-DD format")

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            ref_date = requested_date or entrance.latest_day(cur)
            display_date = ref_date.strftime("%d %b %Y") if ref_date else datetime.now().strftime("%d %b %Y")

            total = entrance.customers_on(cur, ref_date)

            # Yesterday's total, for the vs-yesterday comparison
            yesterday_total = entrance.customers_on(cur, ref_date - timedelta(days=1)) if ref_date else None

            # 7-day trend ending on ref_date, for the Clients Total sparkline
            trend_rows = entrance.customers_per_day(cur, ref_date - timedelta(days=6), ref_date) if ref_date else []

            peak_row = entrance.busiest_hour(cur, ref_date)
            avg_min = service.average_wait_min(cur, ref_date)
            lanes_today = service.lanes_used(cur, ref_date)
            busiest_lane = service.busiest_lane(cur, ref_date)
            # Alert minutes (from ensemble predictions)
            alert_minutes = predictions.alert_slots_on(cur, ref_date) * BUCKET_MIN
            equip_rows = entrance.equipment_counts(cur, ref_date)
            # Hourly entries: computed per requested day, not from dashboard_state,
            # which only ever holds today's cache and would be wrong for past dates.
            hourly_rows = entrance.customers_per_hour(cur, ref_date)
            end_date = ref_date + timedelta(days=1) if ref_date else None
            wait_stats = kpi_wait(cur, ref_date, end_date)
            queue_stats = kpi_queue(cur, ref_date, end_date)

    peak_hour = peak_row["hour"].strftime("%H:%M") if peak_row and peak_row["hour"] else None
    peak_end = (peak_row["hour"] + timedelta(hours=1)).strftime("%H:%M") if peak_row and peak_row["hour"] else None
    peak_count = int(peak_row["cnt"]) if peak_row else 0
    avg_wait_min = float(avg_min) if avg_min else 0.0

    trend_by_day = {r["day"]: int(r["cnt"]) for r in trend_rows}
    trend_7d = []
    if ref_date:
        for i in range(6, -1, -1):
            d = ref_date - timedelta(days=i)
            trend_7d.append({"date": d.strftime("%Y-%m-%d"), "count": trend_by_day.get(d, 0)})

    denom = total or 1
    equipment = []
    for key in EQUIPMENT_ORDER:
        row = next((r for r in equip_rows if r["equipment_type"] == key), None)
        count = int(row["cnt"]) if row else 0
        equipment.append({
            "type":    key,
            "label":   EQUIPMENT_LABELS[key],
            "count":   count,
            "percent": round(count / denom * 100),
            "color":   EQUIPMENT_COLORS[key],
        })

    peak_hourly_cnt = max((int(r["cnt"]) for r in hourly_rows), default=0)
    hourly_entries = [
        {
            "hour":    r["hour"].strftime("%H:00"),
            "count":   int(r["cnt"]),
            "is_peak": int(r["cnt"]) == peak_hourly_cnt and peak_hourly_cnt > 0,
        }
        for r in hourly_rows
    ]

    vs_yesterday_pct = (
        round((total - yesterday_total) / yesterday_total * 100)
        if yesterday_total else None
    )
    peak_pct_of_total = round(peak_count / total * 100) if total else None

    return {
        "date":                display_date,
        "total_customers":     total,
        "vs_yesterday_pct":    vs_yesterday_pct,
        "trend_7d":            trend_7d,
        "avg_wait_min":        avg_wait_min,
        "peak_hour":           peak_hour,
        "peak_hour_end":       peak_end,
        "peak_count":          peak_count,
        "peak_pct_of_total":   peak_pct_of_total,
        "equipment":           equipment,
        "lanes_today":         lanes_today,
        "busiest_lane":        busiest_lane,
        "alert_minutes":       alert_minutes,
        "entries_by_hour":     hourly_entries, "wait_stats": wait_stats, "queue_stats": queue_stats,
    }


def statistics_summary() -> dict:
    """Background demographic/traffic snapshot, with its own freshness timestamp."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            row = dashboard.statistics(cur)
    if not row:
        return {"available": False, "stale": True}
    return {"available": True, **row, **freshness({"updated_at": row["updated_at"]})}
