"""KPI export: preview (JSON) and CSV download for a day, week or month."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..db import _conn
from ..kpis import kpi_queue, kpi_wait
from ..settings import BUCKET_MIN, STORE_TZ

router = APIRouter()


PERIOD_BUCKET = {"day": "hour", "week": "day", "month": "day"}


ALL_KPIS = ["customers", "avg_wait", "max_wait", "avg_waiting", "peak_waiting",
            "lanes_used", "alert_minutes", "trolley", "store_basket", "peak_hour"]


KPI_LABELS = {
    "fr": {
        "customers": "Clients total", "avg_wait": "Temps d'attente moyen (min)",
        "max_wait": "Temps d'attente max (min)", "avg_waiting": "Personnes en attente (moy.)",
        "peak_waiting": "Pic de personnes en attente", "lanes_used": "Files utilisees",
        "alert_minutes": "Temps en alerte (min)", "trolley": "Chariots", "store_basket": "Paniers",
        "peak_hour": "Heure de pointe",
    },
    "en": {
        "customers": "Total customers", "avg_wait": "Avg checkout wait (min)",
        "max_wait": "Max checkout wait (min)", "avg_waiting": "Avg people waiting",
        "peak_waiting": "Peak people waiting", "lanes_used": "Lanes used",
        "alert_minutes": "Time in alert (min)", "trolley": "Trolleys", "store_basket": "Baskets",
        "peak_hour": "Peak hour",
    },
}


def _period_bounds(period: str, date_str: str):
    try:
        ref = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    if period == "day":
        return ref, ref + timedelta(days=1)
    elif period == "week":
        monday = ref - timedelta(days=ref.weekday())
        return monday, monday + timedelta(days=7)
    elif period == "month":
        first = ref.replace(day=1)
        next_first = (first.replace(year=first.year + 1, month=1)
                      if first.month == 12 else first.replace(month=first.month + 1))
        return first, next_first
    raise HTTPException(status_code=400, detail="period must be day, week, or month")


def _export_rows(cur, start_date, end_date, bucket):
    """Returns (rows, total_row). rows: one per hour (day period) or per day
    (week/month period). total_row: sums for counts, the true overall
    avg/max from kpi_wait/kpi_queue (not an average-of-averages), and a
    true distinct-lane count over the whole range (not a sum, which would
    double-count a lane reused across multiple days)."""
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"

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
    entrance_by = {r["bucket"].strftime(fmt): r for r in cur.fetchall()}

    wait = kpi_wait(cur, start_date, end_date, bucket)
    wait_key = "by_hour" if bucket == "hour" else "by_day"
    wait_by = {(r.get("hour") or r.get("date")): r for r in wait[wait_key]}

    queue = kpi_queue(cur, start_date, end_date, bucket)
    queue_by = {(r.get("hour") or r.get("date")): r for r in queue[wait_key]}

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket, COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    lanes_by = {r["bucket"].strftime(fmt): int(r["lanes_used"]) for r in cur.fetchall()}

    cur.execute("""
        SELECT DATE_TRUNC(%s, prediction_for AT TIME ZONE %s) AS bucket, COUNT(*) AS alert_slots
        FROM queue_predictions
        WHERE status = 'ALERT'
          AND prediction_for AT TIME ZONE %s >= %s AND prediction_for AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    alert_by = {r["bucket"].strftime(fmt): int(r["alert_slots"]) * BUCKET_MIN for r in cur.fetchall()}

    peak_hour_by = {}
    if bucket == "day":
        cur.execute("""
            SELECT DATE_TRUNC('day', timestamp AT TIME ZONE %s) AS day,
                   DATE_TRUNC('hour', timestamp AT TIME ZONE %s) AS hour,
                   COUNT(*) AS cnt
            FROM entrance_events
            WHERE camera_id NOT LIKE 'SIM_%%' AND dwell_seconds >= 10
              AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
            GROUP BY 1, 2
        """, (STORE_TZ, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
        by_day_hours = {}
        for r in cur.fetchall():
            d = r["day"].strftime("%Y-%m-%d")
            by_day_hours.setdefault(d, []).append((r["hour"], int(r["cnt"])))
        for d, hrs in by_day_hours.items():
            peak_hour_by[d] = max(hrs, key=lambda x: x[1])[0].strftime("%H:%M")

    rows = []
    if bucket == "hour":
        for h in range(24):
            label = f"{h:02d}:00"
            e = entrance_by.get(label)
            w = wait_by.get(label, {})
            q = queue_by.get(label, {})
            rows.append({
                "label": label, "customers": int(e["customers"]) if e else 0,
                "avg_wait": w.get("avg_wait_min"), "max_wait": w.get("max_wait_min"),
                "avg_waiting": q.get("avg_waiting"), "peak_waiting": q.get("max_waiting"),
                "lanes_used": lanes_by.get(label, 0), "alert_minutes": alert_by.get(label, 0),
                "trolley": int(e["trolley"]) if e else 0, "store_basket": int(e["store_basket"]) if e else 0,
                "peak_hour": None,
            })
    else:
        d = start_date
        while d < end_date:
            label = d.strftime("%Y-%m-%d")
            e = entrance_by.get(label)
            w = wait_by.get(label, {})
            q = queue_by.get(label, {})
            rows.append({
                "label": label, "customers": int(e["customers"]) if e else 0,
                "avg_wait": w.get("avg_wait_min"), "max_wait": w.get("max_wait_min"),
                "avg_waiting": q.get("avg_waiting"), "peak_waiting": q.get("max_waiting"),
                "lanes_used": lanes_by.get(label, 0), "alert_minutes": alert_by.get(label, 0),
                "trolley": int(e["trolley"]) if e else 0, "store_basket": int(e["store_basket"]) if e else 0,
                "peak_hour": peak_hour_by.get(label),
            })
            d += timedelta(days=1)

    cur.execute("""
        SELECT COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (STORE_TZ, start_date, STORE_TZ, end_date))
    total_lanes_row = cur.fetchone()
    total_lanes = int(total_lanes_row["lanes_used"]) if total_lanes_row and total_lanes_row["lanes_used"] else 0

    total_row = {
        "label": "TOTAL",
        "customers": sum(r["customers"] for r in rows),
        "avg_wait": wait.get("avg_wait_min"),
        "max_wait": wait.get("max_wait_min"),
        "avg_waiting": queue.get("avg_waiting"),
        "peak_waiting": queue.get("peak_waiting"),
        "lanes_used": total_lanes,
        "alert_minutes": sum(r["alert_minutes"] for r in rows),
        "trolley": sum(r["trolley"] for r in rows),
        "store_basket": sum(r["store_basket"] for r in rows),
        "peak_hour": None,
    }

    return rows, total_row


@router.get("/export/preview")
def export_preview(period: str, date: str, kpis: Optional[str] = None, lang: str = "fr"):
    if period not in PERIOD_BUCKET:
        raise HTTPException(status_code=400, detail="period must be day, week, or month")
    start_date, end_date = _period_bounds(period, date)
    bucket = PERIOD_BUCKET[period]
    selected = kpis.split(",") if kpis else ALL_KPIS
    unknown = [k for k in selected if k not in ALL_KPIS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown KPI(s): {unknown}")

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows, total_row = _export_rows(cur, start_date, end_date, bucket)

    labels = KPI_LABELS.get(lang, KPI_LABELS["fr"])
    return {
        "period": period, "bucket": bucket, "kpis": selected,
        "labels": {k: labels[k] for k in selected},
        "rows": [{"label": r["label"], **{k: r[k] for k in selected}} for r in rows[:10]],
        "total": {"label": total_row["label"], **{k: total_row[k] for k in selected}},
        "row_count": len(rows),
    }


@router.get("/export")
def export_csv(period: str, date: str, kpis: Optional[str] = None, format: str = "csv", lang: str = "fr"):
    if format != "csv":
        raise HTTPException(status_code=400, detail="Only format=csv is supported currently")
    if period not in PERIOD_BUCKET:
        raise HTTPException(status_code=400, detail="period must be day, week, or month")
    start_date, end_date = _period_bounds(period, date)
    bucket = PERIOD_BUCKET[period]
    selected = kpis.split(",") if kpis else ALL_KPIS
    unknown = [k for k in selected if k not in ALL_KPIS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown KPI(s): {unknown}")

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows, total_row = _export_rows(cur, start_date, end_date, bucket)

    labels = KPI_LABELS.get(lang, KPI_LABELS["fr"])
    sep = ";" if lang == "fr" else ","
    row_label = ("Heure" if bucket == "hour" else "Date") if lang == "fr" else ("Hour" if bucket == "hour" else "Date")

    def fmt_num(v):
        if v is None:
            return ""
        s = f"{v}"
        return s.replace(".", ",") if lang == "fr" else s

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=sep)
    writer.writerow([row_label] + [labels[k] for k in selected])
    for r in rows:
        writer.writerow([r["label"]] + [fmt_num(r[k]) for k in selected])
    writer.writerow([total_row["label"]] + [fmt_num(total_row[k]) for k in selected])

    csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")

    if period == "day":
        filename = f"IQMS_{date}.csv"
    elif period == "week":
        iso_year, iso_week, _ = start_date.isocalendar()
        filename = f"IQMS_semaine_{iso_year}-W{iso_week:02d}.csv"
    else:
        filename = f"IQMS_mois_{start_date.strftime('%Y-%m')}.csv"

    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
