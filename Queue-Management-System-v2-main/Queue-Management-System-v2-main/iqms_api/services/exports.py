"""KPI export for a day (one row per hour), week or month (one row per day): a preview
of the first rows, or the whole table as a CSV file (French: ';' and decimal commas)."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

from ..db import _conn
from ..errors import BadRequest
from ..repositories import kpis as repo
from ..settings import BUCKET_MIN
from .kpis import kpi_queue, kpi_wait

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


def period_bounds(period: str, date_str: str):
    """(first day, day after the last) of the day, week (Monday-Sunday) or month containing date_str."""
    try:
        ref = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise BadRequest("date must be YYYY-MM-DD")
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
    raise BadRequest("period must be day, week, or month")


def _request(period: str, date: str, kpis):
    """Checks the request (period, then date, then KPI names) -> (start, end, bucket, selected)."""
    if period not in PERIOD_BUCKET:
        raise BadRequest("period must be day, week, or month")
    start_date, end_date = period_bounds(period, date)
    bucket = PERIOD_BUCKET[period]
    selected = kpis.split(",") if kpis else ALL_KPIS
    unknown = [k for k in selected if k not in ALL_KPIS]
    if unknown:
        raise BadRequest(f"Unknown KPI(s): {unknown}")
    return start_date, end_date, bucket, selected


def export_rows(cur, start_date, end_date, bucket):
    """Returns (rows, total_row). rows: one per hour (day period) or per day
    (week/month period). total_row: sums for counts, the true overall
    avg/max from kpi_wait/kpi_queue (not an average-of-averages), and a
    true distinct-lane count over the whole range (not a sum, which would
    double-count a lane reused across multiple days)."""
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"

    entrance_by = {r["bucket"].strftime(fmt): r
                   for r in repo.entrance_counts_by_bucket(cur, trunc_unit, start_date, end_date)}

    wait = kpi_wait(cur, start_date, end_date, bucket)
    wait_key = "by_hour" if bucket == "hour" else "by_day"
    wait_by = {(r.get("hour") or r.get("date")): r for r in wait[wait_key]}

    queue = kpi_queue(cur, start_date, end_date, bucket)
    queue_by = {(r.get("hour") or r.get("date")): r for r in queue[wait_key]}

    lanes_by = {r["bucket"].strftime(fmt): int(r["lanes_used"])
                for r in repo.lanes_used_by_bucket(cur, trunc_unit, start_date, end_date)}

    alert_by = {r["bucket"].strftime(fmt): int(r["alert_slots"]) * BUCKET_MIN
                for r in repo.alert_slots_by_bucket(cur, trunc_unit, start_date, end_date)}

    peak_hour_by = {}
    if bucket == "day":
        by_day_hours = {}
        for r in repo.entrance_counts_by_day_and_hour(cur, start_date, end_date):
            d = r["day"].strftime("%Y-%m-%d")
            by_day_hours.setdefault(d, []).append((r["hour"], int(r["cnt"])))
        for d, hrs in by_day_hours.items():
            peak_hour_by[d] = max(hrs, key=lambda x: x[1])[0].strftime("%H:%M")

    def row(label, peak_hour):
        e = entrance_by.get(label)
        w = wait_by.get(label, {})
        q = queue_by.get(label, {})
        return {
            "label": label, "customers": int(e["customers"]) if e else 0,
            "avg_wait": w.get("avg_wait_min"), "max_wait": w.get("max_wait_min"),
            "avg_waiting": q.get("avg_waiting"), "peak_waiting": q.get("max_waiting"),
            "lanes_used": lanes_by.get(label, 0), "alert_minutes": alert_by.get(label, 0),
            "trolley": int(e["trolley"]) if e else 0, "store_basket": int(e["store_basket"]) if e else 0,
            "peak_hour": peak_hour,
        }

    rows = []
    if bucket == "hour":
        for h in range(24):
            rows.append(row(f"{h:02d}:00", None))
    else:
        d = start_date
        while d < end_date:
            label = d.strftime("%Y-%m-%d")
            rows.append(row(label, peak_hour_by.get(label)))
            d += timedelta(days=1)

    total_lanes_row = repo.lanes_used_total(cur, start_date, end_date)
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


def _read_rows(start_date, end_date, bucket):
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            return export_rows(cur, start_date, end_date, bucket)


def preview(period: str, date: str, kpis, lang: str) -> dict:
    """The selected KPIs' labels, the first 10 rows and the total."""
    start_date, end_date, bucket, selected = _request(period, date, kpis)
    rows, total_row = _read_rows(start_date, end_date, bucket)

    labels = KPI_LABELS.get(lang, KPI_LABELS["fr"])
    return {
        "period": period, "bucket": bucket, "kpis": selected,
        "labels": {k: labels[k] for k in selected},
        "rows": [{"label": r["label"], **{k: r[k] for k in selected}} for r in rows[:10]],
        "total": {"label": total_row["label"], **{k: total_row[k] for k in selected}},
        "row_count": len(rows),
    }


def csv_file(period: str, date: str, kpis, file_format: str, lang: str) -> tuple[bytes, str]:
    """(file content, file name). UTF-8 with a BOM so Excel reads the accents."""
    if file_format != "csv":
        raise BadRequest("Only format=csv is supported currently")
    start_date, end_date, bucket, selected = _request(period, date, kpis)
    rows, total_row = _read_rows(start_date, end_date, bucket)

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

    csv_bytes = ("﻿" + buf.getvalue()).encode("utf-8")

    if period == "day":
        filename = f"IQMS_{date}.csv"
    elif period == "week":
        iso_year, iso_week, _ = start_date.isocalendar()
        filename = f"IQMS_semaine_{iso_year}-W{iso_week:02d}.csv"
    else:
        filename = f"IQMS_mois_{start_date.strftime('%Y-%m')}.csv"
    return csv_bytes, filename
