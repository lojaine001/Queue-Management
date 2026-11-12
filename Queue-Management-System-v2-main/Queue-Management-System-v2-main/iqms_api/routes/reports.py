"""Day recap and the background traffic / demographic statistics."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException

from prediction.forecast_state import freshness

from ..db import _conn, _http_error
from ..kpis import kpi_queue, kpi_wait
from ..settings import BUCKET_MIN

router = APIRouter()


@router.get("/day-recap")
def day_recap(date: Optional[str] = None):
    """
    Returns the summary for a given day (?date=YYYY-MM-DD), or falls back
    to the most recent day with data when no date is given.
    """
    try:
        requested_date = None
        if date:
            try:
                requested_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if requested_date:
                    ref_date = requested_date
                else:
                    # Find the most recent day that has data
                    cur.execute("""
                        SELECT DATE(timestamp) AS day
                        FROM entrance_events
                        WHERE camera_id NOT LIKE 'SIM_%%'
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    day_row = cur.fetchone()
                    ref_date = day_row["day"] if day_row else None
                date_filter = f"DATE(timestamp) = '{ref_date}'" if ref_date else "timestamp >= CURRENT_DATE"
                display_date = ref_date.strftime("%d %b %Y") if ref_date else datetime.now().strftime("%d %b %Y")

                cur.execute(f"""
                    SELECT COUNT(*) AS total
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                """)
                total = int((cur.fetchone() or {}).get("total") or 0)

                # Yesterday's total, for the vs-yesterday comparison
                yesterday_date = (ref_date - timedelta(days=1)) if ref_date else None
                yesterday_total = None
                if yesterday_date:
                    cur.execute("""
                        SELECT COUNT(*) AS total
                        FROM entrance_events
                        WHERE DATE(timestamp) = %s
                          AND camera_id NOT LIKE 'SIM_%%'
                          AND dwell_seconds >= 10
                    """, (yesterday_date,))
                    yesterday_total = int((cur.fetchone() or {}).get("total") or 0)

                # 7-day trend ending on ref_date, for the Clients Total sparkline
                trend_rows = []
                if ref_date:
                    cur.execute("""
                        SELECT DATE(timestamp) AS day, COUNT(*) AS cnt
                        FROM entrance_events
                        WHERE DATE(timestamp) BETWEEN %s AND %s
                          AND camera_id NOT LIKE 'SIM_%%'
                          AND dwell_seconds >= 10
                        GROUP BY 1 ORDER BY 1 ASC
                    """, (ref_date - timedelta(days=6), ref_date))
                    trend_rows = cur.fetchall()

                cur.execute(f"""
                    SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 1
                """)
                peak_row   = cur.fetchone()
                peak_hour  = peak_row["hour"].strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_end   = (peak_row["hour"] + timedelta(hours=1)).strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_count = int(peak_row["cnt"]) if peak_row else 0

                cur.execute(f"""
                    SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_min
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                avg_row      = cur.fetchone()
                avg_wait_min = float(avg_row["avg_min"]) if avg_row and avg_row["avg_min"] else 0.0

                # Lanes used today
                cur.execute(f"""
                    SELECT COUNT(DISTINCT lane_id) AS lanes_count
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                lanes_row    = cur.fetchone()
                lanes_today  = int(lanes_row["lanes_count"]) if lanes_row and lanes_row["lanes_count"] else 0

                cur.execute(f"""
                    SELECT lane_id, COUNT(*) AS cnt
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                    GROUP BY lane_id ORDER BY cnt DESC LIMIT 1
                """)
                busiest_row  = cur.fetchone()
                busiest_lane = busiest_row["lane_id"] if busiest_row else None

                # Alert minutes today (from ensemble predictions)
                alert_date_filter = f"DATE(prediction_for) = '{ref_date}'" if ref_date else "DATE(prediction_for) = CURRENT_DATE"
                cur.execute(f"""
                    SELECT COUNT(*) AS alert_slots
                    FROM queue_predictions
                    WHERE {alert_date_filter}
                      AND status = 'ALERT'
                """)
                alert_row     = cur.fetchone()
                alert_minutes = int((alert_row["alert_slots"] or 0) if alert_row else 0) * BUCKET_MIN

                # Equipment mix
                cur.execute(f"""
                    SELECT equipment_type, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND equipment_type IS NOT NULL
                      AND equipment_type != 'none'
                    GROUP BY equipment_type
                """)
                equip_rows = cur.fetchall()

                # Hourly entries — computed fresh per requested day, not from
                # dashboard_state, which only ever holds today's cache and
                # would be wrong for past dates.
                cur.execute(f"""
                    SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour,
                           COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                    GROUP BY 1 ORDER BY 1 ASC
                """)
                hourly_rows = cur.fetchall()
                wait_stats = kpi_wait(cur, ref_date, ref_date + timedelta(days=1)) if ref_date else kpi_wait(cur, None, None)
                queue_stats = kpi_queue(cur, ref_date, ref_date + timedelta(days=1)) if ref_date else kpi_queue(cur, None, None)

        trend_by_day = {r["day"]: int(r["cnt"]) for r in trend_rows}
        trend_7d = []
        if ref_date:
            for i in range(6, -1, -1):
                d = ref_date - timedelta(days=i)
                trend_7d.append({"date": d.strftime("%Y-%m-%d"), "count": trend_by_day.get(d, 0)})

        denom = total or 1
        equipment = []
        order  = ["trolley", "store_basket"]
        colors = {"trolley": "#06b6d4", "store_basket": "#a855f7"}
        labels = {"trolley": "Trolley", "store_basket": "Store basket"}
        for key in order:
            row = next((r for r in equip_rows if r["equipment_type"] == key), None)
            count = int(row["cnt"]) if row else 0
            equipment.append({
                "type":    key,
                "label":   labels[key],
                "count":   count,
                "percent": round(count / denom * 100),
                "color":   colors[key],
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

    except HTTPException:
        raise
    except Exception as exc:
        raise _http_error(exc)


@router.get("/statistics/summary")
def statistics_summary():
    """Background demographic/traffic snapshot, with its own freshness timestamp."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM dashboard_statistics WHERE id=1')
            row=dict(cur.fetchone() or {})
    if not row:return {"available":False,"stale":True}
    return {"available":True,**row,**freshness({"updated_at":row["updated_at"]})}
