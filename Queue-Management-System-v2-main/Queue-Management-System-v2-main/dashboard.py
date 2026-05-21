import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from plotly.subplots import make_subplots

load_dotenv(find_dotenv(usecwd=True))

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from prediction.core import (  # noqa: E402
    DEFAULT_DWELL_MIN,
    DWELL_MAX_CAP,
    DWELL_MIN_FLOOR,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    WAIT_45M_INDEX,
    compute_wait_estimates,
)

CAMERA_ID = os.getenv("CAM_ID", "Bosch_Camera_Entrance")
REFRESH_SEC = int(os.getenv("REFRESH_SEC", 30))
WAIT_BUSY_MIN = float(os.getenv("WAIT_BUSY_MIN", 5.0))
WAIT_ALERT_MIN = float(os.getenv("WAIT_ALERT_MIN", 10.0))
BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))
DEFAULT_LANES = int(os.getenv("DEFAULT_LANES", 2))
SNAPSHOT_MAX_AGE_MIN = int(os.getenv("SNAPSHOT_MAX_AGE_MIN", 30))
BROWSING_GAP_MIN = int(os.getenv("BROWSING_GAP_MIN", 25))

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

LOCAL_TZ = datetime.now(timezone.utc).astimezone().tzinfo


def _conn():
    return psycopg2.connect(**DB_CONFIG)


def _run_prediction():
    _here  = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(_here, "ensemble_predict.py")
    cwd    = os.path.dirname(os.path.dirname(_here))   # Queue-Management/ for prediction imports
    env    = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, script, "--source", "REAL"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
    )
    return result.returncode == 0, result.stdout, result.stderr


def _to_local_timestamp(value):
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(LOCAL_TZ)
    return ts


def _to_local_series(series):
    local = pd.to_datetime(series)
    if getattr(local.dt, "tz", None) is not None:
        local = local.dt.tz_convert(LOCAL_TZ)
    return local


def _snapshot_age_minutes(snapshot_ts):
    ts_local = _to_local_timestamp(snapshot_ts)
    if ts_local is None:
        return None, None
    now_local = pd.Timestamp.now(tz=LOCAL_TZ)
    age_min = max(0.0, (now_local - ts_local).total_seconds() / 60.0)
    return ts_local, age_min


def _is_live_lane_snapshot(snapshot):
    if snapshot is None:
        return False
    if pd.isna(snapshot.get("active_lanes")):
        return False
    _, age_min = _snapshot_age_minutes(snapshot.get("timestamp"))
    return age_min is not None and age_min <= SNAPSHOT_MAX_AGE_MIN


def _format_wait(value):
    return f"{float(value):.1f} min" if value is not None else "Forecast pending"


def _format_metric_value(value, suffix=""):
    if value is None:
        return "—"
    return f"{value}{suffix}"


def _lane_phrase(lanes):
    return f"{lanes} lane" if lanes == 1 else f"{lanes} lanes"


def _hour_phrase(hours):
    return f"{hours} hour" if hours == 1 else f"{hours} hours"


def _status_meta(wait_value):
    if wait_value is None:
        return "status-ok", "Forecast pending"
    if wait_value >= WAIT_ALERT_MIN:
        return "status-alert", "ALERT"
    if wait_value >= WAIT_BUSY_MIN:
        return "status-busy", "BUSY"
    return "status-ok", "ALL CLEAR"


def _badge_html(wait_value):
    if wait_value is None:
        return '<span class="badge badge-muted">PENDING</span>'
    if wait_value >= WAIT_ALERT_MIN:
        return '<span class="badge badge-alert">ALERT</span>'
    if wait_value >= WAIT_BUSY_MIN:
        return '<span class="badge badge-busy">BUSY</span>'
    return '<span class="badge badge-ok">OK</span>'


def _metric_card_html(label, value, tone="blue", delta_html="", supporting_text=""):
    supporting = f'<div class="metric-support">{supporting_text}</div>' if supporting_text else ""
    return f"""
    <div class="metric-card {tone}">
      <div class="metric-lbl">{label}</div>
      <div class="metric-val">{value}</div>
      {delta_html}
      {supporting}
    </div>
    """


def _lane_card_html(lane_count, wait_value, is_active=False, is_default=False):
    tone = "purple" if is_active else "blue"
    chip = "Current live setting" if is_active else "Default assumption" if is_default else "Comparison"
    lane_note = "Used for the main forecast" if is_active else "Fallback lane baseline" if is_default else "What-if scenario"
    return f"""
    <div class="metric-card {tone} lane-card {'lane-current' if is_active else ''}">
      <div class="lane-card-top">
        <div class="metric-lbl">{_lane_phrase(lane_count).title()}</div>
        <span class="context-pill">{chip}</span>
      </div>
      <div class="metric-val">{_format_metric_value(f"{wait_value:.1f}" if wait_value is not None else None, "")}<span class="metric-unit"> min</span></div>
      <div style="margin-top:8px">{_badge_html(wait_value)}</div>
      <div class="metric-support">{lane_note}</div>
    </div>
    """


def _delta_html(diff, suffix="", invert=False):
    if diff is None:
        return '<div class="metric-delta metric-delta-muted">No prior comparison</div>'
    good = diff <= 0 if invert else diff >= 0
    color = "#15803d" if good else "#dc2626"
    sign = "+" if diff > 0 else ""
    return (
        f'<div class="metric-delta" style="color:{color}">'
        f"{sign}{diff}{suffix}</div>"
    )


def _wait_tone(wait_value):
    if wait_value is None:
        return "blue"
    if wait_value >= WAIT_ALERT_MIN:
        return "red"
    if wait_value >= WAIT_BUSY_MIN:
        return "orange"
    return "green"


def _queue_tone(queue_count):
    if queue_count <= 5:
        return "green"
    if queue_count <= 10:
        return "orange"
    return "red"


def _section_title(title, subtitle=""):
    subtitle_html = f'<span class="section-subtitle">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{title}{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def _apply_chart_layout(fig, *, height=260, y_suffix="", xaxis_type=None):
    xaxis = dict(
        showgrid=False,
        title="",
        tickfont=dict(size=11, color="#475569"),
    )
    if xaxis_type == "date":
        xaxis["tickformat"] = "%H:%M"
        xaxis["hoverformat"] = "%H:%M"

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=20, t=18, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=xaxis,
        yaxis=dict(
            gridcolor="#e2e8f0",
            title="",
            zeroline=False,
            ticksuffix=y_suffix,
            tickfont=dict(size=11, color="#475569"),
        ),
        hovermode="x unified",
        showlegend=False,
        font=dict(color="#0f172a"),
    )


def _run_prediction_ui():
    with st.spinner("Running prediction model... (1-3 min)"):
        ok, _out, err = _run_prediction()
    if ok:
        st.cache_data.clear()
        st.rerun()
    st.error("Prediction failed.")
    if err:
        with st.expander("Error details"):
            st.code(err[-3000:])


def _status_label(wait_value):
    if wait_value is None:
        return "Pending"
    if wait_value >= WAIT_ALERT_MIN:
        return "Alert"
    if wait_value >= WAIT_BUSY_MIN:
        return "Busy"
    return "OK"


def _forecast_row_style(row):
    wait_value = row["Estimated Wait (min)"]
    if pd.isna(wait_value):
        status_bg = "#e2e8f0"
        status_color = "#475569"
        wait_color = "#475569"
    elif wait_value >= WAIT_ALERT_MIN:
        status_bg = "#fee2e2"
        status_color = "#b91c1c"
        wait_color = "#b91c1c"
    elif wait_value >= WAIT_BUSY_MIN:
        status_bg = "#ffedd5"
        status_color = "#c2410c"
        wait_color = "#c2410c"
    else:
        status_bg = "#dcfce7"
        status_color = "#166534"
        wait_color = "#166534"

    styles = []
    for col in row.index:
        if col == "Time":
            styles.append("font-weight: 700; color: #0f172a;")
        elif col == "Status":
            styles.append(
                f"background-color: {status_bg}; color: {status_color}; font-weight: 700; border-radius: 999px;"
            )
        elif col == "Estimated Wait (min)":
            styles.append(f"font-weight: 700; color: {wait_color};")
        else:
            styles.append("color: #0f172a;")
    return styles


def _forecast_display_frame(pf):
    display = pf.copy()
    display["display_ds"] = display["ds"].dt.ceil("5min")
    display = (
        display.sort_values("ds")
        .groupby("display_ds", as_index=False)
        .agg(
            arrivals=("arrivals", "last"),
            wait_min=("wait_min", "last"),
        )
        .rename(columns={"display_ds": "ds"})
        .dropna(subset=["arrivals", "wait_min"], how="all")
    )
    return display


def _estimated_service_minutes(snapshot):
    raw_dwell_sec = None
    if snapshot is not None and pd.notna(snapshot.get("avg_dwell_sec")):
        raw_dwell_sec = float(snapshot["avg_dwell_sec"])
    raw_dwell_min = (raw_dwell_sec / 60.0) if raw_dwell_sec is not None else None
    checkout_service_min = float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))
    if raw_dwell_min is None or raw_dwell_min < 1.0:
        base_service = checkout_service_min
    else:
        base_service = raw_dwell_min
    return min(max(base_service, DWELL_MIN_FLOOR), DWELL_MAX_CAP)


def _build_checkout_arrivals(forecast_rows, recent_entries):
    wait_frame = forecast_rows[["ds"]].copy()
    lag_steps = max(1, round(BROWSING_GAP_MIN / BUCKET_MIN))
    hist = recent_entries.copy()
    if not hist.empty:
        hist["bucket"] = _to_local_series(hist["bucket"]).dt.floor(f"{BUCKET_MIN}min")
        hist_by_time = dict(zip(hist["bucket"], hist["entry_count"].astype(float)))
    else:
        hist_by_time = {}

    now_floor = pd.Timestamp.now(tz=LOCAL_TZ).floor(f"{BUCKET_MIN}min")
    arrivals = forecast_rows["arrivals"].fillna(0.0).astype(float).tolist()
    checkout_arrivals = []
    for idx in range(len(forecast_rows)):
        entrance_idx = idx - lag_steps
        if entrance_idx < 0:
            entrance_time = now_floor + pd.Timedelta(minutes=BUCKET_MIN * entrance_idx)
            checkout_arrivals.append(float(hist_by_time.get(entrance_time, 0.0)))
        else:
            checkout_arrivals.append(arrivals[entrance_idx] if entrance_idx < len(arrivals) else 0.0)
    wait_frame["yhat"] = checkout_arrivals
    return wait_frame


def _compute_waits_for_lanes(forecast_rows, recent_entries, current_queue, service_minutes, lane_count):
    wait_frame = _build_checkout_arrivals(forecast_rows, recent_entries)
    wait_rows, wait_15m, wait_30m, _service_per_bucket = compute_wait_estimates(
        wait_frame,
        current_queue=current_queue,
        avg_dwell_min=service_minutes,
        active_lanes=lane_count,
        max_queue_per_lane=MAX_QUEUE_PER_LANE,
        max_wait_min=MAX_WAIT_MIN,
    )
    wait_df = pd.DataFrame(wait_rows)
    wait_45m = float(wait_rows[WAIT_45M_INDEX]["wait_min"]) if len(wait_rows) > WAIT_45M_INDEX else wait_30m
    return wait_df, wait_15m, wait_30m, wait_45m


# ── Live data ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SEC)
def load_snapshot():
    with _conn() as conn:
        row = pd.read_sql(
            """
            SELECT queue_count, avg_dwell_sec, active_lanes, timestamp
            FROM queue_state_snapshots
            ORDER BY timestamp DESC LIMIT 1
            """,
            conn,
        )
    return row.iloc[0] if not row.empty else None


@st.cache_data(ttl=REFRESH_SEC)
def load_queue_delta():
    with _conn() as conn:
        row = pd.read_sql(
            """
            SELECT queue_count FROM queue_state_snapshots
            WHERE timestamp <= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC LIMIT 1
            """,
            conn,
        )
    return int(row.iloc[0]["queue_count"]) if not row.empty else None


@st.cache_data(ttl=REFRESH_SEC)
def load_predictions():
    with _conn() as conn:
        try:
            df = pd.read_sql(
                """
                SELECT DISTINCT ON (prediction_for)
                       prediction_for AS ds,
                       ensemble_yhat AS arrivals,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, wait_45m, status,
                       wait_1lane_15m, wait_2lane_15m, wait_3lane_15m
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '45 minutes'
                ORDER BY prediction_for, predicted_at DESC
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.read_sql(
                """
                SELECT DISTINCT ON (prediction_for)
                       prediction_for AS ds,
                       ensemble_yhat AS arrivals,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, status
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '45 minutes'
                ORDER BY prediction_for, predicted_at DESC
                """,
                conn,
            )
            for col in ["wait_45m", "wait_1lane_15m", "wait_2lane_15m", "wait_3lane_15m"]:
                df[col] = None
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_queue_history(hours):
    with _conn() as conn:
        df = pd.read_sql(
            f"""
            SELECT timestamp, queue_count, avg_dwell_sec
            FROM queue_state_snapshots
            WHERE timestamp >= NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY timestamp ASC
            """,
            conn,
        )
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_recent_entrance_history():
    lookback_min = BROWSING_GAP_MIN + BUCKET_MIN
    with _conn() as conn:
        df = pd.read_sql(
            f"""
            SELECT
                time_bucket('{BUCKET_MIN} minutes', timestamp) AS bucket,
                COUNT(*) AS entry_count
            FROM entrance_events
            WHERE camera_id = %s
              AND timestamp >= NOW() - INTERVAL '{lookback_min} minutes'
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            conn,
            params=(CAMERA_ID,),
        )
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_entries_delta():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '1 hour') AS last_hour,
                    COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '2 hours'
                                      AND timestamp < NOW() - INTERVAL '1 hour') AS prev_hour
                FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= NOW() - INTERVAL '2 hours'
                """,
                (CAMERA_ID,),
            )
            return cur.fetchone()


# ── Daily tracker data ─────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SEC)
def load_entries_today():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id = %s AND timestamp >= CURRENT_DATE
                """,
                (CAMERA_ID,),
            )
            return cur.fetchone()[0]


@st.cache_data(ttl=REFRESH_SEC)
def load_yesterday_entries():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= CURRENT_DATE - INTERVAL '1 day'
                  AND timestamp < CURRENT_DATE
                """,
                (CAMERA_ID,),
            )
            return cur.fetchone()[0]


@st.cache_data(ttl=REFRESH_SEC)
def load_today_traffic():
    with _conn() as conn:
        df = pd.read_sql(
            """
            SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS entries
            FROM entrance_events
            WHERE camera_id = %s AND timestamp >= CURRENT_DATE
            GROUP BY 1 ORDER BY 1
            """,
            conn,
            params=(CAMERA_ID,),
        )
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_full_day_queue():
    with _conn() as conn:
        df = pd.read_sql(
            """
            SELECT timestamp, queue_count
            FROM queue_state_snapshots
            WHERE timestamp >= CURRENT_DATE
            ORDER BY timestamp ASC
            """,
            conn,
        )
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_status_breakdown():
    with _conn() as conn:
        df = pd.read_sql(
            """
            SELECT UPPER(status) AS status, COUNT(*) AS slots
            FROM queue_predictions
            WHERE prediction_for >= CURRENT_DATE
              AND prediction_for < NOW()
            GROUP BY status
            """,
            conn,
        )
    return df


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="IQMS - Live Dashboard", page_icon="📊", layout="wide")

st.markdown(
    f"""
<script>
  setTimeout(function() {{ window.location.reload(); }}, {REFRESH_SEC * 1000});
</script>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at top left, rgba(148, 163, 184, 0.14), transparent 26%),
      linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
  }
  [data-testid="stHeader"] { background: transparent; }
  .top-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
    color: white;
    border-radius: 24px;
    padding: 28px 30px;
    margin-bottom: 16px;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
  }
  .header-kicker {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    opacity: 0.68;
    margin-bottom: 10px;
  }
  .header-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
  }
  .header-sub {
    font-size: 0.93rem;
    opacity: 0.82;
    margin-top: 10px;
  }
  .header-meta {
    text-align: right;
    font-size: 0.88rem;
    opacity: 0.82;
    font-weight: 600;
  }
  .status-banner {
    border-radius: 18px;
    padding: 18px 22px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
  }
  .status-banner-head {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    opacity: 0.8;
    margin-bottom: 8px;
  }
  .status-banner-body {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .status-banner-sub {
    font-size: 0.92rem;
    margin-top: 8px;
    opacity: 0.85;
  }
  .status-ok { background: linear-gradient(135deg, #16a34a, #15803d); }
  .status-busy { background: linear-gradient(135deg, #ea580c, #c2410c); }
  .status-alert { background: linear-gradient(135deg, #dc2626, #b91c1c); }
  .context-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
  }
  .context-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.78);
    color: #0f172a;
    font-size: 0.8rem;
    font-weight: 700;
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.35);
  }
  .metric-card {
    background: rgba(255, 255, 255, 0.94);
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-top: 5px solid #2563eb;
    min-height: 136px;
  }
  .metric-card.green { border-top-color: #16a34a; }
  .metric-card.orange { border-top-color: #ea580c; }
  .metric-card.red { border-top-color: #dc2626; }
  .metric-card.blue { border-top-color: #2563eb; }
  .metric-card.purple { border-top-color: #7c3aed; }
  .metric-lbl {
    font-size: 0.74rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
  }
  .metric-val {
    font-size: 2.15rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.08;
    margin-top: 10px;
    letter-spacing: -0.03em;
  }
  .metric-unit {
    font-size: 1rem;
    font-weight: 500;
    color: #475569;
  }
  .metric-delta {
    font-size: 0.84rem;
    font-weight: 700;
    margin-top: 10px;
  }
  .metric-delta-muted,
  .metric-support {
    color: #64748b;
  }
  .metric-support {
    font-size: 0.83rem;
    margin-top: 8px;
    line-height: 1.35;
  }
  .lane-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }
  .lane-current {
    box-shadow: 0 16px 36px rgba(124, 58, 237, 0.13);
  }
  .section-title {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    font-size: 0.82rem;
    font-weight: 800;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 30px 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(203, 213, 225, 0.8);
  }
  .section-subtitle {
    font-size: 0.74rem;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.08em;
  }
  .detail-note {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(226, 232, 240, 0.9);
    border-radius: 16px;
    padding: 12px 14px;
    color: #334155;
    font-size: 0.86rem;
    margin-bottom: 12px;
  }
  .tracker-heading {
    font-size: 1.18rem;
    font-weight: 800;
    color: #0f172a;
    margin: 10px 0 16px 0;
    letter-spacing: -0.02em;
  }
  .tracker-sub {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    margin-left: 8px;
    letter-spacing: 0.04em;
  }
  .forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    background: rgba(255, 255, 255, 0.92);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
  }
  .forecast-table th {
    background: #f8fafc;
    color: #64748b;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }
  .forecast-table td {
    padding: 11px 14px;
    border-bottom: 1px solid #edf2f7;
    color: #0f172a;
  }
  .forecast-table tr:last-child td {
    border-bottom: none;
  }
  .forecast-table tr:hover td {
    background: #f8fbff;
  }
  .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.06em;
  }
  .badge-ok { background: #dcfce7; color: #166534; }
  .badge-busy { background: #ffedd5; color: #c2410c; }
  .badge-alert { background: #fee2e2; color: #b91c1c; }
  .badge-muted { background: #e2e8f0; color: #475569; }
  .tracker-divider {
    margin: 34px 0 24px 0;
    border: none;
    border-top: 3px solid rgba(203, 213, 225, 0.85);
  }
</style>
""",
    unsafe_allow_html=True,
)

if "history_range_hours" not in st.session_state:
    st.session_state["history_range_hours"] = 2


# ── Load all data ──────────────────────────────────────────────────────────────

try:
    snap = load_snapshot()
    pred_df = load_predictions()
    recent_entry_hist = load_recent_entrance_history()
    queue_hist = load_queue_history(st.session_state["history_range_hours"])
    traffic_today = load_today_traffic()
    entries_today = load_entries_today()
    yesterday_entries = load_yesterday_entries()
    queue_1h_ago = load_queue_delta()
    entries_delta = load_entries_delta()
    full_day_queue = load_full_day_queue()
    status_breakdown = load_status_breakdown()
except Exception as exc:
    st.error(f"Database error: {exc}")
    st.stop()

live_lane_source = _is_live_lane_snapshot(snap)
detected_lanes = int(snap["active_lanes"]) if live_lane_source else DEFAULT_LANES
queue_count = int(snap["queue_count"]) if snap is not None and pd.notna(snap["queue_count"]) else 0
snapshot_ts_local, snapshot_age_min = _snapshot_age_minutes(snap["timestamp"] if snap is not None else None)
service_minutes = _estimated_service_minutes(snap)

if "forecast_active_lanes" not in st.session_state:
    st.session_state["forecast_active_lanes"] = detected_lanes
selected_lanes = int(st.session_state["forecast_active_lanes"])
selected_lanes = max(1, min(5, selected_lanes))

wait_15m = None
wait_30m = None
wait_45m = None
lane_waits = {}
forecast_waits = pd.DataFrame(columns=["ds", "wait_min"])
pred_future = pd.DataFrame(columns=["ds", "arrivals"])

_arrivals_capped = False
# Hard ceiling: > 8 arrivals per 3-min bucket = 160 entries/hr — unrealistic for this store.
# This fires even when the observed rate is unknown (entries_delta = 0).
_ARRIVALS_HARD_CAP = 8.0
if not pred_df.empty:
    pred_future = pred_df[["ds", "arrivals"]].copy()
    pred_future["ds"] = _to_local_series(pred_future["ds"])
    pred_future["arrivals"] = pd.to_numeric(pred_future["arrivals"], errors="coerce")
    _actual_rate = (entries_delta[0] if entries_delta else 0) / max(1.0, 60.0 / BUCKET_MIN)
    _pred_mean = pred_future["arrivals"].mean(skipna=True)
    if _actual_rate > 0 and _pred_mean > _actual_rate * 2.0:
        # Rate-based cap: predictions > 2x observed → clip to 1.5x observed
        _cap_ceiling = min(_actual_rate * 1.5, _ARRIVALS_HARD_CAP)
        pred_future["arrivals"] = pred_future["arrivals"].clip(upper=_cap_ceiling)
        _arrivals_capped = True
    elif _pred_mean > _ARRIVALS_HARD_CAP:
        # Hard cap fallback: fires when actual_rate is 0 or unavailable
        pred_future["arrivals"] = pred_future["arrivals"].clip(upper=_ARRIVALS_HARD_CAP)
        _arrivals_capped = True
    forecast_waits, wait_15m, wait_30m, wait_45m = _compute_waits_for_lanes(
        pred_future.copy(),
        recent_entry_hist,
        queue_count,
        service_minutes,
        selected_lanes,
    )
    for lane_count in range(1, 6):
        _wait_df, lane_wait_15, _wait_30, _wait_45 = _compute_waits_for_lanes(
            pred_future.copy(),
            recent_entry_hist,
            queue_count,
            service_minutes,
            lane_count,
        )
        lane_waits[lane_count] = {
            "wait_15m": lane_wait_15,
            "wait_30m": _wait_30,
            "wait_45m": _wait_45,
        }

entries_last_hr = entries_delta[0] if entries_delta else 0
entries_prev_hr = entries_delta[1] if entries_delta else 0
entries_diff = entries_last_hr - entries_prev_hr
queue_diff = (queue_count - queue_1h_ago) if queue_1h_ago is not None else None

if not full_day_queue.empty:
    peak_queue_today = int(full_day_queue["queue_count"].max())
    avg_queue_today = round(float(full_day_queue["queue_count"].mean()), 1)
    peak_row = full_day_queue.loc[full_day_queue["queue_count"].idxmax()]
    peak_time_str = _to_local_timestamp(peak_row["timestamp"]).strftime("%H:%M")
else:
    peak_queue_today = 0
    avg_queue_today = 0.0
    peak_time_str = "--"

entries_vs_yesterday = int(entries_today) - int(yesterday_entries)

if not traffic_today.empty:
    busiest_idx = traffic_today["entries"].idxmax()
    busiest_hour_str = _to_local_timestamp(traffic_today.loc[busiest_idx, "hour"]).strftime("%H:%M")
else:
    busiest_hour_str = "--"

alert_slots = 0
if not status_breakdown.empty:
    alert_row = status_breakdown[status_breakdown["status"].str.upper() == "ALERT"]
    if not alert_row.empty:
        alert_slots = int(alert_row.iloc[0]["slots"])
alert_minutes_today = alert_slots * BUCKET_MIN

# +45 min estimate always uses 4 lanes (peak rush-hour planning assumption)
RUSH_HOUR_LANES = 4
# +30 min estimate uses 3 lanes (typical busy-period assumption; avoids hitting cap on 2 lanes)
TYPICAL_LANES = 3
wait_45m_rush = lane_waits.get(RUSH_HOUR_LANES, {}).get("wait_45m") if lane_waits else None
wait_30m_typical = lane_waits.get(TYPICAL_LANES, {}).get("wait_30m") if lane_waits else None

status_class, status_label = _status_meta(wait_15m)
lane_source_label = "Live lanes" if live_lane_source else "Default lanes"
lane_source_note = (
    f"Detected queue state is {_lane_phrase(detected_lanes)} open."
    if live_lane_source
    else (
        f"No recent lane snapshot in the last {SNAPSHOT_MAX_AGE_MIN} min. "
        f"Predictions fall back to {_lane_phrase(DEFAULT_LANES)} open."
    )
)
lane_parameter_note = (
    f"Forecast parameter is set to {_lane_phrase(selected_lanes)}."
    if selected_lanes == detected_lanes
    else f"Forecast parameter overrides the detected state: {_lane_phrase(selected_lanes)} selected."
)
status_text = (
    f"{status_label} - Predicted wait with {_lane_phrase(selected_lanes)}: {_format_wait(wait_15m)}"
    if wait_15m is not None
    else f"{status_label} - Waiting for the next lane-aware forecast"
)


# ── Header ─────────────────────────────────────────────────────────────────────

now_str = datetime.now().strftime("%H:%M  -  %A, %d %B %Y")
st.markdown(
    f"""
<div class="top-header">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:18px;">
    <div>
      <div class="header-kicker">Live Queue Operations</div>
      <div class="header-title">IQMS Live Dashboard</div>
      <div class="header-sub">{CAMERA_ID} · auto-refresh every {REFRESH_SEC}s</div>
    </div>
    <div class="header-meta">{now_str}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="status-banner {status_class}">
  <div class="status-banner-head">Lane-Aware Prediction Status</div>
  <div class="status-banner-body">{status_text}</div>
  <div class="status-banner-sub">{lane_source_note} {lane_parameter_note}</div>
</div>
""",
    unsafe_allow_html=True,
)

context_bits = [
    f"{lane_source_label}: {_lane_phrase(detected_lanes)}",
    f"Forecast parameter: {_lane_phrase(selected_lanes)}",
    f"Wait +30 min ({TYPICAL_LANES} lanes): {_format_wait(wait_30m_typical)}",
    f"Wait +45 min ({RUSH_HOUR_LANES} lanes): {_format_wait(wait_45m_rush)}",
]
if snapshot_ts_local is not None:
    snapshot_note = f"Latest snapshot: {snapshot_ts_local.strftime('%H:%M:%S')}"
    if snapshot_age_min is not None:
        snapshot_note += f" ({snapshot_age_min:.0f} min ago)"
    context_bits.append(snapshot_note)

st.markdown(
    '<div class="context-row">' +
    "".join(f'<span class="context-pill">{bit}</span>' for bit in context_bits) +
    "</div>",
    unsafe_allow_html=True,
)

if _arrivals_capped:
    st.warning(
        "Arrival forecast capped: model predicted arrivals were more than 4x the actual recent entry rate. "
        "Wait times are based on a capped estimate. This may indicate a training data issue — run prediction after cleaning historical data."
    )

action_refresh, action_predict, action_param, action_note = st.columns([1, 1.2, 1.8, 2.8])
with action_refresh:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with action_predict:
    if st.button("Run Prediction", use_container_width=True, type="primary"):
        _run_prediction_ui()
with action_param:
    st.select_slider(
        "Forecast active lanes",
        options=[1, 2, 3, 4, 5],
        value=selected_lanes,
        key="forecast_active_lanes",
    )
with action_note:
    st.markdown(
        '<div class="detail-note">Change the forecast lanes parameter to recalculate the +15 and +30 min predictions. '
        'The +45 min card always uses 4 lanes as a peak rush-hour planning estimate.</div>',
        unsafe_allow_html=True,
    )


# ── Live operations ────────────────────────────────────────────────────────────

_section_title("Live Operations", "current queue, lane-aware waits, near-term trend")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        _metric_card_html(
            "In Queue Now",
            str(queue_count),
            tone=_queue_tone(queue_count),
            delta_html=_delta_html(queue_diff, " vs 1h ago", invert=True),
            supporting_text="Latest observed queue depth.",
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        _metric_card_html(
            "Predicted Wait (+15 min)",
            f"{wait_15m:.1f}<span class='metric-unit'> min</span>" if wait_15m is not None else "—",
            tone=_wait_tone(wait_15m),
            supporting_text=f"Computed with forecast parameter: {_lane_phrase(selected_lanes)}.",
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        _metric_card_html(
            "Predicted Wait (+30 min)",
            f"{wait_30m_typical:.1f}<span class='metric-unit'> min</span>" if wait_30m_typical is not None else "—",
            tone=_wait_tone(wait_30m_typical),
            supporting_text=f"Typical estimate: assumes {_lane_phrase(TYPICAL_LANES)} open.",
        ),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        _metric_card_html(
            "Predicted Wait (+45 min)",
            f"{wait_45m_rush:.1f}<span class='metric-unit'> min</span>" if wait_45m_rush is not None else "—",
            tone=_wait_tone(wait_45m_rush),
            supporting_text=f"Rush-hour estimate: assumes {_lane_phrase(RUSH_HOUR_LANES)} open.",
        ),
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        _metric_card_html(
            "Entries Last Hour",
            str(entries_last_hr),
            tone="green" if entries_diff >= 0 else "orange",
            delta_html=_delta_html(entries_diff, " vs prev hr"),
            supporting_text="Entrance events for the tracked camera.",
        ),
        unsafe_allow_html=True,
    )

if snapshot_ts_local is not None:
    freshness = "recent" if live_lane_source else "stale for lane logic"
    st.caption(
        f"Queue snapshot: {snapshot_ts_local.strftime('%H:%M:%S  %d/%m/%Y')} · {freshness}"
    )


# ── Forecast section ───────────────────────────────────────────────────────────

_section_title("Predicted Wait - Next 30 Min", f"assumes {_lane_phrase(selected_lanes)} open")
st.markdown(
    f'<div class="detail-note">{lane_source_note} {lane_parameter_note} The trend below is recalculated in the dashboard '
    f'from the saved arrival forecast and current queue state.</div>',
    unsafe_allow_html=True,
)

if not pred_df.empty:
    pf_raw = pred_future.copy()
    pf_raw = pf_raw.merge(forecast_waits, on="ds", how="left")
    pf_raw = pf_raw[pf_raw["ds"] <= (pd.Timestamp.now(tz=LOCAL_TZ) + pd.Timedelta(minutes=30))]
    pf = _forecast_display_frame(pf_raw)
    if not pf.empty:
        fig_wait = go.Figure()
        fig_wait.add_trace(
            go.Scatter(
                x=pf["ds"],
                y=pf["wait_min"],
                mode="lines",
                line=dict(color="#2563eb", width=3),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.14)",
                hovertemplate="%{x|%H:%M}<br>Predicted wait: %{y:.1f} min<br>Lanes: " + str(selected_lanes) + "<extra></extra>",
            )
        )
        fig_wait.add_hline(
            y=WAIT_BUSY_MIN,
            line=dict(color="#ea580c", width=1.5, dash="dot"),
            annotation_text="Busy",
            annotation_position="top right",
            annotation_font=dict(color="#c2410c", size=11),
        )
        fig_wait.add_hline(
            y=WAIT_ALERT_MIN,
            line=dict(color="#dc2626", width=1.5, dash="dot"),
            annotation_text="Alert",
            annotation_position="top right",
            annotation_font=dict(color="#b91c1c", size=11),
        )
        _apply_chart_layout(fig_wait, height=260, y_suffix=" min", xaxis_type="date")
        st.plotly_chart(fig_wait, use_container_width=True)

        _section_title("Forecast Detail", "5-minute display cadence")
        next_slot = pf.iloc[0]
        peak_candidates = pf.dropna(subset=["wait_min"])
        peak_slot = peak_candidates.loc[peak_candidates["wait_min"].idxmax()] if not peak_candidates.empty else pf.iloc[0]
        end_slot = pf.iloc[-1]
        next_arrivals = 0.0 if pd.isna(next_slot["arrivals"]) else float(next_slot["arrivals"])
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                _metric_card_html(
                    "Next Forecast Slot",
                    next_slot["ds"].strftime("%H:%M"),
                    tone="blue",
                    supporting_text=f"5-minute view: wait {_format_wait(next_slot['wait_min'])} with {next_arrivals:.1f} predicted arrivals.",
                ),
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                _metric_card_html(
                    "Peak Wait In Window",
                    _format_wait(peak_slot["wait_min"]).replace(" min", "<span class='metric-unit'> min</span>"),
                    tone=_wait_tone(peak_slot["wait_min"]),
                    supporting_text=f"Highest wait expected at {peak_slot['ds'].strftime('%H:%M')}.",
                ),
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                _metric_card_html(
                    "Window End",
                    end_slot["ds"].strftime("%H:%M"),
                    tone="purple",
                    supporting_text=f"Closes the 5-minute view at {_format_wait(end_slot['wait_min'])}.",
                ),
                unsafe_allow_html=True,
            )

        forecast_table = pd.DataFrame(
            {
                "Time": pf["ds"].dt.strftime("%H:%M"),
                "Predicted Arrivals": pd.to_numeric(pf["arrivals"], errors="coerce"),
                "Estimated Wait (min)": pd.to_numeric(pf["wait_min"], errors="coerce"),
            }
        )
        forecast_table["Status"] = forecast_table["Estimated Wait (min)"].apply(_status_label)

        styled_forecast = (
            forecast_table.style
            .format(
                {
                    "Predicted Arrivals": "{:.1f}",
                    "Estimated Wait (min)": lambda value: "—" if pd.isna(value) else f"{value:.1f}",
                }
            )
            .hide(axis="index")
            .apply(_forecast_row_style, axis=1)
        )

        st.dataframe(
            styled_forecast,
            use_container_width=True,
            height=min(420, 45 + (len(forecast_table) * 38)),
            column_config={
                "Time": st.column_config.TextColumn(width="small"),
                "Predicted Arrivals": st.column_config.NumberColumn(width="medium"),
                "Estimated Wait (min)": st.column_config.NumberColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("No forecast rows are available inside the next 30-minute display window.")
else:
    st.warning(
        "No forecast is available yet. Use the Run Prediction button above to generate the next lane-aware forecast."
    )


# ── Lane scenarios ────────────────────────────────────────────────────────────

_section_title("Lane Scenarios", "wait at +15 min across forecast lane options")
lane_summary = (
    f"Detected queue state is {_lane_phrase(detected_lanes)}. Dashboard forecast parameter is {_lane_phrase(selected_lanes)}."
)
st.markdown(f'<div class="detail-note">{lane_summary}</div>', unsafe_allow_html=True)

if lane_waits:
    scenario_lane_counts = list(range(1, max(3, selected_lanes) + 1))
    lane_cols = st.columns(len(scenario_lane_counts))
    for col, lane_count in zip(lane_cols, scenario_lane_counts):
        with col:
            wait_value = lane_waits.get(lane_count, {}).get("wait_15m")
            is_active = lane_count == selected_lanes
            is_default = lane_count == detected_lanes and selected_lanes == detected_lanes
            st.markdown(
                _lane_card_html(lane_count, wait_value, is_active=is_active, is_default=is_default),
                unsafe_allow_html=True,
            )
else:
    st.info("Run the prediction model to populate lane scenario comparisons.")


# ── Live queue + dwell history ────────────────────────────────────────────────

history_range_hours = int(st.session_state["history_range_hours"])
_section_title(
    f"Live Queue And Dwell - Last {_hour_phrase(history_range_hours).title()}",
    "people in queue and average dwell time",
)
history_ctrl_col, history_note_col = st.columns([2, 5])
with history_ctrl_col:
    st.select_slider(
        "History range",
        options=list(range(1, 13)),
        value=history_range_hours,
        format_func=lambda value: _hour_phrase(value),
        key="history_range_hours",
    )
with history_note_col:
    st.markdown(
        f'<div class="detail-note">Showing the last {_hour_phrase(history_range_hours)} of queue snapshots. '
        'Top panel shows queue depth in people. Bottom panel shows average dwell time in minutes from the same live snapshots, so you can see whether congestion is coming from more people, slower service, or both.</div>',
        unsafe_allow_html=True,
    )

if not queue_hist.empty:
    qh = queue_hist.copy()
    qh["timestamp"] = _to_local_series(qh["timestamp"])
    qh["queue_count"] = pd.to_numeric(qh["queue_count"], errors="coerce")
    qh["avg_dwell_sec"] = pd.to_numeric(qh["avg_dwell_sec"], errors="coerce")
    qh = qh.set_index("timestamp").resample("1min").mean().dropna(subset=["queue_count"], how="all").reset_index()
    qh["avg_dwell_min"] = qh["avg_dwell_sec"] / 60.0

    has_dwell = qh["avg_dwell_min"].notna().any()
    fig_q = make_subplots(
        rows=2 if has_dwell else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.66, 0.34] if has_dwell else None,
        subplot_titles=["Queue depth (people)", "Average dwell (min)"] if has_dwell else ["Queue depth (people)"],
    )
    fig_q.add_trace(
        go.Scatter(
            x=qh["timestamp"],
            y=qh["queue_count"],
            mode="lines",
            line=dict(color="#16a34a", width=3),
            fill="tozeroy",
            fillcolor="rgba(22, 163, 74, 0.14)",
            hovertemplate="%{x|%H:%M}<br>Queue depth: %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if has_dwell:
        fig_q.add_trace(
            go.Scatter(
                x=qh["timestamp"],
                y=qh["avg_dwell_min"],
                mode="lines",
                line=dict(color="#7c3aed", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124, 58, 237, 0.12)",
                hovertemplate="%{x|%H:%M}<br>Avg dwell: %{y:.2f} min<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig_q.update_layout(
        height=380 if has_dwell else 240,
        margin=dict(l=0, r=20, t=30, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        font=dict(color="#0f172a"),
    )
    fig_q.update_xaxes(
        showgrid=False,
        title="",
        tickfont=dict(size=11, color="#475569"),
        tickformat="%H:%M",
        hoverformat="%H:%M",
    )
    fig_q.update_yaxes(
        gridcolor="#e2e8f0",
        title="People",
        zeroline=False,
        tickfont=dict(size=11, color="#475569"),
        row=1,
        col=1,
    )
    if has_dwell:
        fig_q.update_yaxes(
            gridcolor="#e2e8f0",
            title="Min",
            zeroline=False,
            tickfont=dict(size=11, color="#475569"),
            row=2,
            col=1,
        )

    st.plotly_chart(fig_q, use_container_width=True)
else:
    st.info("No queue history is available for the last 2 hours.")


# ══════════════════════════════════════════════════════════════════════════════
#  TODAY'S OVERVIEW — daily tracker
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<hr class="tracker-divider">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="tracker-heading">
  Today's Overview
  <span class="tracker-sub">since store open · updates every {REFRESH_SEC}s</span>
</div>
""",
    unsafe_allow_html=True,
)

d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    st.markdown(
        _metric_card_html(
            "Peak Queue Today",
            str(peak_queue_today),
            tone="red",
            supporting_text=f"Peak reached at {peak_time_str}.",
        ),
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        _metric_card_html(
            "Avg Queue Today",
            f"{avg_queue_today}",
            tone="blue",
            supporting_text="Average queue depth across today's snapshots.",
        ),
        unsafe_allow_html=True,
    )

with d3:
    delta_today = _delta_html(entries_vs_yesterday, " vs yesterday")
    st.markdown(
        _metric_card_html(
            "Entries Today",
            str(entries_today),
            tone="green",
            delta_html=delta_today,
            supporting_text="Tracked entrance events since midnight.",
        ),
        unsafe_allow_html=True,
    )

with d4:
    st.markdown(
        _metric_card_html(
            "Busiest Hour",
            busiest_hour_str,
            tone="orange",
            supporting_text="Hour bucket with the highest entrance count.",
        ),
        unsafe_allow_html=True,
    )

with d5:
    alert_tone = "red" if alert_minutes_today >= 30 else "orange" if alert_minutes_today >= 15 else "green"
    st.markdown(
        _metric_card_html(
            "Minutes in Alert",
            f"{alert_minutes_today}<span class='metric-unit'> min</span>",
            tone=alert_tone,
            supporting_text="Minutes already classified as alert today.",
        ),
        unsafe_allow_html=True,
    )


# ── Full-day queue chart ───────────────────────────────────────────────────────

_section_title("Queue Depth - Full Day", "store-wide trend since midnight")

if not full_day_queue.empty:
    fdq = full_day_queue.copy()
    fdq["timestamp"] = _to_local_series(fdq["timestamp"])
    fdq = fdq.set_index("timestamp").resample("5min").mean().dropna().reset_index()

    fig_fd = go.Figure()
    fig_fd.add_trace(
        go.Scatter(
            x=fdq["timestamp"],
            y=fdq["queue_count"],
            mode="lines",
            line=dict(color="#7c3aed", width=3),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.12)",
            hovertemplate="%{x|%H:%M}<br>Queue depth: %{y:.1f}<extra></extra>",
        )
    )
    _apply_chart_layout(fig_fd, height=240, xaxis_type="date")
    st.plotly_chart(fig_fd, use_container_width=True)
else:
    st.info("No queue data has been recorded today yet.")


# ── Entries by hour + Status breakdown (two columns) ──────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    _section_title("Entries by Hour", "traffic pattern for today")
    if not traffic_today.empty:
        tt = traffic_today.copy()
        tt["hour"] = _to_local_series(tt["hour"])
        tt["hour_label"] = tt["hour"].dt.strftime("%H:%M")
        max_val = tt["entries"].max()
        colors = ["#dc2626" if value == max_val else "#2563eb" for value in tt["entries"]]

        fig_t = go.Figure()
        fig_t.add_trace(
            go.Bar(
                x=tt["hour_label"],
                y=tt["entries"],
                marker_color=colors,
                text=tt["entries"],
                textposition="outside",
                hovertemplate="%{x}<br>Entries: %{y}<extra></extra>",
            )
        )
        _apply_chart_layout(fig_t, height=280)
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No entry data is available for today yet.")

with col_right:
    _section_title("Status Breakdown Today", "minutes spent in each state")
    if not status_breakdown.empty:
        status_colors = {"OK": "#16a34a", "BUSY": "#ea580c", "ALERT": "#dc2626"}
        sb = status_breakdown.copy()
        sb["status"] = sb["status"].str.upper()
        sb["minutes"] = sb["slots"] * BUCKET_MIN

        fig_donut = go.Figure(
            go.Pie(
                labels=sb["status"],
                values=sb["minutes"],
                hole=0.6,
                marker_colors=[status_colors.get(status, "#94a3b8") for status in sb["status"]],
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value} min<extra></extra>",
            )
        )
        fig_donut.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=18, b=0),
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#0f172a"),
            annotations=[
                dict(
                    text=f"<b>{sb['minutes'].sum()}<br>min</b>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#0f172a"),
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No prediction history is available for today yet.")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Auto-refresh every {REFRESH_SEC}s · Busy threshold: {WAIT_BUSY_MIN} min · "
    f"Alert threshold: {WAIT_ALERT_MIN} min · Default lanes: {DEFAULT_LANES} · "
    f"+45 min uses {RUSH_HOUR_LANES} lanes (rush-hour)"
)