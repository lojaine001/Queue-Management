import os
import psycopg2
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

CAMERA_ID       = os.getenv("CAM_ID",          "Bosch_Camera_Entrance")
REFRESH_SEC     = int(os.getenv("REFRESH_SEC",  30))
WAIT_BUSY_MIN   = float(os.getenv("WAIT_BUSY_MIN",  5.0))
WAIT_ALERT_MIN  = float(os.getenv("WAIT_ALERT_MIN", 10.0))

DB_CONFIG = dict(
    host     = os.getenv("DB_HOST",     "localhost"),
    port     = int(os.getenv("DB_PORT", 5432)),
    dbname   = os.getenv("DB_NAME",     "iqms"),
    user     = os.getenv("DB_USER",     "postgres"),
    password = os.getenv("DB_PASSWORD", "0000"),
)


def _conn():
    return psycopg2.connect(**DB_CONFIG)


@st.cache_data(ttl=REFRESH_SEC)
def load_snapshot():
    with _conn() as conn:
        row = pd.read_sql("""
            SELECT queue_count, avg_dwell_sec, active_lanes, timestamp
            FROM queue_state_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """, conn)
    return row.iloc[0] if not row.empty else None


@st.cache_data(ttl=REFRESH_SEC)
def load_predictions():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT prediction_for        AS ds,
                   ensemble_yhat         AS arrivals,
                   est_wait_minutes      AS wait_min,
                   wait_15m,
                   wait_30m,
                   status
            FROM queue_predictions
            WHERE prediction_for >= NOW()
              AND prediction_for <= NOW() + INTERVAL '90 minutes'
            ORDER BY prediction_for ASC
        """, conn)
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_queue_history():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT timestamp, queue_count
            FROM queue_state_snapshots
            WHERE timestamp >= NOW() - INTERVAL '2 hours'
            ORDER BY timestamp ASC
        """, conn)
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_today_traffic():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS entries
            FROM entrance_events
            WHERE camera_id = %s
              AND timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY 1
            ORDER BY 1
        """, conn, params=(CAMERA_ID,))
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_entries_today():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= NOW() - INTERVAL '24 hours'
            """, (CAMERA_ID,))
            return cur.fetchone()[0]


# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="IQMS - Live Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(f"""
<script>
  setTimeout(function() {{ window.location.reload(); }}, {REFRESH_SEC * 1000});
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f0f2f6; }
  .top-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;
  }
  .metric-card {
    background: white; border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 4px solid #3498db;
    height: 120px;
  }
  .metric-card.green  { border-top-color: #2ecc71; }
  .metric-card.orange { border-top-color: #e67e22; }
  .metric-card.red    { border-top-color: #e74c3c; }
  .metric-card.blue   { border-top-color: #3498db; }
  .metric-val { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
  .metric-lbl { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
  .status-banner {
    border-radius: 14px; padding: 16px 28px; text-align: center;
    font-size: 1.4rem; font-weight: 700; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
  }
  .status-ok    { background: #2ecc71; color: white; }
  .status-busy  { background: #e67e22; color: white; }
  .status-alert { background: #e74c3c; color: white; }
  .section-title {
    font-size: 1rem; font-weight: 600; color: #2c3e50;
    text-transform: uppercase; letter-spacing: 0.06em; margin: 24px 0 12px 0;
  }
</style>
""", unsafe_allow_html=True)


# -- Load data -----------------------------------------------------------------

try:
    snap          = load_snapshot()
    pred_df       = load_predictions()
    queue_hist    = load_queue_history()
    traffic_today = load_today_traffic()
    entries_today = load_entries_today()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

queue_count  = int(snap["queue_count"])          if snap is not None else 0
active_lanes = int(snap["active_lanes"] or 2)    if snap is not None else 2
snap_ts      = snap["timestamp"]                 if snap is not None else None

wait_15m = float(pred_df.iloc[0]["wait_15m"])  if not pred_df.empty and pred_df.iloc[0]["wait_15m"] is not None else None
wait_30m = float(pred_df.iloc[0]["wait_30m"])  if not pred_df.empty and pred_df.iloc[0]["wait_30m"] is not None else None

if wait_15m is None:
    status_class = "status-ok"
    status_text  = "System OK - no forecast available"
elif wait_15m >= WAIT_ALERT_MIN:
    status_class = "status-alert"
    status_text  = f"ALERT - Est. wait {wait_15m:.0f} min in 15 min"
elif wait_15m >= WAIT_BUSY_MIN:
    status_class = "status-busy"
    status_text  = f"BUSY - Est. wait {wait_15m:.0f} min in 15 min"
else:
    status_class = "status-ok"
    status_text  = f"OK - Est. wait {wait_15m:.0f} min in 15 min"


# -- Header --------------------------------------------------------------------

now_str = datetime.now().strftime("%H:%M  .  %A %d %B %Y")
st.markdown(f"""
<div class="top-header">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div style="font-size:1.6rem; font-weight:700;">IQMS Live Dashboard</div>
      <div style="font-size:0.85rem; opacity:0.7; margin-top:4px;">
        {CAMERA_ID} &nbsp;.&nbsp; auto-refresh every {REFRESH_SEC}s
      </div>
    </div>
    <div style="text-align:right; font-size:0.9rem; opacity:0.8;">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="status-banner {status_class}">{status_text}</div>',
            unsafe_allow_html=True)


# -- Metric cards --------------------------------------------------------------

q_color  = "green" if queue_count  <= 5 else "orange" if queue_count  <= 10 else "red"
w15_color = "green" if (wait_15m or 0) < WAIT_BUSY_MIN else \
            "orange" if (wait_15m or 0) < WAIT_ALERT_MIN else "red"
w30_color = "green" if (wait_30m or 0) < WAIT_BUSY_MIN else \
            "orange" if (wait_30m or 0) < WAIT_ALERT_MIN else "red"
l_color  = "red" if active_lanes == 1 else "orange" if active_lanes == 2 else "green"

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card {q_color}">
      <div class="metric-val">{queue_count}</div>
      <div class="metric-lbl">In Queue Now</div>
    </div>""", unsafe_allow_html=True)

with c2:
    w15_disp = f"{wait_15m:.1f}" if wait_15m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w15_color}">
      <div class="metric-val">{w15_disp}<span style="font-size:1rem"> min</span></div>
      <div class="metric-lbl">Est. Wait in 15 min</div>
    </div>""", unsafe_allow_html=True)

with c3:
    w30_disp = f"{wait_30m:.1f}" if wait_30m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w30_color}">
      <div class="metric-val">{w30_disp}<span style="font-size:1rem"> min</span></div>
      <div class="metric-lbl">Est. Wait in 30 min</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card {l_color}">
      <div class="metric-val">{active_lanes}</div>
      <div class="metric-lbl">Active Lanes</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card blue">
      <div class="metric-val">{entries_today}</div>
      <div class="metric-lbl">Entries Today</div>
    </div>""", unsafe_allow_html=True)

if snap_ts is not None:
    ts_obj = pd.Timestamp(snap_ts)
    if ts_obj.tzinfo is not None:
        ts_local = ts_obj.tz_convert(datetime.now(timezone.utc).astimezone().tzinfo)
    else:
        ts_local = ts_obj
    st.caption(f"Queue snapshot: {ts_local.strftime('%H:%M:%S  %d/%m/%Y')}")


# -- Predicted wait timeline ---------------------------------------------------

st.markdown('<div class="section-title">Predicted Wait Time - Next 60 Min</div>',
            unsafe_allow_html=True)

if not pred_df.empty:
    chart_df = pred_df[["ds", "wait_min"]].copy()
    chart_df["ds"] = pd.to_datetime(chart_df["ds"])
    chart_df = chart_df.set_index("ds")

    combined = chart_df.rename(columns={"wait_min": "Est. wait (min)"})
    combined["Busy threshold"]  = WAIT_BUSY_MIN
    combined["Alert threshold"] = WAIT_ALERT_MIN

    st.line_chart(combined, color=["#3498db", "#e67e22", "#e74c3c"], height=220)

    st.markdown('<div class="section-title">Forecast Detail</div>', unsafe_allow_html=True)
    tbl = pred_df.copy()
    tbl["ds"] = pd.to_datetime(tbl["ds"]).dt.strftime("%H:%M")

    def _status(w):
        if w >= WAIT_ALERT_MIN: return "ALERT"
        if w >= WAIT_BUSY_MIN:  return "BUSY"
        return "OK"

    tbl["Status"] = tbl["wait_min"].apply(_status)
    tbl = tbl.rename(columns={
        "ds":       "Time",
        "arrivals": "Arrivals",
        "wait_min": "Est. Wait (min)",
    })
    cols_order = [c for c in ["Time", "Arrivals", "Est. Wait (min)", "Status"]
                  if c in tbl.columns]
    st.dataframe(tbl[cols_order], use_container_width=True, hide_index=True)
else:
    st.info("No predictions found. Run ensemble_predict.py to generate a forecast.")


# -- Live queue depth ----------------------------------------------------------

st.markdown('<div class="section-title">Live Queue Depth - Last 2 Hours</div>',
            unsafe_allow_html=True)

if not queue_hist.empty:
    qh = queue_hist.copy()
    qh["timestamp"] = pd.to_datetime(qh["timestamp"])
    if qh["timestamp"].dt.tz is not None:
        qh["timestamp"] = qh["timestamp"].dt.tz_convert(
            datetime.now(timezone.utc).astimezone().tzinfo
        )
    qh = qh.set_index("timestamp").resample("1min").mean().dropna()
    st.line_chart(qh["queue_count"], color="#2ecc71", height=180)
else:
    st.info("No queue history available.")


# -- Today traffic by hour -----------------------------------------------------

st.markdown("<div class='section-title'>Today's Entries by Hour</div>",
            unsafe_allow_html=True)

if not traffic_today.empty:
    tt = traffic_today.copy()
    tt["hour"] = pd.to_datetime(tt["hour"])
    if tt["hour"].dt.tz is not None:
        tt["hour"] = tt["hour"].dt.tz_convert(
            datetime.now(timezone.utc).astimezone().tzinfo
        )
    tt["hour_label"] = tt["hour"].dt.strftime("%H:%M")
    st.bar_chart(tt.set_index("hour_label")["entries"], color="#3498db", height=200)
else:
    st.info("No entry data for today.")


# -- Footer --------------------------------------------------------------------

st.divider()
col_r, col_b = st.columns([6, 1])
with col_b:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_r:
    st.caption(
        f"Auto-refreshes every {REFRESH_SEC}s  .  "
        f"Busy threshold: {WAIT_BUSY_MIN} min  .  Alert: {WAIT_ALERT_MIN} min"
    )