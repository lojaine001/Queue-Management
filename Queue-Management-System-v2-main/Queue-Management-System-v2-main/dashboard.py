import os
import psycopg2
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

CAMERA_ID      = os.getenv("CAM_ID",          "Bosch_Camera_Entrance")
REFRESH_SEC    = int(os.getenv("REFRESH_SEC",  30))
WAIT_BUSY_MIN  = float(os.getenv("WAIT_BUSY_MIN",  5.0))
WAIT_ALERT_MIN = float(os.getenv("WAIT_ALERT_MIN", 10.0))

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
def load_queue_delta():
    with _conn() as conn:
        row = pd.read_sql("""
            SELECT queue_count FROM queue_state_snapshots
            WHERE timestamp <= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC LIMIT 1
        """, conn)
    return int(row.iloc[0]["queue_count"]) if not row.empty else None

@st.cache_data(ttl=REFRESH_SEC)
def load_predictions():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT prediction_for  AS ds,
                   ensemble_yhat   AS arrivals,
                   est_wait_minutes AS wait_min,
                   wait_15m, wait_30m, status
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
            GROUP BY 1 ORDER BY 1
        """, conn, params=(CAMERA_ID,))
    return df

@st.cache_data(ttl=REFRESH_SEC)
def load_entries_delta():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '1 hour') AS last_hour,
                    COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '2 hours'
                                      AND timestamp <  NOW() - INTERVAL '1 hour')  AS prev_hour
                FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= NOW() - INTERVAL '2 hours'
            """, (CAMERA_ID,))
            return cur.fetchone()

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

st.set_page_config(page_title="IQMS - Live Dashboard", page_icon="📊", layout="wide")

st.markdown(f"""
<script>
  setTimeout(function() {{ window.location.reload(); }}, {REFRESH_SEC * 1000});
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f0f2f6; }
  [data-testid="stHeader"] { background: transparent; }
  .top-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; border-radius: 16px; padding: 24px 32px; margin-bottom: 20px;
  }
  .metric-card {
    background: white; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-top: 4px solid #3498db;
    min-height: 115px;
  }
  .metric-card.green  { border-top-color: #2ecc71; }
  .metric-card.orange { border-top-color: #e67e22; }
  .metric-card.red    { border-top-color: #e74c3c; }
  .metric-card.blue   { border-top-color: #3498db; }
  .metric-val  { font-size: 2.4rem; font-weight: 800; line-height: 1.1; color: #2c3e50; }
  .metric-lbl  { font-size: 0.72rem; color: #999; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }
  .metric-delta { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }
  .status-banner {
    border-radius: 14px; padding: 14px 28px; text-align: center;
    font-size: 1.3rem; font-weight: 700; margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12); letter-spacing: 0.02em;
  }
  .status-ok    { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }
  .status-busy  { background: linear-gradient(135deg, #e67e22, #d35400); color: white; }
  .status-alert { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
  .section-title {
    font-size: 0.78rem; font-weight: 700; color: #7f8c8d;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 28px 0 10px 0;
    padding-bottom: 8px; border-bottom: 2px solid #ecf0f1;
  }
  .forecast-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .forecast-table th {
    background: #f8f9fa; color: #7f8c8d; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 10px 14px; text-align: left; border-bottom: 2px solid #ecf0f1;
  }
  .forecast-table td { padding: 9px 14px; border-bottom: 1px solid #f0f2f6; color: #2c3e50; }
  .forecast-table tr:hover td { background: #f8fbff; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
  }
  .badge-ok    { background: #d5f5e3; color: #1e8449; }
  .badge-busy  { background: #fdebd0; color: #d35400; }
  .badge-alert { background: #fadbd8; color: #c0392b; }
</style>
""", unsafe_allow_html=True)


# -- Load data -----------------------------------------------------------------

try:
    snap          = load_snapshot()
    pred_df       = load_predictions()
    queue_hist    = load_queue_history()
    traffic_today = load_today_traffic()
    entries_today = load_entries_today()
    queue_1h_ago  = load_queue_delta()
    entries_delta = load_entries_delta()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

queue_count  = int(snap["queue_count"])       if snap is not None else 0
active_lanes = int(snap["active_lanes"] or 2) if snap is not None else 2
snap_ts      = snap["timestamp"]              if snap is not None else None

wait_15m = float(pred_df.iloc[0]["wait_15m"]) if not pred_df.empty and pred_df.iloc[0]["wait_15m"] is not None else None
wait_30m = float(pred_df.iloc[0]["wait_30m"]) if not pred_df.empty and pred_df.iloc[0]["wait_30m"] is not None else None

entries_last_hr = entries_delta[0] if entries_delta else 0
entries_prev_hr = entries_delta[1] if entries_delta else 0
entries_diff    = entries_last_hr - entries_prev_hr
queue_diff      = (queue_count - queue_1h_ago) if queue_1h_ago is not None else None

if wait_15m is None:
    status_class, status_text = "status-ok",    "System OK - forecast not available"
elif wait_15m >= WAIT_ALERT_MIN:
    status_class, status_text = "status-alert", f"ALERT - Estimated wait {wait_15m:.0f} min in 15 minutes"
elif wait_15m >= WAIT_BUSY_MIN:
    status_class, status_text = "status-busy",  f"BUSY - Estimated wait {wait_15m:.0f} min in 15 minutes"
else:
    status_class, status_text = "status-ok",    f"All Clear - Estimated wait {wait_15m:.0f} min in 15 minutes"


# -- Header --------------------------------------------------------------------

now_str = datetime.now().strftime("%H:%M  -  %A, %d %B %Y")
st.markdown(f"""
<div class="top-header">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div>
      <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.01em;">IQMS Live Dashboard</div>
      <div style="font-size:0.82rem; opacity:0.6; margin-top:5px;">
        {CAMERA_ID} &nbsp;-&nbsp; auto-refresh every {REFRESH_SEC}s
      </div>
    </div>
    <div style="text-align:right; font-size:0.88rem; opacity:0.75; font-weight:500;">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="status-banner {status_class}">{status_text}</div>', unsafe_allow_html=True)


# -- Metric cards --------------------------------------------------------------

def delta_html(diff, unit="", invert=False):
    if diff is None:
        return ""
    good  = diff <= 0 if invert else diff >= 0
    color = "#27ae60" if good else "#e74c3c"
    arrow = "+" if diff >= 0 else ""
    return f'<div class="metric-delta" style="color:{color}">{arrow}{diff}{unit} vs 1h ago</div>'

q_color   = "green" if queue_count <= 5 else "orange" if queue_count <= 10 else "red"
w15_color = "green" if (wait_15m or 0) < WAIT_BUSY_MIN else "orange" if (wait_15m or 0) < WAIT_ALERT_MIN else "red"
w30_color = "green" if (wait_30m or 0) < WAIT_BUSY_MIN else "orange" if (wait_30m or 0) < WAIT_ALERT_MIN else "red"
l_color   = "red" if active_lanes == 1 else "orange" if active_lanes == 2 else "green"
e_color   = "green" if entries_diff >= 0 else "orange"

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card {q_color}">
      <div class="metric-val">{queue_count}</div>
      <div class="metric-lbl">In Queue Now</div>
      {delta_html(queue_diff, invert=True)}
    </div>""", unsafe_allow_html=True)

with c2:
    w15_disp = f"{wait_15m:.1f}" if wait_15m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w15_color}">
      <div class="metric-val">{w15_disp}<span style="font-size:1.1rem; font-weight:400"> min</span></div>
      <div class="metric-lbl">Est. Wait in 15 min</div>
    </div>""", unsafe_allow_html=True)

with c3:
    w30_disp = f"{wait_30m:.1f}" if wait_30m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w30_color}">
      <div class="metric-val">{w30_disp}<span style="font-size:1.1rem; font-weight:400"> min</span></div>
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
    <div class="metric-card {e_color}">
      <div class="metric-val">{entries_today}</div>
      <div class="metric-lbl">Entries Today</div>
      {delta_html(entries_diff, " this hr")}
    </div>""", unsafe_allow_html=True)

if snap_ts is not None:
    ts_obj = pd.Timestamp(snap_ts)
    if ts_obj.tzinfo is not None:
        ts_local = ts_obj.tz_convert(datetime.now(timezone.utc).astimezone().tzinfo)
    else:
        ts_local = ts_obj
    st.caption(f"Queue snapshot: {ts_local.strftime('%H:%M:%S  %d/%m/%Y')}")


# -- Predicted wait chart (Plotly) ---------------------------------------------

st.markdown('<div class="section-title">Predicted Wait Time - Next 60 Min</div>', unsafe_allow_html=True)

if not pred_df.empty:
    pf = pred_df.copy()
    pf["ds"] = pd.to_datetime(pf["ds"])
    if pf["ds"].dt.tz is not None:
        pf["ds"] = pf["ds"].dt.tz_convert(datetime.now(timezone.utc).astimezone().tzinfo)
    pf["time_str"] = pf["ds"].dt.strftime("%H:%M")

    fig_wait = go.Figure()
    fig_wait.add_trace(go.Scatter(
        x=pf["time_str"], y=pf["wait_min"],
        mode="lines", name="Est. wait (min)",
        line=dict(color="#3498db", width=2.5),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
    ))
    fig_wait.add_hline(
        y=WAIT_BUSY_MIN, line=dict(color="#e67e22", width=1.5, dash="dot"),
        annotation_text="Busy", annotation_position="top right",
        annotation_font=dict(color="#e67e22", size=11),
    )
    fig_wait.add_hline(
        y=WAIT_ALERT_MIN, line=dict(color="#e74c3c", width=1.5, dash="dot"),
        annotation_text="Alert", annotation_position="top right",
        annotation_font=dict(color="#e74c3c", size=11),
    )
    fig_wait.update_layout(
        height=240, margin=dict(l=0, r=60, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=11), title=""),
        yaxis=dict(gridcolor="#f0f2f6", ticksuffix=" min", title="", zeroline=False),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig_wait, use_container_width=True)

    # Forecast detail table with colored badges
    st.markdown('<div class="section-title">Forecast Detail</div>', unsafe_allow_html=True)

    rows_html = ""
    for _, row in pf.iterrows():
        w = float(row["wait_min"])
        if w >= WAIT_ALERT_MIN:
            badge = '<span class="badge badge-alert">ALERT</span>'
        elif w >= WAIT_BUSY_MIN:
            badge = '<span class="badge badge-busy">BUSY</span>'
        else:
            badge = '<span class="badge badge-ok">OK</span>'
        rows_html += f"""
        <tr>
          <td><strong>{row["time_str"]}</strong></td>
          <td>{float(row["arrivals"]):.1f}</td>
          <td>{w:.1f} min</td>
          <td>{badge}</td>
        </tr>"""

    st.markdown(f"""
    <table class="forecast-table">
      <thead><tr>
        <th>Time</th>
        <th>Predicted Arrivals</th>
        <th>Est. Wait</th>
        <th>Status</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table><br>
    """, unsafe_allow_html=True)
else:
    st.info("No predictions found. Run ensemble_predict.py to generate a forecast.")


# -- Live queue depth (Plotly) -------------------------------------------------

st.markdown('<div class="section-title">Live Queue Depth - Last 2 Hours</div>', unsafe_allow_html=True)

if not queue_hist.empty:
    qh = queue_hist.copy()
    qh["timestamp"] = pd.to_datetime(qh["timestamp"])
    if qh["timestamp"].dt.tz is not None:
        qh["timestamp"] = qh["timestamp"].dt.tz_convert(datetime.now(timezone.utc).astimezone().tzinfo)
    qh = qh.set_index("timestamp").resample("1min").mean().dropna().reset_index()
    qh["time_str"] = qh["timestamp"].dt.strftime("%H:%M")

    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(
        x=qh["time_str"], y=qh["queue_count"],
        mode="lines", name="Queue depth",
        line=dict(color="#2ecc71", width=2.5),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.12)",
    ))
    fig_q.update_layout(
        height=200, margin=dict(l=0, r=20, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=11), title=""),
        yaxis=dict(gridcolor="#f0f2f6", title="", zeroline=False),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig_q, use_container_width=True)
else:
    st.info("No queue history available.")


# -- Today traffic by hour (Plotly) --------------------------------------------

st.markdown("<div class='section-title'>Today's Entries by Hour</div>", unsafe_allow_html=True)

if not traffic_today.empty:
    tt = traffic_today.copy()
    tt["hour"] = pd.to_datetime(tt["hour"])
    if tt["hour"].dt.tz is not None:
        tt["hour"] = tt["hour"].dt.tz_convert(datetime.now(timezone.utc).astimezone().tzinfo)
    tt["hour_label"] = tt["hour"].dt.strftime("%H:%M")
    max_val = tt["entries"].max()
    colors  = ["#e74c3c" if v == max_val else "#3498db" for v in tt["entries"]]

    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(
        x=tt["hour_label"], y=tt["entries"],
        marker_color=colors, name="Entries",
        text=tt["entries"], textposition="outside",
        textfont=dict(size=10),
    ))
    fig_t.update_layout(
        height=220, margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=11), title=""),
        yaxis=dict(gridcolor="#f0f2f6", title="", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("No entry data for today.")


# -- Footer --------------------------------------------------------------------

st.divider()
col_r, col_b = st.columns([6, 1])
with col_b:
    if st.button("Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
with col_r:
    st.caption(
        f"Auto-refreshes every {REFRESH_SEC}s  -  "
        f"Busy: {WAIT_BUSY_MIN} min  -  Alert: {WAIT_ALERT_MIN} min"
    )