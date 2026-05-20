import os
import subprocess
import sys
import psycopg2
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

CAMERA_ID      = os.getenv("CAM_ID",          "Bosch_Camera_Entrance")
REFRESH_SEC    = int(os.getenv("REFRESH_SEC",  30))
WAIT_BUSY_MIN  = float(os.getenv("WAIT_BUSY_MIN",  5.0))
WAIT_ALERT_MIN = float(os.getenv("WAIT_ALERT_MIN", 10.0))
BUCKET_MIN     = int(os.getenv("BUCKET_MINUTES", 3))

DB_CONFIG = dict(
    host     = os.getenv("DB_HOST",     "localhost"),
    port     = int(os.getenv("DB_PORT", 5432)),
    dbname   = os.getenv("DB_NAME",     "iqms"),
    user     = os.getenv("DB_USER",     "postgres"),
    password = os.getenv("DB_PASSWORD", "0000"),
)

def _conn():
    return psycopg2.connect(**DB_CONFIG)


def _run_prediction():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_predict.py")
    result = subprocess.run(
        [sys.executable, script, "--source", "REAL"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Live data ──────────────────────────────────────────────────────────────────

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
        try:
            df = pd.read_sql("""
                SELECT DISTINCT ON (prediction_for)
                       prediction_for  AS ds,
                       ensemble_yhat   AS arrivals,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, wait_45m, status,
                       wait_1lane_15m, wait_2lane_15m, wait_3lane_15m
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '30 minutes'
                ORDER BY prediction_for, predicted_at DESC
            """, conn)
        except Exception:
            conn.rollback()
            df = pd.read_sql("""
                SELECT DISTINCT ON (prediction_for)
                       prediction_for  AS ds,
                       ensemble_yhat   AS arrivals,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, status
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '30 minutes'
                ORDER BY prediction_for, predicted_at DESC
            """, conn)
            for col in ["wait_45m", "wait_1lane_15m", "wait_2lane_15m", "wait_3lane_15m"]:
                df[col] = None
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


# ── Daily tracker data ─────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SEC)
def load_entries_today():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id = %s AND timestamp >= CURRENT_DATE
            """, (CAMERA_ID,))
            return cur.fetchone()[0]

@st.cache_data(ttl=REFRESH_SEC)
def load_yesterday_entries():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id = %s
                  AND timestamp >= CURRENT_DATE - INTERVAL '1 day'
                  AND timestamp < CURRENT_DATE
            """, (CAMERA_ID,))
            return cur.fetchone()[0]

@st.cache_data(ttl=REFRESH_SEC)
def load_today_traffic():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS entries
            FROM entrance_events
            WHERE camera_id = %s AND timestamp >= CURRENT_DATE
            GROUP BY 1 ORDER BY 1
        """, conn, params=(CAMERA_ID,))
    return df

@st.cache_data(ttl=REFRESH_SEC)
def load_full_day_queue():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT timestamp, queue_count
            FROM queue_state_snapshots
            WHERE timestamp >= CURRENT_DATE
            ORDER BY timestamp ASC
        """, conn)
    return df

@st.cache_data(ttl=REFRESH_SEC)
def load_status_breakdown():
    with _conn() as conn:
        df = pd.read_sql("""
            SELECT UPPER(status) AS status, COUNT(*) AS slots
            FROM queue_predictions
            WHERE prediction_for >= CURRENT_DATE
              AND prediction_for < NOW()
            GROUP BY status
        """, conn)
    return df


# ── Page config ────────────────────────────────────────────────────────────────

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
  .metric-card.purple { border-top-color: #9b59b6; }
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
  .tracker-heading {
    font-size: 1.1rem; font-weight: 800; color: #2c3e50;
    margin: 8px 0 16px 0; letter-spacing: -0.01em;
  }
  .tracker-sub {
    font-size: 0.75rem; font-weight: 400; color: #aaa; margin-left: 10px;
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
  .tracker-divider {
    margin: 36px 0 24px 0;
    border: none; border-top: 3px solid #ecf0f1;
  }
</style>
""", unsafe_allow_html=True)


# ── Load all data ──────────────────────────────────────────────────────────────

try:
    snap             = load_snapshot()
    pred_df          = load_predictions()
    queue_hist       = load_queue_history()
    traffic_today    = load_today_traffic()
    entries_today    = load_entries_today()
    yesterday_entries = load_yesterday_entries()
    queue_1h_ago     = load_queue_delta()
    entries_delta    = load_entries_delta()
    full_day_queue   = load_full_day_queue()
    status_breakdown = load_status_breakdown()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

queue_count  = int(snap["queue_count"])       if snap is not None else 0
active_lanes = int(snap["active_lanes"] or 2) if snap is not None else 2
snap_ts      = snap["timestamp"]              if snap is not None else None

wait_15m   = float(pred_df.iloc[0]["wait_15m"])  if not pred_df.empty and pred_df.iloc[0]["wait_15m"]  is not None else None
wait_30m   = float(pred_df.iloc[0]["wait_30m"])  if not pred_df.empty and pred_df.iloc[0]["wait_30m"]  is not None else None
wait_45m   = float(pred_df.iloc[0]["wait_45m"])  if not pred_df.empty and pred_df.iloc[0].get("wait_45m") is not None else None
est_wait   = float(pred_df.iloc[0]["wait_min"])  if not pred_df.empty and pred_df.iloc[0]["wait_min"]  is not None else wait_15m
lane_w1    = float(pred_df.iloc[0]["wait_1lane_15m"]) if not pred_df.empty and pred_df.iloc[0].get("wait_1lane_15m") is not None else None
lane_w2    = float(pred_df.iloc[0]["wait_2lane_15m"]) if not pred_df.empty and pred_df.iloc[0].get("wait_2lane_15m") is not None else None
lane_w3    = float(pred_df.iloc[0]["wait_3lane_15m"]) if not pred_df.empty and pred_df.iloc[0].get("wait_3lane_15m") is not None else None

entries_last_hr = entries_delta[0] if entries_delta else 0
entries_prev_hr = entries_delta[1] if entries_delta else 0
entries_diff    = entries_last_hr - entries_prev_hr
queue_diff      = (queue_count - queue_1h_ago) if queue_1h_ago is not None else None

# Daily tracker KPIs
_tz = datetime.now(timezone.utc).astimezone().tzinfo

if not full_day_queue.empty:
    peak_queue_today = int(full_day_queue["queue_count"].max())
    avg_queue_today  = round(float(full_day_queue["queue_count"].mean()), 1)
    peak_row = full_day_queue.loc[full_day_queue["queue_count"].idxmax()]
    peak_ts  = pd.Timestamp(peak_row["timestamp"])
    if peak_ts.tzinfo is not None:
        peak_ts = peak_ts.tz_convert(_tz)
    peak_time_str = peak_ts.strftime("%H:%M")
else:
    peak_queue_today = 0
    avg_queue_today  = 0.0
    peak_time_str    = "--"

entries_vs_yesterday = int(entries_today) - int(yesterday_entries)

if not traffic_today.empty:
    busiest_idx = traffic_today["entries"].idxmax()
    busiest_ts  = pd.Timestamp(traffic_today.loc[busiest_idx, "hour"])
    if busiest_ts.tzinfo is not None:
        busiest_ts = busiest_ts.tz_convert(_tz)
    busiest_hour_str = busiest_ts.strftime("%H:%M")
else:
    busiest_hour_str = "--"

alert_slots = 0
if not status_breakdown.empty:
    alert_row = status_breakdown[status_breakdown["status"].str.upper() == "ALERT"]
    if not alert_row.empty:
        alert_slots = int(alert_row.iloc[0]["slots"])
alert_minutes_today = alert_slots * BUCKET_MIN

if est_wait is None:
    status_class, status_text = "status-ok",    "System OK - forecast not available"
elif est_wait >= WAIT_ALERT_MIN:
    status_class, status_text = "status-alert", f"ALERT - Estimated wait {est_wait:.0f} min"
elif est_wait >= WAIT_BUSY_MIN:
    status_class, status_text = "status-busy",  f"BUSY - Estimated wait {est_wait:.0f} min"
else:
    status_class, status_text = "status-ok",    f"All Clear - Estimated wait {est_wait:.0f} min"


# ── Header ─────────────────────────────────────────────────────────────────────

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


# ── Live metric cards ──────────────────────────────────────────────────────────

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
w45_color = "green" if (wait_45m or 0) < WAIT_BUSY_MIN else "orange" if (wait_45m or 0) < WAIT_ALERT_MIN else "red"
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
      <div class="metric-lbl">Est. Wait +15 min</div>
    </div>""", unsafe_allow_html=True)

with c3:
    w30_disp = f"{wait_30m:.1f}" if wait_30m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w30_color}">
      <div class="metric-val">{w30_disp}<span style="font-size:1.1rem; font-weight:400"> min</span></div>
      <div class="metric-lbl">Est. Wait +30 min</div>
    </div>""", unsafe_allow_html=True)

with c4:
    w45_disp = f"{wait_45m:.1f}" if wait_45m is not None else "-"
    st.markdown(f"""
    <div class="metric-card {w45_color}">
      <div class="metric-val">{w45_disp}<span style="font-size:1.1rem; font-weight:400"> min</span></div>
      <div class="metric-lbl">Est. Wait +45 min</div>
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
        ts_local = ts_obj.tz_convert(_tz)
    else:
        ts_local = ts_obj
    st.caption(f"Queue snapshot: {ts_local.strftime('%H:%M:%S  %d/%m/%Y')}")


# ── Predicted wait chart ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">Predicted Wait Time - Next 30 Min</div>', unsafe_allow_html=True)

if not pred_df.empty:
    pf = pred_df.copy()
    pf["ds"] = pd.to_datetime(pf["ds"])
    if pf["ds"].dt.tz is not None:
        pf["ds"] = pf["ds"].dt.tz_convert(_tz)
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
    _msg_col, _btn_col = st.columns([4, 1])
    with _msg_col:
        st.warning("No forecast available yet. Predictions may not have run today, or all existing rows have expired.")
    with _btn_col:
        if st.button("Run Now", key="run_no_pred", type="primary", use_container_width=True):
            with st.spinner("Running prediction model... (1–3 min)"):
                _ok, _out, _err = _run_prediction()
            if _ok:
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Prediction failed.")
                if _err:
                    with st.expander("Error details"):
                        st.code(_err[-3000:])


# ── Lane scenarios ────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Lane Scenarios — Estimated Wait at +15 min</div>', unsafe_allow_html=True)

if lane_w1 is not None or lane_w2 is not None or lane_w3 is not None:
    st.caption(f"Currently active: {active_lanes} lane{'s' if active_lanes != 1 else ''}")
    _lc1, _lc2, _lc3 = st.columns(3)
    for _col, _n, _w in [(_lc1, 1, lane_w1), (_lc2, 2, lane_w2), (_lc3, 3, lane_w3)]:
        with _col:
            _is_current = (_n == active_lanes)
            _border = "#9b59b6" if _is_current else "#3498db"
            _w_disp = f"{_w:.1f}" if _w is not None else "-"
            _badge_txt = "ALERT" if (_w or 0) >= WAIT_ALERT_MIN else "BUSY" if (_w or 0) >= WAIT_BUSY_MIN else "OK"
            _badge_cls = "badge-alert" if _badge_txt == "ALERT" else "badge-busy" if _badge_txt == "BUSY" else "badge-ok"
            _active_note = " &nbsp;<small style='color:#9b59b6;font-weight:700'>← current</small>" if _is_current else ""
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{_border}">
              <div class="metric-val">{_w_disp}<span style="font-size:1.1rem;font-weight:400"> min</span></div>
              <div class="metric-lbl">{_n} Lane{'s' if _n != 1 else ''} Open{_active_note}</div>
              <div style="margin-top:8px"><span class="badge {_badge_cls}">{_badge_txt}</span></div>
            </div>""", unsafe_allow_html=True)
else:
    st.info("Run the prediction model to see lane scenario comparisons.")


# ── Live queue depth (last 2h) ─────────────────────────────────────────────────

st.markdown('<div class="section-title">Live Queue Depth - Last 2 Hours</div>', unsafe_allow_html=True)

if not queue_hist.empty:
    qh = queue_hist.copy()
    qh["timestamp"] = pd.to_datetime(qh["timestamp"])
    if qh["timestamp"].dt.tz is not None:
        qh["timestamp"] = qh["timestamp"].dt.tz_convert(_tz)
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


# ══════════════════════════════════════════════════════════════════════════════
#  TODAY'S OVERVIEW — daily tracker
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<hr class="tracker-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="tracker-heading">
  Today's Overview
  <span class="tracker-sub">since store open · updates every 30s</span>
</div>
""", unsafe_allow_html=True)

# Daily KPI cards
d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    st.markdown(f"""
    <div class="metric-card red">
      <div class="metric-val">{peak_queue_today}</div>
      <div class="metric-lbl">Peak Queue Today</div>
      <div class="metric-delta" style="color:#999">at {peak_time_str}</div>
    </div>""", unsafe_allow_html=True)

with d2:
    st.markdown(f"""
    <div class="metric-card blue">
      <div class="metric-val">{avg_queue_today}</div>
      <div class="metric-lbl">Avg Queue Today</div>
    </div>""", unsafe_allow_html=True)

with d3:
    yest_color = "#27ae60" if entries_vs_yesterday >= 0 else "#e74c3c"
    yest_arrow = "+" if entries_vs_yesterday >= 0 else ""
    yest_html  = f'<div class="metric-delta" style="color:{yest_color}">{yest_arrow}{entries_vs_yesterday} vs yesterday</div>'
    st.markdown(f"""
    <div class="metric-card green">
      <div class="metric-val">{entries_today}</div>
      <div class="metric-lbl">Entries Today</div>
      {yest_html}
    </div>""", unsafe_allow_html=True)

with d4:
    st.markdown(f"""
    <div class="metric-card orange">
      <div class="metric-val">{busiest_hour_str}</div>
      <div class="metric-lbl">Busiest Hour</div>
    </div>""", unsafe_allow_html=True)

with d5:
    alert_color = "red" if alert_minutes_today >= 30 else "orange" if alert_minutes_today >= 15 else "green"
    st.markdown(f"""
    <div class="metric-card {alert_color}">
      <div class="metric-val">{alert_minutes_today}<span style="font-size:1.1rem; font-weight:400"> min</span></div>
      <div class="metric-lbl">In Alert Today</div>
    </div>""", unsafe_allow_html=True)


# ── Full-day queue chart ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">Queue Depth - Full Day</div>', unsafe_allow_html=True)

if not full_day_queue.empty:
    fdq = full_day_queue.copy()
    fdq["timestamp"] = pd.to_datetime(fdq["timestamp"])
    if fdq["timestamp"].dt.tz is not None:
        fdq["timestamp"] = fdq["timestamp"].dt.tz_convert(_tz)
    fdq = fdq.set_index("timestamp").resample("5min").mean().dropna().reset_index()
    fdq["time_str"] = fdq["timestamp"].dt.strftime("%H:%M")

    fig_fd = go.Figure()
    fig_fd.add_trace(go.Scatter(
        x=fdq["time_str"], y=fdq["queue_count"],
        mode="lines", name="Queue depth",
        line=dict(color="#9b59b6", width=2.5),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.12)",
    ))
    fig_fd.update_layout(
        height=220, margin=dict(l=0, r=20, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=11), title=""),
        yaxis=dict(gridcolor="#f0f2f6", title="", zeroline=False),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig_fd, use_container_width=True)
else:
    st.info("No queue data recorded today yet.")


# ── Entries by hour + Status breakdown (two columns) ──────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("<div class='section-title'>Entries by Hour</div>", unsafe_allow_html=True)
    if not traffic_today.empty:
        tt = traffic_today.copy()
        tt["hour"] = pd.to_datetime(tt["hour"])
        if tt["hour"].dt.tz is not None:
            tt["hour"] = tt["hour"].dt.tz_convert(_tz)
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
            height=260, margin=dict(l=0, r=20, t=20, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=11), title=""),
            yaxis=dict(gridcolor="#f0f2f6", title="", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No entry data for today yet.")

with col_right:
    st.markdown("<div class='section-title'>Status Breakdown Today</div>", unsafe_allow_html=True)
    if not status_breakdown.empty:
        STATUS_COLORS = {"OK": "#2ecc71", "BUSY": "#e67e22", "ALERT": "#e74c3c"}
        sb = status_breakdown.copy()
        sb["status"]  = sb["status"].str.upper()
        sb["minutes"] = sb["slots"] * BUCKET_MIN

        fig_donut = go.Figure(go.Pie(
            labels=sb["status"],
            values=sb["minutes"],
            hole=0.55,
            marker_colors=[STATUS_COLORS.get(s, "#95a5a6") for s in sb["status"]],
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value} min<extra></extra>",
        ))
        fig_donut.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, paper_bgcolor="white",
            annotations=[dict(
                text=f"<b>{sb['minutes'].sum()}<br>min</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#2c3e50"),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No prediction history for today yet.")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
col_r, col_p, col_b = st.columns([5, 1, 1])
with col_b:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_p:
    if st.button("Run Prediction", use_container_width=True, type="primary"):
        with st.spinner("Running prediction model... (1–3 min)"):
            _ok, _out, _err = _run_prediction()
        if _ok:
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Prediction failed.")
            if _err:
                with st.expander("Error details"):
                    st.code(_err[-3000:])
with col_r:
    st.caption(
        f"Auto-refreshes every {REFRESH_SEC}s  -  "
        f"Busy: {WAIT_BUSY_MIN} min  -  Alert: {WAIT_ALERT_MIN} min"
    )
