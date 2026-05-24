import os
import psycopg2
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

CAMERA_ID = os.getenv("CAM_ID", "Bosch_Camera_Entrance")
MAX_LANES  = 4

# ── DB ────────────────────────────────────────────────────────────────────────

def _conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST",     "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME",   "iqms"),
        user=os.getenv("DB_USER",     "postgres"),
        password=os.getenv("DB_PASSWORD", "0000"),
    )

def ensure_table():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lane_events (
                    id           BIGSERIAL,
                    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    camera_id    VARCHAR(100),
                    active_lanes INT         NOT NULL,
                    note         TEXT
                )
            """)
            try:
                cur.execute("""
                    SELECT create_hypertable('lane_events','timestamp',
                        if_not_exists=>TRUE, migrate_data=>TRUE)
                """)
            except Exception:
                pass

def get_current(camera_id):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT active_lanes, timestamp, note
                FROM lane_events WHERE camera_id=%s
                ORDER BY timestamp DESC LIMIT 1
            """, (camera_id,))
            return cur.fetchone()

def set_lanes(camera_id, lanes, note):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lane_events (camera_id, active_lanes, note)
                VALUES (%s,%s,%s)
            """, (camera_id, lanes, note or None))

def get_history(camera_id, limit=50):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, active_lanes, note
                FROM lane_events WHERE camera_id=%s
                ORDER BY timestamp DESC LIMIT %s
            """, (camera_id, limit))
            return cur.fetchall()

def get_latest_snapshot(camera_id):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT queue_count, avg_dwell_sec, max_dwell_sec, timestamp
                FROM queue_state_snapshots WHERE camera_id=%s
                ORDER BY timestamp DESC LIMIT 1
            """, (camera_id,))
            return cur.fetchone()

def get_today_entries(camera_id):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM entrance_events
                WHERE camera_id=%s
                AND timestamp >= NOW() - INTERVAL '24 hours'
            """, (camera_id,))
            return cur.fetchone()[0]

def get_lane_history_today(camera_id):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, active_lanes
                FROM lane_events WHERE camera_id=%s
                AND timestamp >= NOW() - INTERVAL '12 hours'
                ORDER BY timestamp ASC
            """, (camera_id,))
            return cur.fetchall()


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IQMS — Lane Manager",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f0f2f6; }
    .top-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white; border-radius: 16px; padding: 24px 32px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: white; border-radius: 14px; padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #3498db;
        height: 110px;
    }
    .metric-card.green  { border-top-color: #2ecc71; }
    .metric-card.orange { border-top-color: #e67e22; }
    .metric-card.red    { border-top-color: #e74c3c; }
    .metric-card.blue   { border-top-color: #3498db; }
    .metric-val  { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .metric-lbl  { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .lane-card {
        border-radius: 14px; padding: 22px 12px; text-align: center;
        font-weight: 700; font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .lane-open   { background: #2ecc71; color: white; }
    .lane-closed { background: #dfe6e9; color: #b2bec3; }
    .section-title {
        font-size: 1rem; font-weight: 600; color: #2c3e50;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin: 24px 0 12px 0;
    }
    div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Data load ─────────────────────────────────────────────────────────────────

try:
    ensure_table()
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

current      = get_current(CAMERA_ID)
snapshot     = get_latest_snapshot(CAMERA_ID)
today_count  = get_today_entries(CAMERA_ID)
lane_history = get_lane_history_today(CAMERA_ID)

current_lanes = current[0] if current else 2
last_ts       = current[1] if current else None
last_note     = current[2] if current else ""

queue_count   = snapshot[0] if snapshot else 0
avg_dwell     = snapshot[1] if snapshot else 0
max_dwell     = snapshot[2] if snapshot else 0
snap_ts       = snapshot[3] if snapshot else None

avg_wait_min  = round(avg_dwell / 60, 1) if avg_dwell else 0
max_wait_min  = round(max_dwell / 60, 1) if max_dwell else 0


# ── Header ────────────────────────────────────────────────────────────────────

now_str = datetime.now().strftime("%H:%M  ·  %A %d %B %Y")
st.markdown(f"""
<div class="top-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:1.6rem; font-weight:700;">🛒 Checkout Lane Manager</div>
            <div style="font-size:0.85rem; opacity:0.7; margin-top:4px;">
                IQMS &nbsp;·&nbsp; {CAMERA_ID}
            </div>
        </div>
        <div style="text-align:right; font-size:0.9rem; opacity:0.8;">{now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Top metric cards ──────────────────────────────────────────────────────────

lane_color = "red" if current_lanes == 1 else "orange" if current_lanes == 2 else "green"
queue_color = "green" if queue_count <= 5 else "orange" if queue_count <= 10 else "red"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card {lane_color}">
        <div class="metric-val">{current_lanes}</div>
        <div class="metric-lbl">Active Lanes</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card {queue_color}">
        <div class="metric-val">{queue_count}</div>
        <div class="metric-lbl">In Queue Now</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-val">{avg_wait_min}<span style="font-size:1rem"> min</span></div>
        <div class="metric-lbl">Avg Wait Time</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-val">{today_count}</div>
        <div class="metric-lbl">Entries Today</div>
    </div>""", unsafe_allow_html=True)

if last_ts:
    local_ts = last_ts.astimezone().strftime("%H:%M  %d/%m/%Y")
    note_str = f"  ·  *{last_note}*" if last_note else ""
    st.caption(f"Last lane update: **{local_ts}**{note_str}")


# ── Lane status visual ────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Lane Status</div>', unsafe_allow_html=True)

cols = st.columns(MAX_LANES)
for i, col in enumerate(cols, start=1):
    is_open = i <= current_lanes
    css     = "lane-open" if is_open else "lane-closed"
    status  = "OPEN" if is_open else "CLOSED"
    icon    = "✅" if is_open else "⬜"
    col.markdown(f"""
    <div class="lane-card {css}">
        <div style="font-size:1.6rem">{icon}</div>
        <div style="font-size:1.1rem; margin:6px 0">Lane {i}</div>
        <div style="font-size:0.8rem; opacity:0.85">{status}</div>
    </div>""", unsafe_allow_html=True)


# ── Lane selector ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Update Active Lanes</div>', unsafe_allow_html=True)

note_input = st.text_input(
    "Note (optional)",
    placeholder="e.g. lunch rush, cashier absent, self-checkout only…",
    key="note_input",
    label_visibility="collapsed",
)

btn_cols = st.columns(MAX_LANES)
for i, col in enumerate(btn_cols, start=1):
    is_active = (current_lanes == i)
    label     = f"{'✓  ' if is_active else ''}{i} Lane{'s' if i > 1 else ''}"
    with col:
        if st.button(label, key=f"btn_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            try:
                set_lanes(CAMERA_ID, i, note_input.strip() or None)
                st.toast(f"Updated to {i} lane{'s' if i > 1 else ''} ✓", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

st.caption("Click to log the current number of open checkout lanes. Change is recorded immediately in the database.")


# ── Lane history chart ────────────────────────────────────────────────────────

if lane_history:
    st.markdown('<div class="section-title">Lane Count — Last 12 Hours</div>',
                unsafe_allow_html=True)

    df = pd.DataFrame(lane_history, columns=["timestamp", "active_lanes"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        datetime.now(timezone.utc).astimezone().tzinfo
    )

    # Step chart: each value holds until the next change
    df_step = pd.concat([
        df,
        pd.DataFrame({"timestamp": [datetime.now().astimezone()],
                      "active_lanes": [df["active_lanes"].iloc[-1]]}),
    ]).reset_index(drop=True)

    st.line_chart(
        df_step.set_index("timestamp")["active_lanes"],
        color="#2ecc71",
        height=180,
    )


# ── Recent changes table ──────────────────────────────────────────────────────

st.markdown('<div class="section-title">Recent Changes</div>', unsafe_allow_html=True)

history = get_history(CAMERA_ID, limit=20)

if history:
    rows = []
    for ts, lanes, note in history:
        local_ts = ts.astimezone().strftime("%H:%M  %d/%m/%Y")
        icon     = "🟢" if lanes <= 2 else "🟡" if lanes == 3 else "🔴"
        rows.append({
            "Time"         : local_ts,
            "Lanes"        : f"{icon}  {lanes}",
            "Note"         : note or "—",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No lane changes recorded yet.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
col_r, col_b = st.columns([6, 1])
with col_b:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
with col_r:
    if snap_ts:
        st.caption(f"Queue data last updated: {snap_ts.astimezone().strftime('%H:%M:%S')}")
