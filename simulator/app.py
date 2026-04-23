"""
Queue Simulator — Streamlit Dashboard
──────────────────────────────────────
Run with:
    streamlit run simulator/app.py
    streamlit run simulator/app.py -- --calibrate --days 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# ── Path setup (must be before local imports) ─────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib, sys
from simulator.calibrate import calibrate
from simulator.scenarios import SCENARIOS, get_scenario
import simulator.predict_viz as _pv_mod
from prediction import pipeline as pred_pipeline
# Force Streamlit hot-reloads to use the current source, not a cached module.
if "simulator.predict_viz" in sys.modules:
    importlib.reload(sys.modules["simulator.predict_viz"])
import simulator.predict_viz as predict_viz

BACKEND_HOST  = os.getenv("BACKEND_HOST",  "127.0.0.1")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8081))


def _fragment(func):
    fragment_api = getattr(st, "fragment", None)
    if fragment_api is None:
        return func
    return fragment_api(func)


def _sidebar_toggle(label: str, value: bool = False, key: str | None = None):
    toggle_api = getattr(st.sidebar, "toggle", None)
    if toggle_api is not None:
        return toggle_api(label, value=value, key=key)
    return st.sidebar.checkbox(label, value=value, key=key)


# ── Page config MUST be first Streamlit call ──────────────────────────────────
st.set_page_config(layout="wide", page_title="Queue Simulator 2D")

tab_sim, tab_pred = st.tabs(["🛒 Simulator", "📈 Prediction Training"])


# ── CLI args (--calibrate --days N passed after `--`) ─────────────────────────
def _parse_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    args, _ = parser.parse_known_args()
    return args


cli_args = _parse_cli()


@st.cache_data(show_spinner="Loading calibration from DB…")
def load_calibration(days: int):
    return calibrate(days=days)


# ── Load calibration (safe, after page_config) ────────────────────────────────
calibration_cfg: dict = {}
if cli_args.calibrate:
    with st.spinner("Calibrating from real DB data…"):
        try:
            calibration_cfg = load_calibration(cli_args.days)
        except Exception as exc:
            st.warning(f"⚠️ Calibration failed — using scenario defaults. ({exc})")


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛒 Queue Management & Flow Simulator")
if calibration_cfg:
    st.caption(
        f"**Calibrated** from last **{cli_args.days} days** of REAL data "
        f"(generated {calibration_cfg.get('calibrated_at', '?')[:19]})"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
scenario_name = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), key="scenario_name")

default_lanes = int(calibration_cfg.get("num_lanes", SCENARIOS[scenario_name]["num_lanes"]))
default_gap   = int(round(calibration_cfg.get("avg_arrival_gap_seconds",
                                               SCENARIOS[scenario_name]["avg_arrival_gap_seconds"])))
default_caddy_pct = int(round(100 * calibration_cfg.get("prob_caddy", SCENARIOS[scenario_name]["prob_caddy"])))
default_bakery_pct = int(round(100 * calibration_cfg.get("prob_bakery", SCENARIOS[scenario_name]["prob_bakery"])))
default_start_dt = get_scenario(scenario_name)["start_time"]

start_mode = st.sidebar.radio(
    "Simulation start",
    ["Now", "Specific date/time"],
    key="start_mode",
)
if start_mode == "Specific date/time":
    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start_dt.date(),
        key="start_date",
    )
    start_time = st.sidebar.time_input(
        "Start time",
        value=default_start_dt.time(),
        key="start_time",
        step=300,
    )
    selected_start_dt = datetime.combine(start_date, start_time)
else:
    selected_start_dt = datetime.now()

scenario_preview = get_scenario(scenario_name)
open_hour = int(scenario_preview.get("open_hour", 8))
close_hour = int(scenario_preview.get("close_hour", 20))
effective_preview_dt = selected_start_dt
if not (open_hour <= selected_start_dt.hour < close_hour):
    effective_preview_dt = selected_start_dt.replace(
        hour=open_hour, minute=0, second=0, microsecond=0
    )
    if selected_start_dt.hour >= close_hour:
        effective_preview_dt += timedelta(days=1)

num_lanes   = st.sidebar.slider("Active Caisse Lanes", 1, 5, min(5, max(1, default_lanes)),   key="num_lanes")
arrival_gap = st.sidebar.slider("Avg Arrival Gap (sec)", 10, 300, min(300, max(10, default_gap)), key="arrival_gap")
caddy_pct   = st.sidebar.slider("Caddy customers (%)", 0, 100, min(100, max(0, default_caddy_pct)), key="caddy_pct")
bakery_pct  = st.sidebar.slider("Bakery customers (%)", 0, 100, min(100, max(0, default_bakery_pct)), key="bakery_pct")
sim_hours   = st.sidebar.slider("Sim Duration (hours)", 1, 12, 8,   key="sim_hours")
speed       = st.sidebar.slider("Simulation Speed Factor", 1, 200, 30, key="speed")
live_sync    = _sidebar_toggle("Live sync while running", value=False, key="live_sync")
persist_to_db = _sidebar_toggle(
    "Save simulation to DB",
    value=False,
    key="persist_to_db",
)
if persist_to_db:
    st.sidebar.caption(
        "DB write ON — rows tagged SIM_* will appear in the database. "
        "Use --source SIM or --source ALL to include them in forecasts."
    )
else:
    st.sidebar.caption("DB write OFF — simulation runs in memory only, no DB rows created.")
st.sidebar.caption("Closed-hour starts automatically move to the next opening time.")
st.sidebar.caption("Higher speed factor = more simulated time per real second.")
if start_mode == "Specific date/time":
    st.sidebar.caption(f"Requested start: {selected_start_dt:%Y-%m-%d %H:%M}")
st.sidebar.caption(f"Effective start: {effective_preview_dt:%Y-%m-%d %H:%M}")
st.sidebar.caption("Live update: lanes, arrival gap, caddy %, bakery %, speed.")
st.sidebar.caption("Next start only: scenario, start date/time, duration.")

# ── Spawning Background Services ──────────────────────────────────────────────
import socket
import subprocess
import json

def is_port_in_use(port):
    endpoints = [
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    ]
    for family, address in endpoints:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


def _find_port_pids(port: int) -> list[int]:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            pids = set()
            needle = f":{port}"
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and needle in parts[1] and parts[3].upper() == "LISTENING":
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
            return sorted(pids)
    except Exception:
        pass
    return []


def _stop_port_listener(port: int):
    for pid in _find_port_pids(port):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                os.kill(pid, 15)
        except Exception:
            pass


@st.cache_resource
def _start_frontend_server():
    """Bind a simple HTTP server on FRONTEND_PORT and serve the frontend dir.

    Decorated with st.cache_resource so the function body runs exactly once per
    Streamlit process. The returned httpd object is held by the cache, preventing
    garbage collection and keeping the daemon thread alive across all reruns.
    """
    import threading
    import http.server

    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend"))

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=frontend_dir, **kwargs)
        def log_message(self, *args: object) -> None:
            pass

    try:
        httpd = http.server.HTTPServer(("127.0.0.1", FRONTEND_PORT), _Handler)
    except OSError:
        # Port already bound — a previous run's server is still alive. Return a
        # sentinel so the cache doesn't store None (which would allow GC of httpd).
        return "already_running"

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd          # cache holds this reference — prevents GC


# Always call unconditionally so the server starts on every fresh process,
# even after hot-reloads that clear the cache while session_state persists.
_start_frontend_server()


def _start_local_services(force_restart: bool = False):
    if os.environ.get("SIM_EXTERNAL_SERVICES") == "1":
        return

    if force_restart:
        _stop_port_listener(8080)
        time.sleep(0.6)

    # WebSocket simulation engine (subprocess — has complex async dependencies)
    if not is_port_in_use(8080):
        server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))
        subprocess.Popen([sys.executable, server_path])
        time.sleep(1.0)


if "services_bootstrapped" not in st.session_state:
    _start_local_services(force_restart=True)
    st.session_state["services_bootstrapped"] = True

services_managed_externally = os.environ.get("SIM_EXTERNAL_SERVICES") == "1"

if services_managed_externally:
    st.sidebar.caption("Local backend is managed by `run_simulator.py`.")
elif st.sidebar.button("Restart local backend", width="stretch"):
    _start_local_services(force_restart=True)
    st.session_state["simulation_running"] = False
    st.session_state.pop("last_live_payload", None)
    st.toast("Local simulator backend restarted.", icon="🔄")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _send_ws(payload: dict, *, open_timeout: float = 0.5):
    try:
        # websockets >= 11
        from websockets.sync.client import connect as _sync_connect
        with _sync_connect(f"ws://{BACKEND_HOST}:8080", open_timeout=open_timeout) as ws:
            ws.send(json.dumps(payload))
    except ImportError:
        # websockets < 11: run async client in a dedicated thread to avoid
        # clashing with Streamlit's internal event loop.
        import asyncio, threading
        _exc: list = []

        async def _do():
            try:
                import websockets as _ws
                async with _ws.connect(  # type: ignore[attr-defined]
                    f"ws://{BACKEND_HOST}:8080",
                    open_timeout=open_timeout,
                ) as _wsc:
                    await _wsc.send(json.dumps(payload))
            except Exception as _e:
                _exc.append(_e)

        _t = threading.Thread(target=lambda: asyncio.run(_do()), daemon=True)
        _t.start()
        _t.join(timeout=open_timeout + 1.5)
        if _exc:
            raise _exc[0]


def _live_update_payload() -> dict:
    return {
        "action": "update_config",
        "lanes": num_lanes,
        "gap": arrival_gap,
        "prob_caddy": caddy_pct / 100.0,
        "prob_bakery": bakery_pct / 100.0,
        "speed": speed,
    }


def _maybe_push_live_updates():
    if not st.session_state.get("simulation_running"):
        return
    if not live_sync:
        return

    payload = _live_update_payload()
    payload_key = json.dumps(payload, sort_keys=True)
    if st.session_state.get("last_live_payload") == payload_key:
        return

    try:
        _send_ws(payload, open_timeout=0.25)
        st.session_state["last_live_payload"] = payload_key
    except Exception as exc:
        st.session_state["last_live_error"] = str(exc)


# ── Simulator tab — wrapped in @st.fragment so that sidebar slider changes
#    do NOT remount the iframe or reset the canvas JS state.
# ─────────────────────────────────────────────────────────────────────────────
@_fragment
def _sim_tab():
    # Use a raw <iframe> via st.markdown instead of st.iframe().
    # st.iframe() wraps the element in Streamlit's component infrastructure,
    # which manages height dynamically and causes a brief size glitch on every
    # fragment rerender.  A raw <iframe> in unsafe HTML is a static DOM element —
    # Streamlit never touches its dimensions after initial render.
    st.markdown(
        f'<iframe src="http://localhost:{FRONTEND_PORT}/index.html"'
        f' height="680"'
        f' frameborder="0" scrolling="no"'
        f' style="display:block;border:none;width:100%;"></iframe>',
        unsafe_allow_html=True,
    )

    c_start, c_apply, c_stop = st.columns(3)
    with c_start:
        if st.button("▶ Start Simulation", type="primary", width="stretch"):
            payload = {
                "action":    "start",
                "scenario":  scenario_name,
                "lanes":     num_lanes,
                "gap":       arrival_gap,
                "prob_caddy": caddy_pct / 100.0,
                "prob_bakery": bakery_pct / 100.0,
                "speed":     speed,
                "hours":     sim_hours,
                "start_mode": start_mode,
                "start_datetime": selected_start_dt.isoformat(),
                "calibrate": cli_args.calibrate,
                "calibration_days": cli_args.days,
                "persist":   persist_to_db,
            }
            try:
                _send_ws(payload)
                st.session_state["simulation_running"] = True
                st.session_state["last_live_payload"] = json.dumps(_live_update_payload(), sort_keys=True)
                st.session_state.pop("last_live_error", None)
                st.toast("Simulation started!", icon="▶")
            except Exception as e:
                st.error(f"Could not reach engine on :8080 — {e}")
    with c_apply:
        apply_disabled = not st.session_state.get("simulation_running")
        if st.button("↻ Apply Live Changes", width="stretch", disabled=apply_disabled):
            try:
                payload = _live_update_payload()
                _send_ws(payload, open_timeout=0.35)
                st.session_state["last_live_payload"] = json.dumps(payload, sort_keys=True)
                st.session_state.pop("last_live_error", None)
                st.toast("Live changes applied.", icon="🔄")
            except Exception as e:
                st.session_state["last_live_error"] = str(e)
                st.error(f"Could not apply live changes — {e}")
    with c_stop:
        if st.button("⏹ Stop", width="stretch"):
            try:
                _send_ws({"action": "stop"})
                st.session_state["simulation_running"] = False
                st.toast("Simulation stopped.", icon="⏹")
            except Exception as e:
                st.error(f"Could not reach engine on :8080 — {e}")

    if st.session_state.get("last_live_error"):
        st.caption(f"Live update issue: {st.session_state['last_live_error']}")


with tab_sim:
    _sim_tab()
    _maybe_push_live_updates()



# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Prediction Training
# ─────────────────────────────────────────────────────────────────────────────

def _pred_cache_key(source: str, days: int, use_bootstrap: bool,
                    data_since: "datetime | None") -> str:
    ds = data_since.isoformat() if data_since else ""
    return f"{source}|{days}|{use_bootstrap}|{ds}"


def _get_pred_result(source: str, days: int, use_bootstrap: bool,
                     interval_sec: int, force: bool,
                     data_since: "datetime | None" = None) -> dict:
    """Return a cached pipeline result, retraining when the interval has elapsed.

    Uses st.session_state so the TTL is fully dynamic — changing the interval
    slider takes effect immediately without redecorating a cached function.
    """
    key        = _pred_cache_key(source, days, use_bootstrap, data_since)
    cached_key = st.session_state.get("pred_cache_key")
    cached_at  = st.session_state.get("pred_cached_at", 0.0)
    elapsed    = time.time() - cached_at

    if force or cached_key != key or elapsed >= interval_sec:
        result = pred_pipeline.run_prediction_pipeline(
            source=source, days=days, use_bootstrap=use_bootstrap,
            data_since=data_since,
        )
        st.session_state["pred_result"]    = result
        st.session_state["pred_cache_key"] = key
        st.session_state["pred_cached_at"] = time.time()
    return st.session_state["pred_result"]  # type: ignore[return-value]


def _render_pred_results(result: dict, live_actuals: "pd.DataFrame | None" = None) -> None:
    w15 = result["wait_15m"] or 0.0
    banner_color = "#16a34a" if w15 < 5 else "#d97706" if w15 < 10 else "#dc2626"
    label        = "🟢 OK"  if w15 < 5 else "🟡 BUSY" if w15 < 10 else "🔴 ALERT"
    st.markdown(
        f'<div style="background:{banner_color};padding:10px 18px;border-radius:8px;'
        f'font-size:1.2em;font-weight:bold;color:white">'
        f'{label} — Predicted wait in 15 min: {w15} min</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Bag rate KPI (live query, last 24 h, REAL cameras only) ──────────────
    bag_rate = "—"
    try:
        import psycopg2 as _pg2
        _bg_conn = _pg2.connect(**pred_pipeline.DB_CONFIG)
        _bg = pd.read_sql("""
            SELECT
                COUNT(*) FILTER (WHERE has_bag = TRUE) AS with_bag,
                COUNT(*)                                AS total
            FROM entrance_events
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
              AND camera_id NOT LIKE 'SIM_%%'
        """, _bg_conn).iloc[0]
        _bg_conn.close()
        bag_rate = f"{100 * _bg['with_bag'] / max(_bg['total'], 1):.1f}%"
    except Exception:
        pass

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📦 Training rows",  len(result["df_combined"]))
    k2.metric("📅 Real data span", f"{result['real_span_days']:.0f} days")
    k3.metric("⏳ Wait @ +15 min", f"{w15} min")
    k4.metric("⏳ Wait @ +30 min", f"{result['wait_30m']} min")
    k5.metric("👜 Bag rate (24h)", bag_rate)

    st.subheader("Training Data")
    st.plotly_chart(predict_viz.fig_training_data(result, live_actuals),
                    use_container_width=True, key="fig_train")

    st.subheader("Today's Forecast")
    st.plotly_chart(predict_viz.fig_prophet_forecast(result, live_actuals),
                    use_container_width=True, key="fig_forecast")

    st.subheader("Seasonality Components")
    st.plotly_chart(predict_viz.fig_prophet_components(result),
                    use_container_width=True, key="fig_comp")

    st.subheader("Wait Estimates")
    st.plotly_chart(predict_viz.fig_wait_estimates(result),
                    use_container_width=True, key="fig_wait")

    st.subheader("Lane Planning — 1 to 5 Lanes")
    st.plotly_chart(predict_viz.fig_lane_scenarios(result),
                    use_container_width=True, key="fig_lanes")

    st.subheader("Queue History")
    st.plotly_chart(predict_viz.fig_queue_history(result),
                    use_container_width=True, key="fig_queue")

    st.subheader("Dwell & Shop Time Analysis")
    st.plotly_chart(predict_viz.fig_dwell_analysis(result),
                    use_container_width=True, key="fig_dwell")

    st.subheader("Demographics")
    st.plotly_chart(predict_viz.fig_demographics(result, live_actuals),
                    use_container_width=True, key="fig_demo")

    with st.expander("📊 Forecast table"):
        fc = result["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc.columns = ["Time", "Predicted", "Min (80%CI)", "Max (80%CI)"]
        fc["Time"] = fc["Time"].dt.strftime("%H:%M")
        st.dataframe(fc, use_container_width=True)


with tab_pred:
    st.header("📈 Prediction Training & Forecast")

    p_col1, p_col2, p_col3 = st.columns([2, 2, 2])
    with p_col1:
        pred_source = st.selectbox("Data source", ["REAL", "SIM", "ALL"], key="pred_source")
    with p_col2:
        pred_days = st.slider("History (days)", 1, 90, 30, key="pred_days")
    with p_col3:
        use_bootstrap = st.checkbox(
            "Bootstrap sparse data",
            value=False,
            key="use_bootstrap",
            help=(
                "Mix in 7 days of synthetic Poisson history when real data spans "
                "less than 14 days. OFF by default."
            ),
        )

    r_col1, r_col2, r_col3 = st.columns([2, 2, 1])
    with r_col1:
        retrain_interval_min = st.select_slider(
            "Auto-retrain every",
            options=[1, 5, 10, 15, 30, 60],
            value=15,
            key="retrain_interval_min",
            format_func=lambda v: f"{v} min",
        )
    with r_col2:
        use_data_since = st.checkbox("Filter data from date", value=False,
                                     key="use_data_since")
        data_since_date = st.date_input(
            "Data since",
            value=datetime.now().date() - timedelta(days=int(os.getenv("DATA_SINCE_DEFAULT_DAYS", 7))),
            key="data_since_date",
            disabled=not use_data_since,
        )
        data_since: "datetime | None" = (
            datetime.combine(data_since_date, datetime.min.time())
            if use_data_since else None
        )
    with r_col3:
        st.write("")
        force_retrain = st.button("🔄 Force Retrain", key="force_retrain")

    with st.spinner("Training Prophet model…"):
        try:
            result = _get_pred_result(
                pred_source, pred_days, use_bootstrap,
                interval_sec=retrain_interval_min * 60,
                force=force_retrain,
                data_since=data_since,
            )
            # Live actuals: always queried fresh so the chart extends to now
            # even while the Prophet model result is still cached.
            live_actuals = pred_pipeline.load_today_actuals(pred_source)

            cached_at  = st.session_state.get("pred_cached_at", time.time())
            trained_at = datetime.fromtimestamp(cached_at).strftime("%H:%M:%S")
            next_at    = datetime.fromtimestamp(
                cached_at + retrain_interval_min * 60
            ).strftime("%H:%M")
            last_bucket = (
                live_actuals["bucket"].max().strftime("%H:%M")
                if not live_actuals.empty else "—"
            )
            latest_bucket_count = "—"
            if not live_actuals.empty:
                _latest_bucket_ts = live_actuals["bucket"].max()
                latest_bucket_count = int(
                    live_actuals.loc[live_actuals["bucket"] == _latest_bucket_ts, "entry_count"].sum()
                )
            st.caption(
                f"Model trained at **{trained_at}** — next retrain at **{next_at}**. "
                f"Latest data in DB: **{last_bucket}** (**{latest_bucket_count}** entries / 3 min). "
                f"Click 🔄 to retrain immediately."
            )
            _render_pred_results(result, live_actuals)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    # ── Camera diagnostics ────────────────────────────────────────────────────
    with st.expander("🔍 Camera ID diagnostics — what's actually in the DB", expanded=False):
        try:
            import psycopg2 as _pg
            _conn = _pg.connect(**pred_pipeline.DB_CONFIG)

            # ── Per-camera totals ─────────────────────────────────────────────
            st.markdown("**Entries per camera_id**")
            _df = pd.read_sql("""
                SELECT camera_id,
                       COUNT(*) AS total_entries,
                       MAX(timestamp) AS latest_entry
                FROM entrance_events
                GROUP BY camera_id
                ORDER BY total_entries DESC
                LIMIT 30
            """, _conn)

            # ── Top spike buckets aggregated across all cameras ───────────────
            st.markdown("**Top 10 highest 3-min buckets (all cameras)**")
            _spikes = pd.read_sql("""
                WITH per_camera AS (
                    SELECT
                        time_bucket('3 minutes', timestamp) AS bucket,
                        camera_id,
                        COUNT(*) AS cnt
                    FROM entrance_events
                    GROUP BY bucket, camera_id
                ),
                ranked AS (
                    SELECT
                        bucket,
                        SUM(cnt) AS total_cnt,
                        MAX(cnt) AS peak_camera_cnt,
                        COUNT(*) AS camera_count
                    FROM per_camera
                    GROUP BY bucket
                    ORDER BY total_cnt DESC
                    LIMIT 10
                )
                SELECT *
                FROM ranked
                ORDER BY total_cnt DESC, bucket DESC
            """, _conn)
            if not _spikes.empty:
                _spikes["bucket"] = pd.to_datetime(_spikes["bucket"]).dt.tz_convert(None)
                from datetime import datetime as _dt, timezone as _tz
                _utc_off = _dt.now(_tz.utc).astimezone().utcoffset()
                _spikes["bucket_local"] = _spikes["bucket"] + _utc_off
                st.dataframe(
                    _spikes[["bucket_local", "total_cnt", "peak_camera_cnt", "camera_count"]]
                    .rename(columns={
                        "bucket_local": "bucket (local)",
                        "total_cnt": "total count",
                        "peak_camera_cnt": "largest single-camera count",
                        "camera_count": "cameras contributing",
                    }),
                    use_container_width=True,
                )
                _top_cnt = int(_spikes["total_cnt"].max())
                if _top_cnt > 100:
                    st.warning(
                        f"Peak 3-min bucket has **{_top_cnt} total entries** across all cameras. "
                        "If this looks unrealistic, check SIM rows and duplicate restart rows below."
                    )

            if _df.empty:
                _conn.close()
                st.info("No rows in entrance_events.")
            else:
                _df["is_sim"] = _df["camera_id"].str.startswith("SIM_", na=False)
                st.dataframe(_df[["camera_id", "total_entries", "latest_entry", "is_sim"]],
                             use_container_width=True)

                st.markdown("**All 3-min bucket values for one camera**")
                _camera_options = _df["camera_id"].dropna().astype(str).tolist()
                _selected_camera = st.selectbox(
                    "Select camera_id",
                    _camera_options,
                    key="diag_selected_camera",
                )
                _camera_buckets = pd.read_sql("""
                    SELECT
                        time_bucket('3 minutes', timestamp) AS bucket,
                        COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE camera_id = %s
                    GROUP BY bucket
                    ORDER BY bucket DESC
                """, _conn, params=(_selected_camera,))
                if not _camera_buckets.empty:
                    _camera_buckets["bucket"] = pd.to_datetime(_camera_buckets["bucket"]).dt.tz_convert(None)
                    from datetime import datetime as _dt, timezone as _tz
                    _utc_off = _dt.now(_tz.utc).astimezone().utcoffset()
                    _camera_buckets["bucket_local"] = _camera_buckets["bucket"] + _utc_off
                    _camera_total = int(_camera_buckets["cnt"].sum())
                    _camera_peak = int(_camera_buckets["cnt"].max())
                    _camera_latest_bucket = _camera_buckets["bucket_local"].max()
                    c_diag1, c_diag2, c_diag3 = st.columns(3)
                    c_diag1.metric("Selected camera rows", _camera_total)
                    c_diag2.metric("Peak 3-min bucket", _camera_peak)
                    c_diag3.metric(
                        "Latest bucket",
                        _camera_latest_bucket.strftime("%Y-%m-%d %H:%M"),
                    )
                    st.dataframe(
                        _camera_buckets[["bucket_local", "cnt"]]
                        .rename(columns={"bucket_local": "bucket (local)", "cnt": "count"}),
                        use_container_width=True,
                    )
                else:
                    st.info(f"No entrance_events rows found for `{_selected_camera}`.")

                non_sim_non_real = _df[~_df["is_sim"]]
                # Known bad SIM camera ID patterns (no SIM_ prefix)
                _known_sim_patterns = [
                    "normal_day", "lunch_rush", "evening_rush",
                    "WS_normal_day", "WS_lunch_rush", "WS_evening_rush",
                ]
                _bad_sim_ids = _df[
                    ~_df["is_sim"] &
                    _df["camera_id"].isin(_known_sim_patterns)
                ]["camera_id"].tolist()

                if _bad_sim_ids:
                    st.warning(
                        f"Found {len(_bad_sim_ids)} SIM camera ID(s) without the "
                        f"`SIM_` prefix: **{', '.join(_bad_sim_ids)}**. "
                        f"These are counted as REAL and inflate your training data."
                    )
                    if st.button("🔧 Fix SIM annotations in DB", key="fix_sim_ids"):
                        try:
                            _fix_conn = _pg.connect(**pred_pipeline.DB_CONFIG)
                            _cur = _fix_conn.cursor()
                            for _cid in _bad_sim_ids:
                                _cur.execute(
                                    "UPDATE entrance_events "
                                    "SET camera_id = %s WHERE camera_id = %s",
                                    (f"SIM_{_cid}", _cid)
                                )
                                _cur.execute(
                                    "UPDATE queue_state_snapshots "
                                    "SET camera_id = %s WHERE camera_id = %s",
                                    (f"SIM_{_cid}", _cid)
                                )
                            _fix_conn.commit()
                            _fix_conn.close()
                            st.success(
                                f"Fixed {len(_bad_sim_ids)} camera ID(s). "
                                f"Click 🔄 Force Retrain to reload clean data."
                            )
                        except Exception as _fe:
                            st.error(f"Fix failed: {_fe}")
                elif not non_sim_non_real.empty:
                    st.info(
                        "Non-SIM_ camera IDs found but not in the known bad-pattern list. "
                        "These are likely real cameras. If any look wrong, rename them manually."
                    )

                # ── Deduplicate restart entries ───────────────────────────────
                st.markdown("---")
                st.markdown(
                    "**Duplicate restart entries** — removes rows where the same "
                    "`track_id / camera_id / day` was inserted more than once (happens "
                    "when the detector restarts mid-day)."
                )
                if st.button("🧹 Run cleanup", key="run_dedup"):
                    try:
                        _clean_conn = _pg.connect(**pred_pipeline.DB_CONFIG)
                        _clean_cur = _clean_conn.cursor()
                        _clean_cur.execute("""
                            DELETE FROM entrance_events
                            WHERE camera_id NOT LIKE 'SIM_%%'
                              AND id NOT IN (
                                  SELECT MIN(id)
                                  FROM entrance_events
                                  WHERE camera_id NOT LIKE 'SIM_%%'
                                  GROUP BY track_id, camera_id, DATE(timestamp)
                              )
                        """)
                        _deleted = _clean_cur.rowcount
                        _clean_conn.commit()
                        _clean_conn.close()
                        if _deleted > 0:
                            st.success(
                                f"Removed **{_deleted} duplicate entries** from restart cycles. "
                                f"Click 🔄 Force Retrain to reload clean data."
                            )
                        else:
                            st.success("No duplicate restart entries found.")
                    except Exception as _ce:
                        st.error(f"Cleanup failed: {_ce}")
                _conn.close()
        except Exception as _e:
            st.error(f"Diagnostic query failed: {_e}")
