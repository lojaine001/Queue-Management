import os
import json
import time as _time
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", ".*SQLAlchemy.*")
warnings.filterwarnings("ignore", ".*DataFrame concatenation with empty or all-NA.*")
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from plotly.subplots import make_subplots
from streamlit.components.v1 import html as st_html

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
except Exception:
    tf = None
    TF_AVAILABLE = False

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
    SERVICE_MIN_DWELL_SEC,
    SHOP_OPEN,
    SHOP_CLOSE,
    SHOP_OPEN_TOT,
    SHOP_CLOSE_TOT,
    WAIT_15M_INDEX,
    WAIT_30M_INDEX,
    WAIT_45M_INDEX,
    _day_hours,
    compute_wait_estimates,
    is_open,
)
from prediction.pipeline import load_lane_history_agg  # noqa: E402
from prediction.dwell_modifier import compute_dwell_modifier_from_recent  # noqa: E402

CAMERA_ID = os.getenv("CAM_ID", "Bosch_Camera_Entrance")
ARRIVAL_CALIB_PATH = Path(__file__).parent / "calibration.json"


def _load_arrival_calibration_debug() -> dict[str, object]:
    if not ARRIVAL_CALIB_PATH.exists():
        return {
            "loaded": False,
            "path": str(ARRIVAL_CALIB_PATH),
            "k_global": 1.0,
            "k_by_hour": {},
            "error": None,
        }
    try:
        payload = json.loads(ARRIVAL_CALIB_PATH.read_text(encoding="utf-8"))
        return {
            "loaded": True,
            "path": str(ARRIVAL_CALIB_PATH),
            "k_global": float(payload.get("k_global", 1.0) or 1.0),
            "k_by_hour": {
                int(hour): float(value)
                for hour, value in payload.get("k_by_hour", {}).items()
            },
            "trained_at_utc": payload.get("trained_at_utc"),
            "error": None,
        }
    except Exception as exc:
        return {
            "loaded": False,
            "path": str(ARRIVAL_CALIB_PATH),
            "k_global": 1.0,
            "k_by_hour": {},
            "error": str(exc),
        }

def _today_hours():
    """Return (open_hour, open_min, close_hour, close_min, open_tot, close_tot) for today."""
    _oh, _om, _ch, _cm = _day_hours(pd.Timestamp.now().weekday())
    return _oh, _om, _ch, _cm, _oh * 60 + _om, _ch * 60 + _cm
# Scheduler interval drives everything time-based on this page: the
# @st.cache_data TTL below, and the page auto-refresh timer further down —
# so a browser reload always lands right when a new prediction is due.
if "sched_interval" not in st.session_state:
    _qp_interval = st.query_params.get("sched_interval", None)
    _interval_options = [5, 10, 15, 30, 60]
    if _qp_interval is not None and str(_qp_interval).isdigit() and int(_qp_interval) in _interval_options:
        st.session_state["sched_interval"] = int(_qp_interval)
    else:
        st.session_state["sched_interval"] = 15
REFRESH_SEC = int(st.session_state["sched_interval"]) * 60
WAIT_BUSY_MIN = float(os.getenv("WAIT_BUSY_MIN", 2.0))
WAIT_ALERT_MIN = float(os.getenv("WAIT_ALERT_MIN", 5.0))
BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))
PRED_SMOOTH_MIN = int(os.getenv("PRED_SMOOTH_MIN", 5))
FORECAST_DISPLAY_MIN = int(os.getenv("FORECAST_DISPLAY_MIN", 10))
DEFAULT_LANES = int(os.getenv("DEFAULT_LANES", 2))
MAX_LANES     = int(os.getenv("MAX_LANES", 4))
SNAPSHOT_MAX_AGE_MIN = int(os.getenv("SNAPSHOT_MAX_AGE_MIN", 30))
BROWSING_GAP_MIN = int(os.getenv("BROWSING_GAP_MIN", 25))
W_PROPHET = float(os.getenv("W_PROPHET", 0.40))
W_LSTM = float(os.getenv("W_LSTM", 0.30))
W_XGB = float(os.getenv("W_XGB", 0.30))
PROPHET_DISPLAY_SHIFT_MIN = int(os.getenv("PROPHET_DISPLAY_SHIFT_MIN", 0))
DWELL_MODEL_MODE_DEFAULT = os.getenv("DWELL_MODEL_MODE", "safe").strip().lower()
DWELL_MIN_TRAIN_BUCKETS = int(os.getenv("DWELL_MIN_TRAIN_BUCKETS", 24))
DWELL_MIN_TRAIN_EVENTS = int(os.getenv("DWELL_MIN_TRAIN_EVENTS", 60))
DWELL_LSTM_SEQ_LEN = int(os.getenv("DWELL_LSTM_SEQ_LEN", 8))
DWELL_LSTM_EPOCHS = int(os.getenv("DWELL_LSTM_EPOCHS", 12))
DWELL_LSTM_MIN_BUCKETS = int(os.getenv("DWELL_LSTM_MIN_BUCKETS", 48))
DWELL_LSTM_MIN_EVENTS = int(os.getenv("DWELL_LSTM_MIN_EVENTS", 120))

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


def _run_prediction(data_span_days: int = 30):
    _here  = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(_here, "ensemble_predict.py")
    cwd    = os.path.dirname(os.path.dirname(_here))   # Queue-Management/ for prediction imports
    env    = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    print(
        f"[Dashboard] Starting prediction subprocess: source=REAL days={data_span_days} "
        f"script={script} cwd={cwd}",
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, script, "--source", "REAL", "--days", str(data_span_days)],
        env=env,
        cwd=cwd,
    )
    print(f"[Dashboard] Prediction subprocess finished: returncode={result.returncode}", flush=True)
    return result.returncode == 0, "", ""


def _to_local_timestamp(value):
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(LOCAL_TZ).tz_localize(None)
    return ts


def _to_local_series(series):
    local = pd.to_datetime(series)
    if getattr(local.dt, "tz", None) is not None:
        local = local.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return local


def _now_local_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)


def _as_local_naive_ts(value):
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(LOCAL_TZ).tz_localize(None)
    return ts


def _plotly_local_iso(value):
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts = ts.tz_convert(LOCAL_TZ).tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _plotly_local_iso_series(values):
    return [_plotly_local_iso(v) for v in values]


def _build_dashboard_dwell_model(service_history: pd.DataFrame, mode: str):
    train = service_history.dropna(subset=["dwell_min"]).copy()
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No dwell history buckets."}

    train["dwell_min"] = pd.to_numeric(train["dwell_min"], errors="coerce")
    train["n_events"] = pd.to_numeric(train.get("n_events"), errors="coerce").fillna(1)
    train = train.dropna(subset=["dwell_min"])
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No valid dwell values."}

    train["ds"] = _to_local_series(train["ds"])
    train = train[train["ds"].apply(lambda t: is_open(pd.Timestamp(t)))]
    train["dwell_min"] = train["dwell_min"].clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)
    train["hour"] = train["ds"].apply(lambda t: pd.Timestamp(t).hour)
    train["minute_of_hour"] = train["ds"].apply(lambda t: pd.Timestamp(t).minute)
    train["day_of_week"] = train["ds"].apply(lambda t: pd.Timestamp(t).dayofweek)
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)

    bucket_count = int(len(train))
    event_count = int(train["n_events"].sum())
    if mode == "safe":
        if bucket_count < DWELL_MIN_TRAIN_BUCKETS or event_count < DWELL_MIN_TRAIN_EVENTS:
            return None, {
                "mode": mode,
                "status": "fallback",
                "reason": (
                    f"Need at least {DWELL_MIN_TRAIN_BUCKETS} dwell buckets and "
                    f"{DWELL_MIN_TRAIN_EVENTS} service events."
                ),
                "bucket_count": bucket_count,
                "event_count": event_count,
            }

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    fit_kwargs = {}
    if mode == "safe":
        fit_kwargs["sample_weight"] = train["n_events"].clip(lower=1).to_numpy(dtype=float)
    model.fit(train[_DWELL_FEAT], train["dwell_min"], **fit_kwargs)
    return model, {
        "mode": mode,
        "status": "trained",
        "reason": "Weighted bucket model." if mode == "safe" else "Legacy bucket model.",
        "bucket_count": bucket_count,
        "event_count": event_count,
    }


def _build_dashboard_dwell_lstm_model(service_history: pd.DataFrame, mode: str):
    if not TF_AVAILABLE:
        return None, {"mode": mode, "status": "unavailable", "reason": "TensorFlow not available."}

    train = service_history.dropna(subset=["dwell_min"]).copy()
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No dwell history buckets."}

    train["dwell_min"] = pd.to_numeric(train["dwell_min"], errors="coerce")
    train["n_events"] = pd.to_numeric(train.get("n_events"), errors="coerce").fillna(1)
    train = train.dropna(subset=["dwell_min"])
    if train.empty:
        return None, {"mode": mode, "status": "empty", "reason": "No valid dwell values."}

    train["ds"] = _to_local_series(train["ds"])
    train = train[train["ds"].apply(lambda t: is_open(pd.Timestamp(t)))]
    train = train.sort_values("ds").reset_index(drop=True)
    train["dwell_min"] = train["dwell_min"].clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)

    bucket_count = int(len(train))
    event_count = int(train["n_events"].sum())
    min_buckets = max(DWELL_MIN_TRAIN_BUCKETS, DWELL_LSTM_MIN_BUCKETS) if mode == "safe" else max(12, DWELL_LSTM_SEQ_LEN + 4)
    min_events = max(DWELL_MIN_TRAIN_EVENTS, DWELL_LSTM_MIN_EVENTS) if mode == "safe" else 1
    if bucket_count < min_buckets or event_count < min_events:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": f"Need at least {min_buckets} dwell buckets and {min_events} service events.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }

    values = train["dwell_min"].to_numpy(dtype=float)
    seq_len = max(4, min(DWELL_LSTM_SEQ_LEN, len(values) - 1))
    if len(values) <= seq_len:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": f"Need more than {seq_len} buckets for LSTM sequences.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    X_train, y_train = [], []
    for i in range(len(scaled) - seq_len):
        X_train.append(scaled[i:i + seq_len])
        y_train.append(scaled[i + seq_len])
    if not X_train:
        return None, {
            "mode": mode,
            "status": "fallback",
            "reason": "Not enough sequential dwell samples.",
            "bucket_count": bucket_count,
            "event_count": event_count,
        }

    X_arr = np.array(X_train, dtype=float).reshape(-1, seq_len, 1)
    y_arr = np.array(y_train, dtype=float)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, 1)),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_arr, y_arr, epochs=DWELL_LSTM_EPOCHS, batch_size=min(16, len(X_arr)), verbose=0)
    return {
        "model": model,
        "scaler": scaler,
        "seq_len": seq_len,
        "history_values": values,
    }, {
        "mode": mode,
        "status": "trained",
        "reason": "Sequential open-hour dwell model.",
        "bucket_count": bucket_count,
        "event_count": event_count,
    }


def _predict_dashboard_dwell_lstm(model_pack, slots) -> list[float]:
    if model_pack is None or not slots:
        return []
    values = np.array(model_pack["history_values"], dtype=float)
    seq_len = int(model_pack["seq_len"])
    scaler = model_pack["scaler"]
    model = model_pack["model"]
    if len(values) < seq_len:
        pad_val = float(np.median(values)) if len(values) else float(DEFAULT_DWELL_MIN)
        values = np.concatenate([np.full(seq_len - len(values), pad_val), values])
    history = values[-seq_len:].astype(float)
    preds = []
    for _ in slots:
        scaled_seq = scaler.transform(history.reshape(-1, 1)).reshape(1, seq_len, 1)
        next_scaled = float(model.predict(scaled_seq, verbose=0)[0][0])
        next_val = float(scaler.inverse_transform(np.array([[next_scaled]])).flatten()[0])
        next_val = float(np.clip(next_val, DWELL_MIN_FLOOR, DWELL_MAX_CAP))
        preds.append(next_val)
        history = np.append(history[1:], next_val)
    return preds


def _insert_overnight_gaps(df: pd.DataFrame, ds_col: str = "ds", threshold_min: int = 60) -> pd.DataFrame:
    """Insert a None row wherever consecutive timestamps gap > threshold_min.

    Plotly treats None as a break in the line, preventing cross-night connections.
    """
    if df.empty or len(df) < 2:
        return df
    df = df.sort_values(ds_col).reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    prev = 0
    _gap_row = df.iloc[0:1].copy()
    for _c in df.columns:
        _gap_row[_c] = pd.NaT if pd.api.types.is_datetime64_any_dtype(df[_c]) else float("nan") if pd.api.types.is_float_dtype(df[_c]) else None
    for i in range(1, len(df)):
        t0 = _as_local_naive_ts(df.loc[i - 1, ds_col])
        t1 = _as_local_naive_ts(df.loc[i, ds_col])
        if (t1 - t0).total_seconds() / 60 > threshold_min:
            parts.append(df.iloc[prev:i])
            parts.append(_gap_row)
            prev = i
    parts.append(df.iloc[prev:])
    return pd.concat(parts, ignore_index=True)


def _snapshot_age_minutes(snapshot_ts):
    ts_local = _to_local_timestamp(snapshot_ts)
    if ts_local is None:
        return None, None
    now_local = _now_local_ts()
    age_min = max(0.0, (now_local - ts_local).total_seconds() / 60.0)
    return ts_local, age_min


def _is_live_lane_snapshot(snapshot):
    if snapshot is None:
        return False
    if pd.isna(snapshot.get("active_lanes")):
        return False
    _, age_min = _snapshot_age_minutes(snapshot.get("timestamp"))
    return age_min is not None and age_min <= SNAPSHOT_MAX_AGE_MIN


WAIT_DISPLAY_CAP_MIN = 10

def _format_wait(value):
    return f"{float(value):.1f} min" if value is not None else "Forecast pending"

def _format_wait_capped(value):
    if value is None:
        return "Forecast pending"
    if float(value) >= WAIT_DISPLAY_CAP_MIN:
        return f"{WAIT_DISPLAY_CAP_MIN:.0f}+ min"
    return f"{float(value):.1f} min"

def _wait_card_value(value):
    """HTML value string for a predicted-wait metric card, capped at WAIT_DISPLAY_CAP_MIN."""
    if value is None:
        return "—"
    if value >= WAIT_DISPLAY_CAP_MIN:
        return f"{WAIT_DISPLAY_CAP_MIN:.0f}+<span class='metric-unit'> min</span>"
    return f"{value:.1f}<span class='metric-unit'> min</span>"


def _format_metric_value(value, suffix=""):
    if value is None:
        return "—"
    return f"{value}{suffix}"


def _lane_phrase(lanes):
    return f"{lanes} lane" if lanes == 1 else f"{lanes} lanes"


def _hour_phrase(hours):
    if hours < 24:
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    days = hours // 24
    return f"{days} day" if days == 1 else f"{days} days"


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


def _fmt_yhat(val):
    """Format a model yhat value, handling None, NaN, and pd.NA safely."""
    try:
        f = float(val)
        return f"{f:.1f}" if pd.notna(f) else "—"
    except (TypeError, ValueError):
        return "—"


def _lane_recommendation(lane_waits, selected_lanes, wait_10m):
    """Return a recommendation string when changing lanes would cross a threshold."""
    if not lane_waits or wait_10m is None:
        return None
    if wait_10m >= WAIT_BUSY_MIN and selected_lanes < MAX_LANES:
        next_lanes = selected_lanes + 1
        next_wait = lane_waits.get(next_lanes, {}).get("wait_10m")
        if next_wait is not None and next_wait < WAIT_BUSY_MIN:
            return f"Opening {_lane_phrase(next_lanes)} would bring +10 min wait to {next_wait:.1f} min (below busy threshold)"
        if next_wait is not None:
            return f"Opening {_lane_phrase(next_lanes)} would reduce +10 min wait to {next_wait:.1f} min"
    if wait_10m < WAIT_BUSY_MIN and selected_lanes > 1:
        fewer_lanes = selected_lanes - 1
        fewer_wait = lane_waits.get(fewer_lanes, {}).get("wait_10m")
        if fewer_wait is not None and fewer_wait < WAIT_BUSY_MIN:
            return f"Reducing to {_lane_phrase(fewer_lanes)} would still keep wait at {fewer_wait:.1f} min (below busy threshold)"
    return None


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


_ARRIVAL_WEIGHT_BY_COL = {
    "prophet_yhat": W_PROPHET,
    "lstm_yhat": W_LSTM,
    "xgb_yhat": W_XGB,
}


def _weighted_arrival_blend(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    valid = [c for c in cols if c in frame.columns]
    if not valid:
        return pd.Series(0.0, index=frame.index)
    weights = np.array([_ARRIVAL_WEIGHT_BY_COL.get(c, 1.0) for c in valid], dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(valid), dtype=float)
    values = pd.concat([pd.to_numeric(frame[c], errors="coerce") for c in valid], axis=1)
    weighted_values = values.mul(weights, axis=1)
    available_weight = values.notna().mul(weights, axis=1).sum(axis=1)
    blended = weighted_values.sum(axis=1, skipna=True)
    return blended.div(available_weight.where(available_weight > 0), fill_value=np.nan)


def _run_prediction_ui(data_span_days: int = 30):
    with st.spinner(f"Running prediction model on last {data_span_days} days... (1-3 min)"):
        ok, _out, err = _run_prediction(data_span_days=data_span_days)
    if ok:
        st.cache_data.clear()
        st.rerun()
    st.error("Prediction failed.")
    if err:
        with st.expander("Error details"):
            st.code(err[-3000:])


def _run_backtest(data_span_days: int = 30, overwrite: bool = False):
    _here  = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(_here, "backtest_predict.py")
    cwd    = os.path.dirname(os.path.dirname(_here))
    env    = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd    = [sys.executable, script, "--days", str(data_span_days)]
    if overwrite:
        cmd.append("--overwrite")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, cwd=cwd,
    )
    return result.returncode == 0, result.stdout, result.stderr


def _run_backtest_ui(data_span_days: int = 30):
    with st.spinner(f"Running backtest on last {data_span_days} days... (1-5 min)"):
        ok, out, err = _run_backtest(data_span_days=data_span_days)
    if ok:
        st.cache_data.clear()
        st.rerun()
    st.error("Backtest failed.")
    if err:
        with st.expander("Error details"):
            st.code(err[-3000:])


# ── Auto-prediction scheduler (background process) ────────────────────────────

_HERE_DIR   = os.path.dirname(os.path.abspath(__file__))
_SCHED_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(_HERE_DIR)), "run_scheduler.py"
)
_PID_FILE = os.path.join(_HERE_DIR, ".scheduler.pid")


def _scheduler_pid() -> int | None:
    """Return the running scheduler PID, or None if not running."""
    try:
        with open(_PID_FILE) as fh:
            pid = int(fh.read().strip())
        try:
            import psutil
            if not psutil.pid_exists(pid):
                raise ProcessLookupError
        except ImportError:
            # psutil not installed — fall back to os.kill(pid, 0)
            os.kill(pid, 0)
        return pid
    except Exception:
        pass
    try:
        os.remove(_PID_FILE)
    except FileNotFoundError:
        pass
    return None


def _start_scheduler(interval_min: int, data_span_days: int) -> int:
    """Launch run_scheduler.py in the background and save its PID."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [
            sys.executable, _SCHED_SCRIPT,
            "--interval", str(interval_min),
            "--days",     str(data_span_days),
        ],
        cwd=os.path.dirname(os.path.dirname(_HERE_DIR)),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    with open(_PID_FILE, "w") as fh:
        fh.write(str(proc.pid))
    return proc.pid


def _stop_scheduler() -> bool:
    """Kill the running scheduler process."""
    pid = _scheduler_pid()
    if pid is None:
        return False
    try:
        import psutil
        psutil.Process(pid).terminate()
    except Exception:
        pass
    try:
        os.remove(_PID_FILE)
    except FileNotFoundError:
        pass
    return True


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
            arrivals=("arrivals", "mean"),
            wait_min=("wait_min", "last"),
        )
        .rename(columns={"display_ds": "ds"})
        .dropna(subset=["arrivals", "wait_min"], how="all")
    )
    return display


@st.cache_data(ttl=REFRESH_SEC)
def load_service_time() -> float:
    """Return median pure service time from service_events (7 days).

    Priority:
    1. Rows where active_head_tracks_in_lane = 0 (person was alone → dwell ≈ service time).
    2. Fallback JOIN with entrance_events for historical rows lacking the column.
    3. All rows median if neither source has enough data.
    4. Config/env default.
    """
    try:
        with _conn() as conn:
            # Path 1: new column present and populated
            df = pd.read_sql(
                """
                SELECT total_dwell_sec / 60.0 AS service_min
                FROM service_events
                WHERE total_dwell_sec > 0
                  AND active_head_tracks_in_lane = 0
                  AND timestamp >= NOW() - INTERVAL '7 days'
                """,
                conn,
            )
            if not df.empty and len(df) >= 10:
                vals = pd.to_numeric(df["service_min"], errors="coerce").dropna()
                vals = vals[vals >= DWELL_MIN_FLOOR]
                if not vals.empty:
                    return float(min(max(float(vals.median()), DWELL_MIN_FLOOR), 10.0))

            # Path 2: historical rows — join with entrance_events
            df = pd.read_sql(
                """
                SELECT se.total_dwell_sec / 60.0 AS service_min
                FROM service_events se
                JOIN entrance_events ee ON ee.track_id = se.track_id
                WHERE se.total_dwell_sec > 0
                  AND ee.active_head_tracks_in_lane = 0
                  AND se.timestamp >= NOW() - INTERVAL '7 days'
                """,
                conn,
            )
            if not df.empty and len(df) >= 10:
                vals = pd.to_numeric(df["service_min"], errors="coerce").dropna()
                vals = vals[vals >= DWELL_MIN_FLOOR]
                if not vals.empty:
                    return float(min(max(float(vals.median()), DWELL_MIN_FLOOR), 10.0))

            # Path 3: not enough solo events — fall back to all events
            df = pd.read_sql(
                """
                SELECT total_dwell_sec / 60.0 AS service_min
                FROM service_events
                WHERE total_dwell_sec > 0
                  AND timestamp >= NOW() - INTERVAL '7 days'
                """,
                conn,
            )
            if not df.empty:
                vals = pd.to_numeric(df["service_min"], errors="coerce").dropna()
                vals = vals[vals >= DWELL_MIN_FLOOR]
                if not vals.empty:
                    return float(min(max(float(vals.median()), DWELL_MIN_FLOOR), 10.0))
    except Exception:
        pass
    return float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))


@st.cache_data(ttl=REFRESH_SEC)
def load_service_time_per_lane() -> dict:
    """Return per-lane median pure service time (minutes) keyed by lane_id.
    Uses solo events (active_head_tracks_in_lane = 0) to exclude wait time.
    Falls back to JOIN with entrance_events for historical rows, then all events."""
    try:
        with _conn() as conn:
            # Path 1: new column
            df = pd.read_sql(
                """
                SELECT lane_id, total_dwell_sec / 60.0 AS service_min
                FROM service_events
                WHERE total_dwell_sec > 0
                  AND lane_id IS NOT NULL
                  AND active_head_tracks_in_lane = 0
                  AND timestamp >= NOW() - INTERVAL '7 days'
                """,
                conn,
            )
            if df.empty or df.groupby("lane_id").size().min() < 5:
                # Path 2: historical JOIN
                df2 = pd.read_sql(
                    """
                    SELECT se.lane_id, se.total_dwell_sec / 60.0 AS service_min
                    FROM service_events se
                    JOIN entrance_events ee ON ee.track_id = se.track_id
                    WHERE se.total_dwell_sec > 0
                      AND se.lane_id IS NOT NULL
                      AND ee.active_head_tracks_in_lane = 0
                      AND se.timestamp >= NOW() - INTERVAL '7 days'
                    """,
                    conn,
                )
                df = pd.concat([df, df2], ignore_index=True).drop_duplicates()

            if not df.empty:
                result = {}
                for lane_id, group in df.groupby("lane_id"):
                    vals = pd.to_numeric(group["service_min"], errors="coerce").dropna()
                    vals = vals[vals >= DWELL_MIN_FLOOR]
                    if not vals.empty:
                        result[int(lane_id)] = float(min(max(float(vals.median()), DWELL_MIN_FLOOR), 10.0))
                return result
    except Exception:
        pass
    return {}


def _estimated_service_minutes(_snapshot):
    return load_service_time()


def _build_checkout_arrivals(forecast_rows, recent_entries):
    wait_frame = forecast_rows[["ds"]].copy()
    wait_frame["yhat"] = forecast_rows["arrivals"].fillna(0.0).astype(float).values
    return wait_frame


def _compute_waits_for_lanes(forecast_rows, recent_entries, current_queue, service_minutes, lane_count, dwell_series=None):
    wait_frame = _build_checkout_arrivals(forecast_rows, recent_entries)
    avg_dwell = dwell_series if dwell_series is not None else service_minutes
    wait_rows, wait_15m, wait_30m, _service_per_bucket = compute_wait_estimates(
        wait_frame,
        current_queue=current_queue,
        avg_dwell_min=avg_dwell,
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
                       prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, wait_45m, status,
                       wait_1lane_15m, wait_2lane_15m, wait_3lane_15m
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '24 hours'
                ORDER BY prediction_for, predicted_at DESC
                LIMIT 20
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.read_sql(
                """
                SELECT DISTINCT ON (prediction_for)
                       prediction_for AS ds,
                       prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat,
                       est_wait_minutes AS wait_min,
                       wait_15m, wait_30m, status
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '24 hours'
                ORDER BY prediction_for, predicted_at DESC
                LIMIT 20
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
            SELECT timestamp, queue_count, avg_dwell_sec,
                   COALESCE(active_lanes, 1) AS active_lanes
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
def load_demographics_today():
    """Load today's gender and age distribution from entrance_events."""
    with _conn() as conn:
        gender_df = pd.read_sql(
            """
            SELECT gender, COUNT(*) AS count
            FROM entrance_events
            WHERE camera_id = %s
              AND timestamp >= CURRENT_DATE
              AND gender IS NOT NULL
              AND gender != 'unknown'
            GROUP BY gender
            """,
            conn,
            params=(CAMERA_ID,),
        )
        age_df = pd.read_sql(
            """
            SELECT
                CASE
                    WHEN age_estimate < 30 THEN '18-30'
                    WHEN age_estimate < 50 THEN '30-50'
                    ELSE '50+'
                END AS age_group,
                COUNT(*) AS count
            FROM entrance_events
            WHERE camera_id = %s
              AND timestamp >= CURRENT_DATE
              AND age_estimate IS NOT NULL
            GROUP BY 1
            ORDER BY MIN(age_estimate)
            """,
            conn,
            params=(CAMERA_ID,),
        )
        hourly_df = pd.read_sql(
            """
            SELECT
                EXTRACT(HOUR FROM timestamp)::int   AS hour,
                COUNT(*)                             AS total,
                COUNT(*) FILTER (WHERE gender = 'female') AS female_count,
                COUNT(*) FILTER (WHERE gender = 'male')   AS male_count,
                ROUND(AVG(age_estimate)::numeric, 1)      AS avg_age
            FROM entrance_events
            WHERE camera_id = %s
              AND timestamp >= CURRENT_DATE
            GROUP BY 1
            ORDER BY 1
            """,
            conn,
            params=(CAMERA_ID,),
        )
    return gender_df, age_df, hourly_df


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
    _oh, _om, _ch, _cm, _open_tot, _close_tot = _today_hours()
    with _conn() as conn:
        df = pd.read_sql(
            f"""
            SELECT UPPER(status) AS status, COUNT(*) AS slots
            FROM queue_predictions
            WHERE prediction_for >= CURRENT_DATE
              AND prediction_for < NOW()
              AND EXTRACT(HOUR FROM prediction_for AT TIME ZONE current_setting('TIMEZONE')) * 60
                  + EXTRACT(MINUTE FROM prediction_for AT TIME ZONE current_setting('TIMEZONE')) >= {_open_tot}
              AND EXTRACT(HOUR FROM prediction_for AT TIME ZONE current_setting('TIMEZONE')) * 60
                  + EXTRACT(MINUTE FROM prediction_for AT TIME ZONE current_setting('TIMEZONE')) < {_close_tot}
            GROUP BY status
            """,
            conn,
        )
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_model_breakdown():
    """Per-model yhat for the most recent prediction run's upcoming slots (next display horizon)."""
    with _conn() as conn:
        try:
            df = pd.read_sql(
                f"""
                SELECT prediction_for AS ds,
                       prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat
                FROM queue_predictions
                WHERE prediction_for >= NOW()
                  AND prediction_for <= NOW() + INTERVAL '{FORECAST_DISPLAY_MIN} minutes'
                  AND predicted_at = (SELECT MAX(predicted_at) FROM queue_predictions)
                ORDER BY prediction_for
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.DataFrame()
    return df


@st.cache_data(ttl=300)
def load_training_stats():
    """Count and date span of clean real entrance_events used for model training."""
    with _conn() as conn:
        try:
            row = pd.read_sql(
                """
                SELECT
                    COUNT(*) AS row_count,
                    MIN(timestamp)::date AS first_day,
                    MAX(timestamp)::date AS last_day,
                    (MAX(timestamp) - MIN(timestamp)) AS span
                FROM entrance_events
                WHERE camera_id NOT LIKE 'SIM_%%'
                  AND dwell_seconds >= 10
                """,
                conn,
            )
            return row.iloc[0] if not row.empty else None
        except Exception:
            conn.rollback()
            return None


@st.cache_data(ttl=REFRESH_SEC)
def load_raw_arrivals(days: int):
    """Bucketed entrance arrivals for the last N days (training data view)."""
    with _conn() as conn:
        try:
            df = pd.read_sql(
                f"""
                SELECT
                    time_bucket('{BUCKET_MIN} minutes', timestamp) AS ds,
                    COUNT(*) AS arrivals
                FROM entrance_events
                WHERE camera_id NOT LIKE 'SIM_%%'
                  AND dwell_seconds >= 10
                  AND timestamp >= NOW() - INTERVAL '{days} days'
                GROUP BY ds
                ORDER BY ds
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.DataFrame(columns=["ds", "arrivals"])
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_service_history(days: int):
    """15-min median dwell time from service_events over the last N days."""
    with _conn() as conn:
        try:
            df = pd.read_sql(
                f"""
                SELECT
                    time_bucket('15 minutes', timestamp) AS ds,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_dwell_sec / 60.0) AS dwell_min,
                    COUNT(*) AS n_events
                FROM service_events
                WHERE total_dwell_sec >= {int(DWELL_MIN_FLOOR * 60)}
                  AND total_dwell_sec <= 1800
                  AND timestamp >= NOW() - INTERVAL '{days} days'
                GROUP BY ds
                ORDER BY ds
                """,
                conn,
            )
            return df
        except Exception:
            conn.rollback()
            return pd.DataFrame(columns=["ds", "dwell_min", "n_events"])


_LANE_INFER_WINDOW_MIN = 15  # shorter window reduces overcounting from accumulated events
_LANE_COUNT_WINDOW_MIN = 30  # longer window for lane-count stability (avoids flicker when a lane is quiet)


@st.cache_data(ttl=REFRESH_SEC)
def load_active_lane_count(window_min: int = _LANE_COUNT_WINDOW_MIN) -> "int | None":
    """Count distinct active lanes from service_events over the last window_min minutes.
    Uses a longer window than the throughput estimator so a temporarily quiet lane
    does not cause the displayed lane count to flicker down."""
    try:
        with _conn() as conn:
            df = pd.read_sql(
                f"""
                SELECT COUNT(DISTINCT lane_id) AS lane_count
                FROM service_events
                WHERE lane_id IS NOT NULL
                  AND total_dwell_sec > {DWELL_MIN_FLOOR * 60}
                  AND timestamp >= NOW() - INTERVAL '{int(window_min)} minutes'
                """,
                conn,
            )
        if not df.empty:
            val = int(df.iloc[0]["lane_count"])
            return min(val, 5) if val > 0 else None
    except Exception:
        pass
    return None


@st.cache_data(ttl=REFRESH_SEC)
def load_recent_service_events(window_min: int = _LANE_INFER_WINDOW_MIN):
    """Raw service_events rows from the last `window_min` minutes for lane inference."""
    with _conn() as conn:
        try:
            return pd.read_sql(
                f"""
                SELECT total_dwell_sec, lane_id
                FROM service_events
                WHERE timestamp >= NOW() - INTERVAL '{window_min} minutes'
                  AND total_dwell_sec > 0
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            return pd.DataFrame(columns=["total_dwell_sec"])


@st.cache_data(ttl=300)
def load_lane_history(days: int, svc_min: float, slot_minutes: int = 15) -> pd.DataFrame:
    """Historical median active lanes per slot_minutes time-of-day slot (three-source merge)."""
    import importlib
    import prediction.pipeline as _pipeline_mod
    importlib.reload(_pipeline_mod)
    _agg_fn = _pipeline_mod.load_lane_history_agg
    with _conn() as conn:
        try:
            _lh_oh, _lh_om, _lh_ch, _lh_cm, _, _ = _today_hours()
            return _agg_fn(
                conn,
                days=days,
                shop_open=_lh_oh,
                shop_close=_lh_ch,
                checkout_service_min=svc_min,
                service_min_dwell_sec=SERVICE_MIN_DWELL_SEC,
                slot_minutes=slot_minutes,
            )
        except Exception as _e:
            conn.rollback()
            st.warning(f"Lane history error: {_e}")
            return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SEC)
def load_historical_wait_predictions(hours: int):
    """Past est_wait_minutes from queue_predictions for the last N hours."""
    with _conn() as conn:
        try:
            df = pd.read_sql(
                f"""
                SELECT DISTINCT ON (prediction_for)
                       prediction_for AS ds,
                       est_wait_minutes AS wait_min
                FROM queue_predictions
                WHERE prediction_for < NOW()
                  AND prediction_for >= NOW() - INTERVAL '{int(hours)} hours'
                ORDER BY prediction_for, predicted_at DESC
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.DataFrame()
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_full_model_predictions(days: int = 30):
    """Per-model yhat for every prediction slot over the last N days + upcoming future slots.

    Uses DISTINCT ON (prediction_for) to pick the most recent prediction run per slot,
    so past slots show the latest saved forecast for that slot, not an in-sample
    model fit over the raw training data.
    """
    with _conn() as conn:
        try:
            df = pd.read_sql(
                f"""
                SELECT DISTINCT ON (prediction_for)
                       prediction_for AS ds,
                       predicted_at,
                       prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat
                FROM queue_predictions
                WHERE prediction_for >= NOW() - INTERVAL '{days} days'
                ORDER BY prediction_for, predicted_at DESC
                """,
                conn,
            )
        except Exception:
            conn.rollback()
            df = pd.DataFrame()
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_prophet_training_fit():
    fit_path = Path(__file__).resolve().parent / "models" / f"prophet_training_fit_{BUCKET_MIN}m.csv"
    if not fit_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fit_path)
    except Exception:
        return pd.DataFrame()


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="IQMS - Live Dashboard", page_icon="📊", layout="wide")

# Auto-refresh via a full browser reload (not a Streamlit fragment rerun —
# that path used to blank the page mid-render). A plain page reload always
# re-executes the script top to bottom, so it's safe against that failure mode.
# REFRESH_SEC == sched_interval minutes (set above), so the reload cadence
# always matches the prediction recalculation cadence.
st_html(
    f"<script>setTimeout(function(){{ window.parent.location.reload(); }}, {REFRESH_SEC * 1000});</script>",
    height=0,
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

_HISTORY_RANGE_OPTIONS = [1, 2, 4, 6, 12, 24, 48, 168]
if "history_range_hours" not in st.session_state:
    _qp_history_range = st.query_params.get("history_range_hours", None)
    if (
        _qp_history_range is not None
        and str(_qp_history_range).isdigit()
        and int(_qp_history_range) in _HISTORY_RANGE_OPTIONS
    ):
        st.session_state["history_range_hours"] = int(_qp_history_range)
    else:
        st.session_state["history_range_hours"] = 24

if "lane_infer_window_min" not in st.session_state:
    st.session_state["lane_infer_window_min"] = _LANE_INFER_WINDOW_MIN

_ARRIVAL_MODEL_OPTIONS = ["prophet", "lstm", "xgboost"]
_ARRIVAL_MODEL_DEFAULT = ["prophet", "lstm", "xgboost"]
_DWELL_MODEL_OPTIONS = ["safe", "legacy"]
_DWELL_FORECAST_MODEL_OPTIONS = ["xgboost", "lstm"]
_DWELL_FORECAST_MODEL_DEFAULT = ["xgboost", "lstm"]
if "arrival_models" not in st.session_state:
    _qp_am = st.query_params.get("arrival_models", "")
    _parsed_am = [m for m in _qp_am.split(",") if m in _ARRIVAL_MODEL_OPTIONS] if _qp_am else []
    st.session_state["arrival_models"] = _parsed_am if _parsed_am else list(_ARRIVAL_MODEL_DEFAULT)
else:
    # Sanitize: drop stale values no longer in options (e.g. old "ensemble")
    _sanitized = [m for m in st.session_state["arrival_models"] if m in _ARRIVAL_MODEL_OPTIONS]
    if not _sanitized:
        _sanitized = list(_ARRIVAL_MODEL_DEFAULT)
    st.session_state["arrival_models"] = _sanitized

if "pred_smooth_min" not in st.session_state:
    st.session_state["pred_smooth_min"] = PRED_SMOOTH_MIN

if "calibration_enabled" not in st.session_state:
    st.session_state["calibration_enabled"] = True

if "dwell_model_mode" not in st.session_state:
    _qp_dwell_mode = st.query_params.get("dwell_mode", DWELL_MODEL_MODE_DEFAULT)
    st.session_state["dwell_model_mode"] = (
        _qp_dwell_mode if _qp_dwell_mode in _DWELL_MODEL_OPTIONS else "safe"
    )

if "dwell_forecast_models" not in st.session_state:
    _qp_dwell_models = st.query_params.get("dwell_forecast_models", "")
    _parsed_dwell_models = [m for m in _qp_dwell_models.split(",") if m in _DWELL_FORECAST_MODEL_OPTIONS] if _qp_dwell_models else []
    st.session_state["dwell_forecast_models"] = _parsed_dwell_models if _parsed_dwell_models else list(_DWELL_FORECAST_MODEL_DEFAULT)
else:
    _sanitized_dwell_models = [m for m in st.session_state["dwell_forecast_models"] if m in _DWELL_FORECAST_MODEL_OPTIONS]
    if not _sanitized_dwell_models:
        _sanitized_dwell_models = list(_DWELL_FORECAST_MODEL_DEFAULT)
    st.session_state["dwell_forecast_models"] = _sanitized_dwell_models

if "show_live_wait_ref" not in st.session_state:
    st.session_state["show_live_wait_ref"] = False

if "training_span_days" not in st.session_state:
    _qp_days_init = st.query_params.get("training_days", None)
    _valid_days_init = [3, 7, 14, 21, 30, 60, 90]
    st.session_state["training_span_days"] = (
        int(_qp_days_init)
        if _qp_days_init is not None and str(_qp_days_init).isdigit() and int(_qp_days_init) in _valid_days_init
        else int(os.getenv("DATA_SPAN_DAYS", 7))
    )


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
    demo_gender, demo_age, demo_hourly = load_demographics_today()
    model_breakdown = load_model_breakdown()
    training_stats = load_training_stats()
    _span = int(st.session_state["training_span_days"])
    raw_arrivals = load_raw_arrivals(_span)
    service_history = load_service_history(_span)
    full_model_preds = load_full_model_predictions(_span)
    prophet_training_fit = load_prophet_training_fit()
    if not full_model_preds.empty and "ds" in full_model_preds.columns:
        _ds_local = _to_local_series(full_model_preds["ds"])
        _open_mask = _ds_local.apply(lambda t: is_open(pd.Timestamp(t)))
        full_model_preds = full_model_preds[_open_mask].reset_index(drop=True)
except Exception as exc:
    st.error(f"Database error: {exc}")
    st.stop()

# ── Inline dwell model training from service_history bucket medians ────────────
_DWELL_FEAT = ["hour", "minute_of_hour", "day_of_week", "is_weekend"]
_dwell_mode = st.session_state.get("dwell_model_mode", "safe")
_dwell_forecast_models = st.session_state.get("dwell_forecast_models", list(_DWELL_FORECAST_MODEL_DEFAULT))
_dwell_cache_key = f"dwell_models_{_span}_{_dwell_mode}_{','.join(sorted(_dwell_forecast_models))}"
_dwell_meta_cache_key = f"dwell_model_meta_{_span}_{_dwell_mode}_{','.join(sorted(_dwell_forecast_models))}"
if _dwell_cache_key not in st.session_state:
    _dwell_models = {}
    _dwell_meta = {}
    if "xgboost" in _dwell_forecast_models:
        _dwell_models["xgboost"], _dwell_meta["xgboost"] = _build_dashboard_dwell_model(service_history, _dwell_mode)
    if "lstm" in _dwell_forecast_models:
        _dwell_models["lstm"], _dwell_meta["lstm"] = _build_dashboard_dwell_lstm_model(service_history, _dwell_mode)
    st.session_state[_dwell_cache_key] = _dwell_models
    st.session_state[_dwell_meta_cache_key] = _dwell_meta
_dwell_models = st.session_state.get(_dwell_cache_key, {})
_dwell_model_meta = st.session_state.get(_dwell_meta_cache_key, {})

live_lane_source = _is_live_lane_snapshot(snap)
detected_lanes = int(snap["active_lanes"]) if live_lane_source else DEFAULT_LANES

# Infer active lanes from recent service_events (overrides snapshot if available).
# Lane *count* uses a 30-min window so a temporarily quiet lane doesn't cause flicker.
# Throughput fallback still uses the user-selected shorter window.
_lane_window = int(st.session_state.get("lane_infer_window_min", _LANE_INFER_WINDOW_MIN))
_svc_recent = load_recent_service_events(_lane_window)
_inferred_lanes: "int | None" = None
_inferred_lanes_source: str = ""
# Path 1: stable lane count from distinct lane_ids over 30 min
_stable_lane_count = load_active_lane_count(_LANE_COUNT_WINDOW_MIN)
if _stable_lane_count is not None:
    _inferred_lanes = _stable_lane_count
    _inferred_lanes_source = "lane_id"
else:
    # Path 2: throughput-based estimate (no lane_id data) over shorter window
    if not _svc_recent.empty:
        _svc_ok = _svc_recent[
            pd.to_numeric(_svc_recent["total_dwell_sec"], errors="coerce").fillna(0)
            > DWELL_MIN_FLOOR * 60
        ]
        if not _svc_ok.empty:
            _svc_min = load_service_time()
            _cap = _lane_window / max(_svc_min, 0.5)
            _inferred_lanes = int(min(max(round(len(_svc_ok) / max(_cap, 1)), 1), 5))
            _inferred_lanes_source = "throughput"

if _inferred_lanes is not None:
    detected_lanes = _inferred_lanes
queue_count = int(snap["queue_count"]) if snap is not None and pd.notna(snap["queue_count"]) else 0
snapshot_ts_local, snapshot_age_min = _snapshot_age_minutes(snap["timestamp"] if snap is not None else None)
service_minutes = _estimated_service_minutes(snap)
service_minutes_per_lane = load_service_time_per_lane()

# Apply fuzzy dwell modifier: adjusts service_minutes based on current
# queue demographics (age, equipment type, group size).
try:
    with _conn() as _dm_conn:
        _dwell_mod = compute_dwell_modifier_from_recent(_dm_conn)
except Exception:
    _dwell_mod = 1.0
service_minutes = service_minutes * _dwell_mod

if "forecast_active_lanes" not in st.session_state:
    _qp_lanes = st.query_params.get("lanes", None)
    if _qp_lanes is not None and str(_qp_lanes).isdigit() and int(_qp_lanes) in range(1, 6):
        st.session_state["forecast_active_lanes"] = int(_qp_lanes)
    else:
        try:
            with _conn() as _init_conn:
                with _init_conn.cursor() as _init_cur:
                    _init_cur.execute("SELECT open_lanes FROM dashboard_state WHERE id = 1")
                    _init_row = _init_cur.fetchone()
                    _db_lanes = int(_init_row[0]) if _init_row and _init_row[0] else None
            st.session_state["forecast_active_lanes"] = _db_lanes if _db_lanes else detected_lanes
        except Exception:
            st.session_state["forecast_active_lanes"] = detected_lanes
    st.session_state["_dashboard_last_written_lanes"] = st.session_state["forecast_active_lanes"]
else:
    try:
        with _conn() as _sync_conn:
            with _sync_conn.cursor() as _sync_cur:
                _sync_cur.execute("SELECT open_lanes FROM dashboard_state WHERE id = 1")
                _sync_row = _sync_cur.fetchone()
                _sync_db_lanes = int(_sync_row[0]) if _sync_row and _sync_row[0] else None
        if (_sync_db_lanes
                and _sync_db_lanes != st.session_state.get("_dashboard_last_written_lanes")
                and _sync_db_lanes != st.session_state.get("forecast_active_lanes")):
            st.session_state["forecast_active_lanes"] = _sync_db_lanes
    except Exception:
        pass
selected_lanes = int(st.session_state["forecast_active_lanes"])
selected_lanes = max(1, min(MAX_LANES, selected_lanes))

# Defined unconditionally — used later at module level regardless of whether
# pred_df has any rows. Only depend on queue_count/selected_lanes (both
# already set above), not on predictions being available, so there's no
# reason these should ever be missing. Recomputed below when pred_df isn't
# empty (same formula), this is just the fallback for when it is.
_in_service = min(queue_count, selected_lanes)
_waiting_backlog = max(0.0, float(queue_count - selected_lanes))

wait_0m  = None
wait_5m  = None
wait_10m = None
wait_15m = None
wait_20m = None
wait_30m = None
wait_45m = None
lane_waits = {}
forecast_waits = pd.DataFrame(columns=["ds", "wait_min"])
pred_future = pd.DataFrame(columns=["ds", "arrivals"])
_arrival_calib = _load_arrival_calibration_debug()
_arrival_calib_note = ""
_arrival_calib_debug_rows: list[dict[str, str]] = []
_arrival_cols_used: list[str] = []
_arrival_models_selected: list[str] = []
_arrival_calib_active = False

_arrivals_capped = False
_pred_mean = _pred_min = _pred_max = float("nan")
_dwell_per_slot = None
_dwell_per_slot_base = None
_dwell_pred_by_model: dict[str, list[float]] = {}
_model_col_map = {"ensemble": "ensemble_yhat", "prophet": "prophet_yhat",
                  "lstm": "lstm_yhat", "xgboost": "xgb_yhat"}
_selected_models = st.session_state.get("arrival_models") or ["prophet", "lstm", "xgboost"]
if not pred_df.empty:
    _model_col_map = {"ensemble": "ensemble_yhat", "prophet": "prophet_yhat",
                      "lstm": "lstm_yhat", "xgboost": "xgb_yhat"}
    _selected_models = st.session_state.get("arrival_models") or ["prophet", "lstm", "xgboost"]
    _arrival_models_selected = list(_selected_models)
    _valid_cols = [_model_col_map[m] for m in _selected_models
                   if _model_col_map.get(m) in pred_df.columns]
    if not _valid_cols:
        for _fb in ["prophet_yhat", "lstm_yhat", "xgb_yhat", "ensemble_yhat", "arrivals"]:
            if _fb in pred_df.columns:
                _valid_cols = [_fb]
                break
    _arrival_cols_used = list(_valid_cols)
    _arrival_calib_active = any(col == "ensemble_yhat" for col in _valid_cols)
    _pred_ds_local = _to_local_series(pred_df["ds"])
    _open_pred_mask = _pred_ds_local.apply(lambda t: is_open(pd.Timestamp(t)))
    pred_df = pred_df[_open_pred_mask].reset_index(drop=True)
    pred_future = pred_df[["ds"]].copy()
    pred_future["ds"] = _to_local_series(pred_future["ds"])
    if _valid_cols:
        pred_future["arrivals"] = _weighted_arrival_blend(pred_df, _valid_cols)
    else:
        pred_future["arrivals"] = 0.0

    # Apply calibration to non-ensemble blends. ensemble_yhat already has k baked in
    # from ensemble_predict.py; prophet/lstm/xgb raw columns do not.
    _cal_enabled = bool(st.session_state.get("calibration_enabled", True))
    _cal_global_pre  = float(_arrival_calib.get("k_global", 1.0) or 1.0)
    _cal_hourly_pre  = _arrival_calib.get("k_by_hour", {}) or {}
    if _cal_enabled and not _arrival_calib_active and _arrival_calib.get("loaded") and not pred_future.empty:
        _k_per_slot = [
            float(_cal_hourly_pre.get(str(int(pd.Timestamp(t).hour)), _cal_global_pre))
            for t in pred_future["ds"]
        ]
        pred_future["arrivals"] = (
            (pred_future["arrivals"] * pd.Series(_k_per_slot, index=pred_future.index))
            .clip(lower=0)
        )
        _arrival_calib_active = True

    _actual_rate = (entries_delta[0] if entries_delta else 0) / max(1.0, 60.0 / BUCKET_MIN)
    if _actual_rate > 0 and pred_future["arrivals"].mean() > _actual_rate * 5:
        pred_future["arrivals"] = pred_future["arrivals"].clip(upper=_actual_rate * 5)
        _arrivals_capped = True
    _smooth_win = max(1, round(int(st.session_state.get("pred_smooth_min", PRED_SMOOTH_MIN)) / BUCKET_MIN))
    pred_future["arrivals"] = (
        pred_future["arrivals"]
        .rolling(window=_smooth_win, center=True, min_periods=1)
        .mean()
    )
    # Compute stats AFTER capping and smoothing — must match the chart bars
    _display_cutoff = _now_local_ts() + pd.Timedelta(minutes=FORECAST_DISPLAY_MIN)
    _pred_display = pred_future[pred_future["ds"] <= _display_cutoff]["arrivals"]
    _pred_mean = _pred_display.mean(skipna=True) if not _pred_display.empty else pred_future["arrivals"].mean(skipna=True)
    _pred_min  = _pred_display.min(skipna=True) if not _pred_display.empty else pred_future["arrivals"].min(skipna=True)
    _pred_max  = _pred_display.max(skipna=True) if not _pred_display.empty else pred_future["arrivals"].max(skipna=True)
    _cal_hour = int(pd.Timestamp(pred_future.iloc[0]["ds"]).hour) if not pred_future.empty else int(_now_local_ts().hour)
    _cal_global = float(_arrival_calib.get("k_global", 1.0) or 1.0)
    _cal_hourly_map = _arrival_calib.get("k_by_hour", {}) or {}
    _cal_hour_k = float(_cal_hourly_map.get(_cal_hour, _cal_global))
    _cal_mode = (
        f"hour {str(_cal_hour).zfill(2)} override"
        if _cal_hour in _cal_hourly_map else
        "global only"
    )
    if _arrival_calib.get("error"):
        _arrival_calib_note = (
            f"Arrival calibration file could not be read ({_arrival_calib.get('error')}). "
            "Saved arrival forecasts are shown without dashboard-side calibration metadata."
        )
    elif _arrival_calib.get("loaded"):
        _cols_label = ', '.join(_arrival_cols_used) if _arrival_cols_used else 'raw model columns'
        _src_label  = "ensemble_yhat (baked in at save time)" if any("ensemble" in c for c in _arrival_cols_used) else f"{_cols_label} (applied in dashboard)"
        if _cal_enabled:
            _arrival_calib_note = (
                f"Arrival calibration active: <code>k={_cal_hour_k:.4f}</code> ({_cal_mode}, "
                f"k_global={_cal_global:.4f}) applied to <code>{_src_label}</code>."
            )
        else:
            _arrival_calib_note = (
                f"Arrival calibration is <b>OFF</b>. <code>calibration.json</code> is loaded "
                f"(k_global={_cal_global:.4f}), but arrivals are shown uncorrected."
            )
    else:
        _arrival_calib_note = (
            "No <code>calibration.json</code> file is loaded, so saved arrival forecasts are being used without "
            "any multiplicative arrival calibration."
        )
    _arrival_calib_debug_rows = [
        {"Field": "Calibration file", "Value": str(_arrival_calib.get("path", ARRIVAL_CALIB_PATH))},
        {"Field": "Loaded", "Value": "yes" if _arrival_calib.get("loaded") else "no"},
        {"Field": "Selected models", "Value": ", ".join(_arrival_models_selected) if _arrival_models_selected else "—"},
        {"Field": "Saved columns used", "Value": ", ".join(_arrival_cols_used) if _arrival_cols_used else "—"},
        {"Field": "Calibration enabled", "Value": "yes" if _cal_enabled else "no"},
        {"Field": "Calibration active here", "Value": "yes" if _arrival_calib_active else "no"},
        {"Field": "k_global", "Value": f"{_cal_global:.4f}"},
        {"Field": "Current forecast hour", "Value": f"{_cal_hour:02d}:00"},
        {"Field": "Current k", "Value": f"{_cal_hour_k:.4f} ({_cal_mode})"},
        {"Field": "Hourly overrides", "Value": str(len(_cal_hourly_map))},
    ]
    _in_service = min(queue_count, selected_lanes)
    _waiting_backlog = max(0.0, float(queue_count - selected_lanes))

    # Per-slot dwell from the selected dwell forecast models.
    _dwell_per_slot = None
    if not pred_future.empty:
        _pf_t = pred_future["ds"]
        _pf_feat = pd.DataFrame({
            "hour":           _pf_t.apply(lambda t: pd.Timestamp(t).hour),
            "minute_of_hour": _pf_t.apply(lambda t: pd.Timestamp(t).minute),
            "day_of_week":    _pf_t.apply(lambda t: pd.Timestamp(t).dayofweek),
            "is_weekend":     _pf_t.apply(lambda t: int(pd.Timestamp(t).dayofweek >= 5)),
        })
        if _dwell_models.get("xgboost") is not None:
            _dwell_pred_by_model["xgboost"] = (
                _dwell_models["xgboost"].predict(_pf_feat[_DWELL_FEAT]).clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP).tolist()
            )
        if _dwell_models.get("lstm") is not None:
            _dwell_pred_by_model["lstm"] = _predict_dashboard_dwell_lstm(_dwell_models["lstm"], list(_pf_t))
        _active_dwell_pred = [vals for vals in _dwell_pred_by_model.values() if vals]
        if _active_dwell_pred:
            _dwell_per_slot = [
                float(np.mean([vals[i] for vals in _active_dwell_pred if i < len(vals)]))
                for i in range(max(len(vals) for vals in _active_dwell_pred))
            ]

    _dwell_per_slot_base = list(_dwell_per_slot) if _dwell_per_slot else None
    _base_dwell = (_dwell_per_slot_base[0] if _dwell_per_slot_base else service_minutes)

    def _apply_residual_correction(dwell_slots, lane_count):
        """Inflate slot-0 dwell when all lanes are busy to reflect reduced first-bucket capacity."""
        in_svc = min(queue_count, lane_count)
        if in_svc < lane_count or BUCKET_MIN <= _base_dwell / 2:
            return dwell_slots
        residual = _base_dwell / 2
        first_dwell = BUCKET_MIN * _base_dwell / (BUCKET_MIN - residual)
        if dwell_slots:
            return [first_dwell] + list(dwell_slots[1:])
        return [first_dwell] + [service_minutes] * max(0, len(pred_future) - 1)

    _dwell_per_slot = _apply_residual_correction(_dwell_per_slot_base, selected_lanes)

    forecast_waits, wait_15m, wait_30m, wait_45m = _compute_waits_for_lanes(
        pred_future.copy(),
        recent_entry_hist,
        _waiting_backlog,
        service_minutes,
        selected_lanes,
        dwell_series=_dwell_per_slot,
    )
    def _wait_at(df, minutes):
        idx = min(round(minutes / BUCKET_MIN), len(df) - 1) if not df.empty else -1
        return float(df.iloc[idx]["wait_min"]) if idx >= 0 else None
    wait_0m  = _wait_at(forecast_waits, 0)
    wait_5m  = _wait_at(forecast_waits, 5)
    wait_10m = _wait_at(forecast_waits, 10)
    wait_15m = _wait_at(forecast_waits, 15)
    for lane_count in range(1, MAX_LANES + 1):
        _lane_dwell = _apply_residual_correction(_dwell_per_slot_base, lane_count)
        _wait_df, lane_wait_15, _wait_30, _wait_45 = _compute_waits_for_lanes(
            pred_future.copy(),
            recent_entry_hist,
            _waiting_backlog,
            service_minutes,
            lane_count,
            dwell_series=_lane_dwell,
        )
        lane_waits[lane_count] = {
            "wait_10m": _wait_at(_wait_df, 10),
            "wait_15m": lane_wait_15,
            "wait_30m": _wait_30,
            "wait_45m": _wait_45,
        }

# ── Write computed state for the mobile app API to consume ───────────────────
# The API reads this table directly so the app always shows the exact same
# values as the dashboard without needing to replicate the simulation logic.
try:
    import json as _json
    with _conn() as _sc:
        _scc = _sc.cursor()
        _scc.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_state (
                id                  INTEGER PRIMARY KEY DEFAULT 1,
                updated_at          TIMESTAMPTZ DEFAULT NOW(),
                queue_now           INTEGER,
                service_min         FLOAT,
                wait_0m             FLOAT,
                wait_5m             FLOAT,
                wait_10m            FLOAT,
                wait_15m            FLOAT,
                lane1_wait_15m      FLOAT,
                lane2_wait_15m      FLOAT,
                lane3_wait_15m      FLOAT,
                lane4_wait_15m      FLOAT,
                open_lanes          INTEGER,
                demographics_json   TEXT,
                entries_hour_json   TEXT
            )
        """)
        # Add columns if upgrading from old schema
        for _col, _type in [("demographics_json", "TEXT"), ("entries_hour_json", "TEXT")]:
            _scc.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_name='dashboard_state' AND column_name='{_col}')
                    THEN ALTER TABLE dashboard_state ADD COLUMN {_col} {_type}; END IF;
                END $$;
            """)

        def _rv(v): return round(float(v), 2) if v is not None and v == v else None

        # Build demographics JSON exactly as dashboard computes it
        _gender_total = int(demo_gender["count"].sum()) if not demo_gender.empty else 0
        _gender_data = []
        _gender_colors = {"male": "#2563eb", "female": "#db2777"}
        for _, _gr in demo_gender.iterrows():
            _gk = str(_gr["gender"]).lower()
            _gc = int(_gr["count"])
            _gender_data.append({
                "key": _gk, "label": _gk.capitalize(),
                "count": _gc,
                "percent": round(_gc / _gender_total * 100) if _gender_total else 0,
                "color": _gender_colors.get(_gk, "#94a3b8"),
            })

        _age_total = int(demo_age["count"].sum()) if not demo_age.empty else 0
        _age_colors = {"18-30": "#f97316", "30-50": "#3fb950", "50+": "#58a6ff"}
        _age_data = []
        for _, _ar in demo_age.iterrows():
            _ag = str(_ar["age_group"])
            _ac = int(_ar["count"])
            _age_data.append({
                "group": _ag, "count": _ac,
                "percent": round(_ac / _age_total * 100) if _age_total else 0,
                "color": _age_colors.get(_ag, "#8b949e"),
            })

        _demo_json = _json.dumps({"gender": _gender_data, "age": _age_data})

        # Build hourly entries JSON
        _hourly = []
        if not traffic_today.empty:
            _peak_cnt = int(traffic_today["entries"].max())
            for _, _hr in traffic_today.iterrows():
                _hc = int(_hr["entries"])
                _hourly.append({
                    "hour": _to_local_timestamp(_hr["hour"]).strftime("%H:00"),
                    "count": _hc,
                    "is_peak": _hc == _peak_cnt and _peak_cnt > 0,
                })
        _hourly_json = _json.dumps(_hourly)

        _scc.execute("""
            INSERT INTO dashboard_state
                (id, updated_at, queue_now, service_min,
                 wait_0m, wait_5m, wait_10m, wait_15m,
                 lane1_wait_15m, lane2_wait_15m, lane3_wait_15m, lane4_wait_15m,
                 open_lanes, demographics_json, entries_hour_json)
            VALUES (1, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                updated_at          = NOW(),
                queue_now           = EXCLUDED.queue_now,
                service_min         = EXCLUDED.service_min,
                wait_0m             = EXCLUDED.wait_0m,
                wait_5m             = EXCLUDED.wait_5m,
                wait_10m            = EXCLUDED.wait_10m,
                wait_15m            = EXCLUDED.wait_15m,
                lane1_wait_15m      = EXCLUDED.lane1_wait_15m,
                lane2_wait_15m      = EXCLUDED.lane2_wait_15m,
                lane3_wait_15m      = EXCLUDED.lane3_wait_15m,
                lane4_wait_15m      = EXCLUDED.lane4_wait_15m,
                open_lanes          = EXCLUDED.open_lanes,
                demographics_json   = EXCLUDED.demographics_json,
                entries_hour_json   = EXCLUDED.entries_hour_json
        """, (
            int(queue_count),
            _rv(service_minutes),
            _rv(wait_0m), _rv(wait_5m), _rv(wait_10m), _rv(wait_15m),
            _rv(lane_waits.get(1, {}).get("wait_15m")),
            _rv(lane_waits.get(2, {}).get("wait_15m")),
            _rv(lane_waits.get(3, {}).get("wait_15m")),
            _rv(lane_waits.get(4, {}).get("wait_15m")),
            int(selected_lanes),
            _demo_json,
            _hourly_json,
        ))
        _sc.commit()
        _scc.close()
        st.session_state["_dashboard_last_written_lanes"] = int(selected_lanes)
except Exception:
    pass  # never crash the dashboard because of a state write failure

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


status_class, status_label = _status_meta(wait_15m)
if _inferred_lanes is not None:
    lane_source_label = "Inferred lanes"
elif live_lane_source:
    lane_source_label = "Live lanes"
else:
    lane_source_label = "Default lanes"

if _inferred_lanes is not None:
    if _inferred_lanes_source == "lane_id":
        lane_source_note = (
            f"Detected queue state is {_lane_phrase(detected_lanes)} open "
            f"({detected_lanes} distinct lane{'s' if detected_lanes != 1 else ''} with service completions in the last {_LANE_COUNT_WINDOW_MIN} min)."
        )
    else:
        lane_source_note = (
            f"Detected queue state is {_lane_phrase(detected_lanes)} open "
            f"(estimated from service throughput in the last {_lane_window} min — no per-lane data yet)."
        )
elif live_lane_source:
    lane_source_note = f"Detected queue state is {_lane_phrase(detected_lanes)} open."
else:
    lane_source_note = (
        f"No recent lane snapshot in the last {SNAPSHOT_MAX_AGE_MIN} min. "
        f"Predictions fall back to {_lane_phrase(DEFAULT_LANES)} open."
    )
lane_parameter_note = (
    f"Forecast parameter is set to {_lane_phrase(selected_lanes)}."
    if selected_lanes == detected_lanes
    else f"Forecast parameter overrides the detected state: {_lane_phrase(selected_lanes)} selected."
)
status_text = (
    f"{status_label} - Predicted wait with {_lane_phrase(selected_lanes)}: {_format_wait_capped(wait_10m)}"
    if wait_10m is not None
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

_th_oh, _th_om, _th_ch, _th_cm, _, _ = _today_hours()
_hours_pill = f"Hours today: {_th_oh:02d}:{_th_om:02d}–{_th_ch:02d}:{_th_cm:02d}"
_day_name = pd.Timestamp.now().strftime("%a")
_default_open  = _day_hours(0)[:2]   # Mon as baseline default
_default_close = _day_hours(0)[2:]
if (_th_ch, _th_cm) != _default_close or (_th_oh, _th_om) != _default_open:
    _hours_pill += f" ({_day_name} override)"

context_bits = [
    f"{lane_source_label}: {_lane_phrase(detected_lanes)}",
    f"Forecast parameter: {_lane_phrase(selected_lanes)}",
    f"Wait +30 min: {_format_wait(wait_30m)}",
    _hours_pill,
]
if snapshot_ts_local is not None:
    snapshot_note = f"Latest snapshot: {snapshot_ts_local.strftime('%H:%M:%S')}"
    if snapshot_age_min is not None:
        snapshot_note += f" ({snapshot_age_min:.0f} min ago)"
    context_bits.append(snapshot_note)
if training_stats is not None:
    try:
        _rows = int(training_stats["row_count"])
        _span_days = round(training_stats["span"].total_seconds() / 86400) if training_stats["span"] is not None else None
        _span_str = f"{_span_days}d span" if _span_days is not None else ""
        context_bits.append(f"Training data: {_rows:,} rows{(' · ' + _span_str) if _span_str else ''}")
    except Exception:
        pass

st.markdown(
    '<div class="context-row">' +
    "".join(f'<span class="context-pill">{bit}</span>' for bit in context_bits) +
    "</div>",
    unsafe_allow_html=True,
)

_lane_rec = _lane_recommendation(lane_waits, selected_lanes, wait_10m)
if _lane_rec:
    _rec_color = "#ea580c" if wait_15m is not None and wait_15m >= WAIT_BUSY_MIN else "#16a34a"
    st.markdown(
        f'<div class="detail-note" style="border-left: 4px solid {_rec_color}; padding-left: 14px;">'
        f'<strong>Lane suggestion:</strong> {_lane_rec}</div>',
        unsafe_allow_html=True,
    )

if _arrivals_capped:
    st.warning(
        "Arrival forecast capped: model predicted arrivals were more than 5x the actual recent entry rate. "
        "Wait times are based on a capped estimate. This may indicate a training data issue — run prediction after cleaning historical data."
    )



def _sync_days_to_qp():
    st.query_params["training_days"] = str(st.session_state.get("training_span_days", int(os.getenv("DATA_SPAN_DAYS", 7))))

def _sync_lanes_to_qp():
    st.query_params["lanes"] = str(st.session_state.get("forecast_active_lanes", detected_lanes))

def _sync_history_range_to_qp():
    st.query_params["history_range_hours"] = str(st.session_state.get("history_range_hours", 24))

def _guard_arrival_models():
    current = st.session_state.get("arrival_models") or []
    if not current:
        current = list(_ARRIVAL_MODEL_DEFAULT)
        st.session_state["arrival_models"] = current
    st.query_params["arrival_models"] = ",".join(current)

def _sync_dwell_mode_to_qp():
    st.query_params["dwell_mode"] = str(st.session_state.get("dwell_model_mode", "safe"))

def _guard_dwell_forecast_models():
    current = st.session_state.get("dwell_forecast_models") or []
    if not current:
        current = list(_DWELL_FORECAST_MODEL_DEFAULT)
        st.session_state["dwell_forecast_models"] = current
    st.query_params["dwell_forecast_models"] = ",".join(current)

action_refresh, action_predict, action_backtest, action_param, action_days, action_model, action_smooth, action_dwell, action_dwell_models = st.columns([1, 1.2, 1.2, 1.7, 1.7, 1.8, 1.4, 1.2, 2.0])
with action_refresh:
    if st.button("Refresh", width='stretch'):
        st.cache_data.clear()
        st.rerun()
with action_predict:
    if st.button("Run Prediction", width='stretch', type="primary"):
        _run_prediction_ui(data_span_days=int(st.session_state["training_span_days"]))
with action_backtest:
    if st.button("Fill History", width='stretch', help="Run backtest to fill past prediction gaps using trained models"):
        _run_backtest_ui(data_span_days=int(st.session_state["training_span_days"]))
with action_param:
    st.select_slider(
        "Forecast active lanes",
        options=list(range(1, MAX_LANES + 1)),
        key="forecast_active_lanes",
        on_change=_sync_lanes_to_qp,
    )
with action_days:
    st.select_slider(
        "Training data span",
        options=[3, 7, 14, 21, 30, 60, 90],
        format_func=lambda v: f"{v} days",
        key="training_span_days",
        on_change=_sync_days_to_qp,
    )
with action_model:
    st.multiselect(
        "Arrivals model(s)",
        options=["prophet", "lstm", "xgboost"],
        format_func=lambda v: {"prophet": "Prophet", "lstm": "LSTM", "xgboost": "XGBoost"}[v],
        key="arrival_models",
        on_change=_guard_arrival_models,
    )
    st.toggle(
        "Apply arrival calibration",
        key="calibration_enabled",
        value=True,
        help="Multiply arrivals by the k learned from accuracy_eval.py (calibration.json).",
    )
with action_smooth:
    st.select_slider(
        "Prediction smoothing",
        options=[3, 5, 15, 30],
        format_func=lambda v: f"{v} min",
        key="pred_smooth_min",
    )
with action_dwell:
    st.selectbox(
        "Dwell model",
        options=_DWELL_MODEL_OPTIONS,
        format_func=lambda v: {"safe": "Safe", "legacy": "Legacy"}[v],
        key="dwell_model_mode",
        on_change=_sync_dwell_mode_to_qp,
    )
with action_dwell_models:
    st.multiselect(
        "Dwell forecast",
        options=_DWELL_FORECAST_MODEL_OPTIONS,
        format_func=lambda v: {"xgboost": "XGBoost", "lstm": "LSTM"}[v],
        key="dwell_forecast_models",
        on_change=_guard_dwell_forecast_models,
    )

# ── Auto-prediction scheduler UI ───────────────────────────────────────────────
# (sched_interval is initialized earlier, near st.set_page_config, since the
# page auto-refresh timer needs it before this point in the script.)

_sched_pid = _scheduler_pid()
_sched_running = _sched_pid is not None

_col_status, _col_interval, _col_start, _col_stop, _col_hint = st.columns([2, 1.5, 1, 1, 3])

with _col_status:
    if _sched_running:
        st.markdown(
            f'<div class="detail-note" style="border-left:4px solid #16a34a; padding-left:12px;">'
            f'<strong style="color:#16a34a">● Auto-predict running</strong>'
            f'<span style="color:#64748b; font-size:0.8rem;"> — every {st.session_state["sched_interval"]} min · PID {_sched_pid}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="detail-note" style="border-left:4px solid #94a3b8; padding-left:12px;">'
            '<strong style="color:#94a3b8">○ Auto-predict stopped</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

def _sync_interval_to_qp():
    st.query_params["sched_interval"] = str(st.session_state.get("sched_interval", 15))

with _col_interval:
    st.select_slider(
        "Interval (min)",
        options=[5, 10, 15, 30, 60],
        key="sched_interval",
        disabled=_sched_running,
        on_change=_sync_interval_to_qp,
    )

with _col_start:
    if st.button(
        "▶ Start",
        width='stretch',
        disabled=_sched_running,
        type="primary" if not _sched_running else "secondary",
    ):
        _pid = _start_scheduler(
            interval_min=int(st.session_state["sched_interval"]),
            data_span_days=int(st.session_state["training_span_days"]),
        )
        st.success(f"Scheduler started (PID {_pid})")
        st.rerun()

with _col_stop:
    if st.button(
        "■ Stop",
        width='stretch',
        disabled=not _sched_running,
    ):
        _stop_scheduler()
        st.info("Scheduler stopped.")
        st.rerun()

with _col_hint:
    st.markdown(
        '<div class="detail-note">Auto-predict runs <strong>Run Prediction</strong> automatically '
        'in the background. Interval and training span are locked while running.</div>',
        unsafe_allow_html=True,
    )


# ── Live operations ────────────────────────────────────────────────────────────

_section_title("Live Operations", "current queue, lane-aware waits, near-term trend")

c1, c2, c3, c4, c6 = st.columns(5)

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
            "Predicted Wait (now)",
            _wait_card_value(wait_0m),
            tone=_wait_tone(wait_0m),
            supporting_text=f"Next slot · {_lane_phrase(selected_lanes)}.",
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        _metric_card_html(
            "Predicted Wait (+5 min)",
            _wait_card_value(wait_5m),
            tone=_wait_tone(wait_5m),
            supporting_text=f"~{round(5 / BUCKET_MIN) * BUCKET_MIN} min ahead · {_lane_phrase(selected_lanes)}.",
        ),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        _metric_card_html(
            "Predicted Wait (+10 min)",
            _wait_card_value(wait_10m),
            tone=_wait_tone(wait_10m),
            supporting_text=f"~{round(10 / BUCKET_MIN) * BUCKET_MIN} min ahead · {_lane_phrase(selected_lanes)}.",
        ),
        unsafe_allow_html=True,
    )

with c6:
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

with st.expander("How was this predicted? (model inputs & simulation trace)"):
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        _exp_selected = st.session_state.get("arrival_models") or list(_ARRIVAL_MODEL_DEFAULT)
        _exp_col_map = {"prophet": ("prophet_yhat", "Prophet"),
                        "lstm":    ("lstm_yhat",    "LSTM"),
                        "xgboost": ("xgb_yhat",     "XGBoost")}
        _exp_weight = f"{1/len(_exp_selected):.0%}" if _exp_selected else "—"
        _exp_labels = {"prophet": "Prophet", "lstm": "LSTM", "xgboost": "XGBoost"}
        st.markdown(
            f"**Model ensemble — next slot arrivals** "
            f"({' + '.join(_exp_labels.get(m, m) for m in _exp_selected)}, "
            f"equal weight {_exp_weight} each)"
        )
        if not model_breakdown.empty:
            mb = model_breakdown.copy()
            mb["ds"] = _to_local_series(mb["ds"])
            rows_model = []
            for _, mrow in mb.iterrows():
                _dyn_vals = []
                for m in _exp_selected:
                    col, label = _exp_col_map.get(m, (None, None))
                    if col and col in mrow.index:
                        v = pd.to_numeric(mrow[col], errors="coerce")
                        rows_model.append({
                            "Model": label,
                            "Weight": _exp_weight,
                            "Predicted arrivals": _fmt_yhat(v),
                            "Time slot": mrow["ds"].strftime("%H:%M"),
                        })
                        if not pd.isna(v):
                            _dyn_vals.append(float(v))
                # Dynamic average row (only if >1 model selected)
                if len(_exp_selected) > 1:
                    _dyn_avg = sum(_dyn_vals) / len(_dyn_vals) if _dyn_vals else float("nan")
                    rows_model.append({
                        "Model": "Effective average",
                        "Weight": "100%",
                        "Predicted arrivals": _fmt_yhat(_dyn_avg),
                        "Time slot": mrow["ds"].strftime("%H:%M"),
                    })
                break  # first slot only
            st.dataframe(pd.DataFrame(rows_model), width='stretch', hide_index=True)
        else:
            st.info("Run prediction to populate model breakdown.")

        st.markdown("**Simulation inputs**")
        _svc_source = "service_events median (7d)" if service_minutes != DEFAULT_DWELL_MIN else "config fallback"
        _has_per_lane = bool(service_minutes_per_lane)
        _svc_source_full = f"{_svc_source} — per-lane breakdown available" if _has_per_lane else _svc_source
        _mod_label = f"{_dwell_mod:.2f}×  (age + equipment + group)" if abs(_dwell_mod - 1.0) > 0.01 else "1.00×  (no adjustment — no recent data)"
        _per_lane_str = "  |  ".join(
            f"Lane {lid + 1}: {svc:.1f} min"
            for lid, svc in sorted(service_minutes_per_lane.items())
        ) if _has_per_lane else "no per-lane data yet"
        inputs_df = pd.DataFrame([
            {"Input": "Current queue (backlog)", "Value": str(_waiting_backlog if not pred_df.empty else queue_count)},
            {"Input": "Active lanes", "Value": str(selected_lanes)},
            {"Input": "Avg service time (global)", "Value": f"{service_minutes:.1f} min/customer  ({_svc_source_full})"},
            {"Input": "Service time per lane", "Value": _per_lane_str},
            {"Input": "Demographic modifier", "Value": _mod_label},
            {"Input": "Service capacity", "Value": f"{selected_lanes * BUCKET_MIN / service_minutes:.1f} customers per {BUCKET_MIN}-min bucket"},
            {"Input": "Browsing gap", "Value": f"{BROWSING_GAP_MIN} min  (entrance → checkout shift)"},
            {"Input": "Bucket size", "Value": f"{BUCKET_MIN} min"},
        ])
        st.dataframe(inputs_df, width='stretch', hide_index=True)

    with exp_c2:
        st.markdown("**Queue simulation — first 5 buckets (→ +15 min)**")
        if not forecast_waits.empty and not pred_future.empty:
            _sim_rows = pred_future.merge(forecast_waits, on="ds", how="left")
            _sim_rows = _sim_rows.head(5).copy()
            _sim_rows["ds"] = _to_local_series(_sim_rows["ds"])
            _svc_cap = selected_lanes * BUCKET_MIN / max(service_minutes, 0.5)
            _running_q = float(_waiting_backlog if not pred_df.empty else queue_count)
            trace_rows = []
            for _, sr in _sim_rows.iterrows():
                _arr = max(0.0, float(sr["arrivals"]) if not pd.isna(sr["arrivals"]) else 0.0)
                _running_q = max(0.0, _running_q + _arr - _svc_cap)
                trace_rows.append({
                    "Time": sr["ds"].strftime("%H:%M"),
                    "Arrivals": f"{_arr:.1f}",
                    "Queue after": f"{_running_q:.1f}",
                    "Wait (min)": f"{float(sr['wait_min']):.2f}" if not pd.isna(sr.get("wait_min", float("nan"))) else "—",
                })
            st.dataframe(pd.DataFrame(trace_rows), width='stretch', hide_index=True)
            st.caption(
                f"Queue after = max(0, prev_queue + arrivals − {_svc_cap:.1f} served).  "
                f"Wait = queue ÷ {_svc_cap:.1f} × {BUCKET_MIN} min.  "
                f"Arrivals shown are raw entrance forecast (browsing-gap shift applied by prediction run)."
            )
        else:
            st.info("No forecast data available. Run prediction first.")

        if training_stats is not None:
            st.markdown("**Training data**")
            try:
                _rows = int(training_stats["row_count"])
                _first = str(training_stats["first_day"])
                _last = str(training_stats["last_day"])
                _span_days = round(training_stats["span"].total_seconds() / 86400) if training_stats["span"] is not None else "?"
                train_df = pd.DataFrame([
                    {"Metric": "Clean entrance events", "Value": f"{_rows:,}"},
                    {"Metric": "First record", "Value": _first},
                    {"Metric": "Last record", "Value": _last},
                    {"Metric": "Data span", "Value": f"{_span_days} days"},
                    {"Metric": "Bucket size", "Value": f"{BUCKET_MIN} min"},
                    {"Metric": "Training buckets (est.)", "Value": f"~{_rows // max(1, _span_days * (12 * 60 // BUCKET_MIN // 12))} / day"},
                ])
                st.dataframe(train_df, width='stretch', hide_index=True)
            except Exception:
                st.info("Training stats unavailable.")


# ── Global chart display range ────────────────────────────────────────────────

_display_range_col, _display_range_note_col = st.columns([2, 5])
with _display_range_col:
    st.select_slider(
        "Chart display range",
        options=_HISTORY_RANGE_OPTIONS,
        format_func=_hour_phrase,
        key="history_range_hours",
        on_change=_sync_history_range_to_qp,
    )
with _display_range_note_col:
    st.markdown(
        f'<div class="detail-note">Controls the history window shown on all time-series charts below. '
        f'Training data span ({int(st.session_state.get("training_span_days", 7))} days) is set above and is independent.</div>',
        unsafe_allow_html=True,
    )
history_range_hours = int(st.session_state["history_range_hours"])
_chart_x_start = _now_local_ts() - pd.Timedelta(hours=history_range_hours)
_chart_x_end   = _now_local_ts() + pd.Timedelta(hours=2)

# ── Model comparison chart ─────────────────────────────────────────────────────

_training_days_val = int(st.session_state.get("training_span_days", 30))
_cmp_active_models = st.session_state.get("arrival_models", ["prophet", "lstm", "xgboost"]) or ["prophet", "lstm", "xgboost"]
_cmp_model_labels  = {"ensemble": "Ensemble", "prophet": "Prophet", "lstm": "LSTM", "xgboost": "XGBoost"}
_cmp_label_str     = " / ".join(_cmp_model_labels.get(m, m) for m in _cmp_active_models)
_section_title(
    "Raw Data vs Model Predictions",
    f"showing last {_hour_phrase(history_range_hours)} · trained on {_training_days_val} days · {_cmp_label_str}",
)
_latest_predicted_at = None
if not full_model_preds.empty and "predicted_at" in full_model_preds.columns:
    try:
        _latest_predicted_at = _to_local_timestamp(full_model_preds["predicted_at"].max())
    except Exception:
        _latest_predicted_at = None
_pred_source_note = (
    f"Latest saved prediction run: {_latest_predicted_at.strftime('%d/%m %H:%M:%S')}. "
    if _latest_predicted_at is not None else ""
)
st.markdown(
    f'<div class="detail-note">{_pred_source_note}'
    f'Prophet display shift: {PROPHET_DISPLAY_SHIFT_MIN} min. '
    'Model lines in this chart come from saved <code>queue_predictions</code> rows across the visible time window. '
    'The green combination line is blended from those same visible curves.</div>',
    unsafe_allow_html=True,
)

_COLORS = {
    "raw":      "#94a3b8",
    "prophet":  "#2563eb",
    "lstm":     "#7c3aed",
    "xgboost":  "#ea580c",
    "ensemble": "#16a34a",
}

if not raw_arrivals.empty or not full_model_preds.empty or not prophet_training_fit.empty:
    fig_cmp = go.Figure()

    # ── Historical raw arrivals (resampled to 15-min median) ─────────────────
    if not raw_arrivals.empty:
        ra = raw_arrivals.copy()
        ra["ds"] = _to_local_series(ra["ds"])
        ra["arrivals"] = pd.to_numeric(ra["arrivals"], errors="coerce")
        ra = (
            ra.set_index("ds")
            .resample("15min")
            .median()
            .fillna(0.0)
            .reset_index()
        )
        ra["arrivals"] = ra["arrivals"].where(
            ra["ds"].apply(lambda t: is_open(pd.Timestamp(t))),
            None,
        )
        ra = _insert_overnight_gaps(ra)
        fig_cmp.add_trace(go.Scatter(
            x=_plotly_local_iso_series(ra["ds"]),
            y=ra["arrivals"],
            mode="lines",
            name="Raw data (15 min median)",
            line=dict(color=_COLORS["raw"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.18)",
            hovertemplate="%{x|%d/%m %H:%M}<br>Raw arrivals (15 min): %{y:.1f}<extra></extra>",
        ))

    # ── Per-model saved forecasts (LSTM/XGBoost + fallback Prophet) ──────────
    if not full_model_preds.empty:
        fmp = full_model_preds.copy()
        fmp["ds"] = _to_local_series(fmp["ds"])
        _prophet_shift_df = fmp[["ds", "prophet_yhat"]].copy()
        _prophet_shift_df["ds"] = _prophet_shift_df["ds"] + pd.Timedelta(minutes=PROPHET_DISPLAY_SHIFT_MIN)
        _prophet_shift_df["_plot_prophet_yhat"] = pd.to_numeric(_prophet_shift_df["prophet_yhat"], errors="coerce")
        _prophet_shift_df = _prophet_shift_df[["ds", "_plot_prophet_yhat"]]
        fmp = fmp[(fmp["ds"] >= _chart_x_start) & (fmp["ds"] <= _chart_x_end)].copy()
        _prophet_shift_df = _prophet_shift_df[
            (_prophet_shift_df["ds"] >= _chart_x_start) & (_prophet_shift_df["ds"] <= _chart_x_end)
        ].copy()
        fmp = fmp.drop(columns=["_plot_prophet_yhat"], errors="ignore").merge(_prophet_shift_df, on="ds", how="left")

        _cmp_col_map = {
            "prophet":  ("_plot_prophet_yhat",  "Prophet",  _COLORS["prophet"],  "solid", 2),
            "lstm":     ("_plot_lstm_yhat",     "LSTM",     _COLORS["lstm"],     "dot",   2),
            "xgboost":  ("_plot_xgb_yhat",      "XGBoost",  _COLORS["xgboost"],  "dash",  2),
        }
        fmp["_plot_lstm_yhat"] = pd.to_numeric(fmp.get("lstm_yhat"), errors="coerce")
        fmp["_plot_xgb_yhat"] = pd.to_numeric(fmp.get("xgb_yhat"), errors="coerce")
        # Apply prediction smoothing per-day to prevent cross-day boundary blending
        _cmp_smooth_win = max(1, round(int(st.session_state.get("pred_smooth_min", PRED_SMOOTH_MIN)) / BUCKET_MIN))
        if _cmp_smooth_win > 1:
            fmp = fmp.sort_values("ds").reset_index(drop=True)
            fmp["_date"] = fmp["ds"].dt.date
            for _sc in list(_cmp_col_map.values()):
                _sc_col = _sc[0]
                if _sc_col in fmp.columns:
                    fmp[_sc_col] = (
                        fmp.groupby("_date", sort=False)[_sc_col]
                        .transform(lambda x: pd.to_numeric(x, errors="coerce")
                                   .rolling(window=_cmp_smooth_win, center=True, min_periods=1)
                                   .mean())
                    )
            fmp = fmp.drop(columns=["_date"])

        _base_model_keys = ["prophet", "lstm", "xgboost"]
        _sel_cols = [_cmp_col_map[m][0] for m in _cmp_active_models
                     if m in _base_model_keys and _cmp_col_map[m][0] in fmp.columns]
        if _sel_cols:
            fmp["_dynamic_ensemble"] = _weighted_arrival_blend(fmp, _sel_cols)
            _dynamic_ensemble_label = (
                "Weighted blend (" +
                " + ".join(_cmp_model_labels.get(m, m) for m in _cmp_active_models if m in _base_model_keys) +
                ")"
            )
        else:
            _dynamic_ensemble_label = None

        fmp = _insert_overnight_gaps(fmp)
        # Individual selected-model traces
        for m in _cmp_active_models:
            if m not in _cmp_col_map:
                continue
            col, label, color, dash, width = _cmp_col_map[m]
            if col not in fmp.columns:
                continue
            y_vals = pd.to_numeric(fmp[col], errors="coerce")
            fig_cmp.add_trace(go.Scatter(
                x=_plotly_local_iso_series(fmp["ds"]),
                y=y_vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f"%{{x|%H:%M}}<br>{label}: %{{y:.1f}}<extra></extra>",
            ))

        # Dynamic ensemble trace (blend of selected visible model curves)
        if "_dynamic_ensemble" in fmp.columns and len(_sel_cols) > 1 and _dynamic_ensemble_label:
            _ens_label = _dynamic_ensemble_label
            fig_cmp.add_trace(go.Scatter(
                x=_plotly_local_iso_series(fmp["ds"]),
                y=pd.to_numeric(fmp["_dynamic_ensemble"], errors="coerce"),
                mode="lines",
                name=_ens_label,
                line=dict(color=_COLORS["ensemble"], width=3, dash="solid"),
                hovertemplate=f"%{{x|%H:%M}}<br>{_ens_label}: %{{y:.1f}}<extra></extra>",
            ))
        elif "_dynamic_ensemble" in fmp.columns and len(_sel_cols) == 1:
            # Only one model selected — no separate ensemble line needed
            pass

    # ── "Now" separator ───────────────────────────────────────────────────────
    _now_local = _now_local_ts()
    fig_cmp.add_shape(
        type="line",
        x0=_now_local,
        x1=_now_local,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#0f172a", width=1.5, dash="dot"),
    )
    fig_cmp.add_annotation(
        x=_now_local,
        y=1,
        xref="x",
        yref="paper",
        text="Now",
        showarrow=False,
        yshift=8,
        font=dict(color="#0f172a", size=11),
    )

    fig_cmp.update_layout(
        height=320,
        margin=dict(l=0, r=20, t=18, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        font=dict(color="#0f172a"),
        xaxis=dict(
            type="date",
            showgrid=False,
            tickformat="%d/%m %H:%M",
            hoverformat="%d/%m %H:%M",
            tickfont=dict(size=11, color="#475569"),
            autorange=False,
            range=[
                _plotly_local_iso(_chart_x_start),
                _plotly_local_iso(_chart_x_end),
            ],
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            zeroline=False,
            ticksuffix=" arr.",
            tickfont=dict(size=11, color="#475569"),
            title="",
        ),
    )
    st.plotly_chart(fig_cmp, width='stretch')
else:
    st.info("No data available. Run prediction and ensure entrance events are being recorded.")


# ── Dwell time history + XGBoost forecast ─────────────────────────────────────

_dwell_forecast_label = {"xgboost": "XGBoost", "lstm": "LSTM"}
_dwell_forecast_label_str = " + ".join(_dwell_forecast_label.get(m, m) for m in _dwell_forecast_models)
_section_title("Service Dwell Time", f"last {_hour_phrase(history_range_hours)} · {_dwell_forecast_label_str} forecast")
_per_lane_dwell_note = (
    " Per-lane median (7d): "
    + ", ".join(f"Lane {lid + 1} = {svc:.1f} min" for lid, svc in sorted(service_minutes_per_lane.items()))
    + "."
) if service_minutes_per_lane else ""
st.markdown(
    '<div class="detail-note">Observed checkout dwell times (15-min median) from <code>service_events</code>. '
    'Only <strong>solo-lane events</strong> (person alone in the lane when they left) are used to estimate pure service time — '
    'this removes queue wait time from the signal. '
    'Historical rows without this flag are recovered via a join with <code>entrance_events</code>; '
    'all events are used as a last fallback. '
    'The dwell forecast models learn how service time varies across the day and feed the result '
    'into the wait simulation as the per-lane capacity.'
    + (' ' + _per_lane_dwell_note if _per_lane_dwell_note else '')
    + '</div>',
    unsafe_allow_html=True,
)
_dwell_mode_label = "Safe" if _dwell_mode == "safe" else "Legacy"
_dwell_meta_parts = []
for _model_name in _dwell_forecast_models:
    _meta = _dwell_model_meta.get(_model_name, {})
    _counts = []
    if _meta.get("bucket_count") is not None:
        _counts.append(f"{_meta.get('bucket_count')} buckets")
    if _meta.get("event_count") is not None:
        _counts.append(f"{_meta.get('event_count')} events")
    _detail = _meta.get("reason", "—")
    if _counts:
        _detail += f" ({' / '.join(_counts)})"
    _dwell_meta_parts.append(f"{_dwell_forecast_label.get(_model_name, _model_name)}: {_detail}")
st.caption(
    f"Dwell model mode: {_dwell_mode_label}"
    + (f" · {' · '.join(_dwell_meta_parts)}" if _dwell_meta_parts else "")
)

if not service_history.empty:
    _sh = service_history.copy()
    _sh["ds"] = _to_local_series(_sh["ds"])
    _sh["dwell_min"] = pd.to_numeric(_sh["dwell_min"], errors="coerce")
    # Resample to fill all 15-min slots, smooth with rolling median, clip closed hours to None
    _sh = (
        _sh.set_index("ds")["dwell_min"]
        .resample("15min")
        .median()
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .reset_index()
        .rename(columns={"dwell_min": "dwell_min"})
    )
    _sh["dwell_min"] = _sh["dwell_min"].where(
        _sh["ds"].apply(lambda t: is_open(pd.Timestamp(t))),
        None,
    )
    _sh = _sh.reset_index(drop=True)

    def _dwell_features(ds_series):
        d = pd.DataFrame({"ds": ds_series})
        d["hour"]           = d["ds"].apply(lambda t: pd.Timestamp(t).hour)
        d["minute_of_hour"] = d["ds"].apply(lambda t: pd.Timestamp(t).minute)
        d["day_of_week"]    = d["ds"].apply(lambda t: pd.Timestamp(t).dayofweek)
        d["is_weekend"]     = (d["day_of_week"] >= 5).astype(int)
        return d

    # Forecast traces for selected dwell models
    _dwell_chart_traces = []
    if _dwell_models.get("xgboost") is not None:
        _model_x: list = []
        _model_y: list = []
        _now_local  = _now_local_ts()
        _grid_start = _chart_x_start
        _all_slots  = pd.date_range(
            start=_grid_start.floor("15min"),
            end=_now_local,
            freq="15min",
        )
        _open_slots = [t for t in _all_slots if is_open(pd.Timestamp(t))]
        if not pred_future.empty:
            _fut_slots = [pd.Timestamp(t) for t in pred_future["ds"] if is_open(pd.Timestamp(t))]
            _open_slots = _open_slots + _fut_slots

        if _open_slots:
            _grid_df = pd.DataFrame({"ds": _open_slots})
            _gf = _dwell_features(_grid_df["ds"])
            _grid_df["dwell_pred"] = _dwell_models["xgboost"].predict(_gf[_DWELL_FEAT]).clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)
            _prev_t = None
            for _, _gr in _grid_df.iterrows():
                _t = _as_local_naive_ts(_gr["ds"])
                if _prev_t is not None and (_t - _prev_t).total_seconds() > 30 * 60:
                    # Gap: midpoint timestamp with None y (don't put None in x)
                    _gap_t = _prev_t + (_t - _prev_t) / 2
                    _model_x.append(_gap_t)
                    _model_y.append(None)
                _model_x.append(_t)
                _model_y.append(float(_gr["dwell_pred"]))
                _prev_t = _t
        _dwell_chart_traces.append({
            "name": "XGBoost (history + forecast)",
            "x": pd.to_datetime(_model_x),
            "y": pd.Series(_model_y, dtype="float64"),
            "color": "#7c3aed",
            "dash": "solid",
        })
    if _dwell_pred_by_model.get("lstm"):
        _dwell_chart_traces.append({
            "name": "LSTM (forecast)",
            "x": pd.to_datetime(pred_future["ds"][:len(_dwell_pred_by_model['lstm'])]),
            "y": pd.Series(_dwell_pred_by_model["lstm"], dtype="float64"),
            "color": "#ea580c",
            "dash": "dash",
        })

    _dwell_x_start = int(_chart_x_start.timestamp() * 1000)
    _dwell_x_end   = int(_chart_x_end.timestamp() * 1000)


    fig_dwell = go.Figure()
    fig_dwell.add_trace(go.Scatter(
        x=_sh["ds"], y=_sh["dwell_min"],
        mode="lines",
        name="Observed (15 min median)",
        line=dict(color="#0891b2", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(8,145,178,0.12)",
        hovertemplate="%{x|%d/%m %H:%M}<br>Observed: %{y:.2f} min<extra></extra>",
    ))
    for _trace in _dwell_chart_traces:
        fig_dwell.add_trace(go.Scatter(
            x=_trace["x"], y=_trace["y"],
            mode="lines",
            name=_trace["name"],
            line=dict(color=_trace["color"], width=2, dash=_trace["dash"]),
            connectgaps=False,
            hovertemplate="%{x|%d/%m %H:%M}<br>" + _trace["name"] + ": %{y:.2f} min<extra></extra>",
        ))
    fig_dwell.add_hline(
        y=service_minutes,
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        annotation_text=f"Overall median: {service_minutes:.2f} min",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#64748b"),
    )
    fig_dwell.add_vline(
        x=pd.Timestamp.now().timestamp() * 1000,
        line=dict(color="#0f172a", width=1.5, dash="dot"),
        annotation_text="Now",
        annotation_position="top",
        annotation_font=dict(color="#0f172a", size=11),
    )

    # Single-shot layout — no incremental updates that could conflict
    fig_dwell.update_layout({
        "height": 260,
        "margin": {"l": 0, "r": 20, "t": 18, "b": 0},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "hovermode": "x unified",
        "showlegend": True,
        "legend": {"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 11}},
        "font": {"color": "#0f172a"},
        "xaxis": {
            "type": "date",
            "autorange": False,
            "range": [_dwell_x_start, _dwell_x_end],
            "showgrid": False,
            "tickformat": "%d/%m %H:%M",
            "hoverformat": "%d/%m %H:%M",
            "tickfont": {"size": 11, "color": "#475569"},
        },
        "yaxis": {
            "gridcolor": "#e2e8f0",
            "zeroline": False,
            "ticksuffix": " min",
            "tickfont": {"size": 11, "color": "#475569"},
            "title": "",
        },
    })
    st.plotly_chart(fig_dwell, width="stretch", key=f"dwell_{_span}")
    if not any(_dwell_models.get(m) is not None for m in _dwell_forecast_models):
        st.caption(
            "Dwell model unavailable — using flat median fallback. "
            + " / ".join(
                str(_dwell_model_meta.get(m, {}).get("reason", "Not enough service history."))
                for m in _dwell_forecast_models
            )
        )
else:
    st.info("No service_events data for this period — dwell is using the config fallback value.")


# ── Forecast section ───────────────────────────────────────────────────────────

_model_labels = {"ensemble": "Ensemble", "prophet": "Prophet", "lstm": "LSTM", "xgboost": "XGBoost"}
_active_models = st.session_state.get("arrival_models", ["prophet", "lstm", "xgboost"]) or ["prophet", "lstm", "xgboost"]
_active_model_label = " + ".join(_model_labels.get(m, m) for m in _active_models)
_show_live_wait_ref = st.checkbox(
    "Show live queue wait reference on forecast chart",
    value=st.session_state.get("show_live_wait_ref", False),
    key="show_live_wait_ref",
    help=(
        "Draws a horizontal reference line at (queue/lanes − 1) × service_time — "
        "the expected wait for someone joining the queue right now. "
        "Compare against the simulation forecast at t=0 to detect drift."
    ),
)
_live_wait_ref_min = max(0.0, (queue_count / max(selected_lanes, 1) - 1.0) * service_minutes)
_section_title(f"Predicted Wait - Next {FORECAST_DISPLAY_MIN} Min", f"{_active_model_label} arrivals · {_lane_phrase(selected_lanes)} open")
# Use base dwell (no residual correction) for all display values — correction only affects simulation
_eff_dwell = float(_dwell_per_slot_base[0]) if _dwell_per_slot_base else service_minutes
_dwell_source_label = (
    "Average of " + _dwell_forecast_label_str
    if len(_dwell_pred_by_model) > 1
    else _dwell_forecast_label.get(next(iter(_dwell_pred_by_model)), "Forecast")
    if _dwell_pred_by_model else ""
)
_dwell_label = f"{_eff_dwell:.1f} min/customer ({_dwell_source_label})" if _dwell_per_slot_base and _dwell_source_label else f"{_eff_dwell:.1f} min/customer"
_capacity_per_lane = BUCKET_MIN / _eff_dwell if _eff_dwell > 0 else 0
_capacity_per_bucket = selected_lanes * _capacity_per_lane
# Effective capacity = lane throughput + in-service customers completing this bucket
_effective_cap = _capacity_per_bucket + _in_service
_net_flow = round(_pred_mean, 1) - round(_effective_cap, 1)
# Use simulation result to qualify the trend — arithmetic on means ignores current state
_sim_wait_now = wait_0m if wait_0m is not None else 0.0
if _net_flow > 1.0:
    if _sim_wait_now <= 0.1 and _waiting_backlog < 1:
        _flow_label = f"+{_net_flow:.1f}/bucket above capacity — may build if arrivals continue"
    else:
        _flow_label = f"+{_net_flow:.1f}/bucket above capacity — queue growing"
elif _net_flow < -1.0:
    _flow_label = f"{_net_flow:.1f}/bucket below capacity — queue shrinking"
else:
    _flow_label = "arrivals ≈ capacity — balanced"
_per_lane_svc_note = (
    "Per-lane service time (7d median): "
    + ", ".join(f"Lane {lid + 1} = {svc:.1f} min" for lid, svc in sorted(service_minutes_per_lane.items()))
    + "."
) if service_minutes_per_lane else "Per-lane service time: no data yet (snapshot interval now 3 s — data will accumulate)."
_arrivals_note = (
    f"Forecast arrivals — mean: {_pred_mean:.1f} (range {_pred_min:.1f}–{_pred_max:.1f}) per {BUCKET_MIN}-min bucket. "
    f"Each lane handles {_capacity_per_lane:.1f} customers/slot (global avg service: {_dwell_label}). "
    f"{selected_lanes} lane{'s' if selected_lanes != 1 else ''} → {_capacity_per_bucket:.1f}/bucket"
    + (f" + {_in_service} in service = {_effective_cap:.1f} effective capacity. " if _in_service > 0 else ". ")
    + f"Net flow: {_flow_label}. "
    f"Waiting: {int(max(0, queue_count - selected_lanes))} · In service: {_in_service}. "
    + _per_lane_svc_note
    + (" Arrivals capped to observed rate." if _arrivals_capped else "")
) if not pred_df.empty else ""
st.markdown(
    f'<div class="detail-note">{lane_source_note} {lane_parameter_note} The trend below is recalculated in the dashboard '
    f'from the saved arrival forecast and current queue state.'
    + (f" {_arrivals_note}" if _arrivals_note else "")
    + (f" {_arrival_calib_note}" if _arrival_calib_note else "")
    + '</div>',
    unsafe_allow_html=True,
)

with st.expander("Arrival calibration debug", expanded=False):
    if _arrival_calib_debug_rows:
        st.dataframe(pd.DataFrame(_arrival_calib_debug_rows), width="stretch", hide_index=True)
        if _arrival_calib_active:
            st.markdown(
                '<div class="detail-note">Current arrivals are using the calibrated saved ensemble forecast from '
                '<code>queue_predictions.ensemble_yhat</code>.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="detail-note">Current arrivals are blended directly from the selected saved model columns, '
                'so any coefficient in <code>calibration.json</code> only affects this view if '
                '<code>ensemble_yhat</code> is the column being used.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No arrival forecast has been loaded yet.")

if not pred_df.empty:
    pf_raw = pred_future.copy()
    # Direct positional assignment — forecast_waits is built from pred_future in the same order
    if not forecast_waits.empty and len(forecast_waits) == len(pf_raw):
        pf_raw["wait_min"] = forecast_waits["wait_min"].values
    else:
        pf_raw["wait_min"] = float("nan")
    # Clip to display horizon
    _forecast_cutoff = _now_local_ts() + pd.Timedelta(minutes=FORECAST_DISPLAY_MIN)
    pf_raw = pf_raw[pf_raw["ds"] <= _forecast_cutoff]
    pf = _forecast_display_frame(pf_raw)
    # Smooth pf arrivals so singleton 5-min groups (1 raw bucket) don't show raw peak values
    if not pf.empty and len(pf) > 1:
        pf = pf.copy()
        pf["arrivals"] = pf["arrivals"].rolling(2, center=True, min_periods=1).mean().round(1)
    # Recompute stats from smoothed pf so note, mean line, and bars all use the same values
    if not pf.empty:
        _pf_arr = pf["arrivals"].dropna()
        if not _pf_arr.empty:
            _pred_mean = float(_pf_arr.mean())
            _pred_min  = float(_pf_arr.min())
            _pred_max  = float(_pf_arr.max())
    # Build wait chart — history always shown; future forecast trace only when pf has data
    fig_wait = go.Figure()
    # Historical actual wait approximation from queue snapshots
    if not queue_hist.empty:
        _qh_raw = queue_hist.copy()
        _qh_raw["timestamp"] = _to_local_series(_qh_raw["timestamp"])
        _qh_raw["queue_count"] = pd.to_numeric(_qh_raw["queue_count"], errors="coerce").fillna(0)
        _qh_raw["actual_wait"] = (_qh_raw["queue_count"] * service_minutes / max(detected_lanes, 1)).clip(lower=0)
        _qh_raw["actual_wait"] = _qh_raw["actual_wait"].where(
            _qh_raw["timestamp"].apply(lambda t: is_open(pd.Timestamp(t))),
            None,
        )
        fig_wait.add_trace(go.Scatter(
            x=_qh_raw["timestamp"],
            y=_qh_raw["actual_wait"],
            mode="lines",
            name="Actual raw",
            line=dict(color="#cbd5e1", width=1, dash="dot"),
            hovertemplate="%{x|%H:%M}<br>Raw wait (est.): %{y:.1f} min<extra></extra>",
        ))
        _qh = _qh_raw.copy()
        _qh["actual_wait"] = (
            _qh_raw["actual_wait"]
            .rolling(window=20, center=True, min_periods=1)
            .mean()
        )
        fig_wait.add_trace(go.Scatter(
            x=_qh["timestamp"],
            y=_qh["actual_wait"],
            mode="lines",
            name="Actual (smoothed)",
            line=dict(color="#94a3b8", width=2),
            fill="tozeroy",
            fillcolor="rgba(148,163,184,0.10)",
            hovertemplate="%{x|%H:%M}<br>Actual wait (smoothed): %{y:.1f} min<extra></extra>",
        ))
    # Historical forecast trace — derived from full_model_preds (past slots only)
    if not full_model_preds.empty:
        _hwp = full_model_preds.copy()
        _hwp["ds"] = _to_local_series(_hwp["ds"])
        _hwp = _hwp[_hwp["ds"] < _now_local_ts()].copy()
        _hwp_cols = [_model_col_map[m] for m in _selected_models
                     if _model_col_map.get(m) in _hwp.columns]
        if not _hwp_cols:
            for _fb in ["ensemble_yhat", "prophet_yhat", "lstm_yhat", "xgb_yhat"]:
                if _fb in _hwp.columns:
                    _hwp_cols = [_fb]
                    break
        if _hwp_cols:
            _hwp_arr = _weighted_arrival_blend(_hwp, _hwp_cols)
            _hwp["wait_min"] = (_hwp_arr * service_minutes / max(selected_lanes, 1)).clip(lower=0)
            _hwp = _hwp.sort_values("ds")
            _hwp["wait_min"] = (
                _hwp["wait_min"]
                .rolling(window=max(1, round(PRED_SMOOTH_MIN / BUCKET_MIN)),
                          center=True, min_periods=1)
                .mean()
            )
            _hwp = _insert_overnight_gaps(_hwp)
            fig_wait.add_trace(go.Scatter(
                x=_hwp["ds"],
                y=_hwp["wait_min"],
                mode="lines",
                name="Forecast (past)",
                line=dict(color="#2563eb", width=1.5, dash="dash"),
                hovertemplate="%{x|%H:%M}<br>Predicted wait (past): %{y:.1f} min<extra></extra>",
            ))
    # Future forecast trace — only when current predictions exist
    if not pf.empty:
        fig_wait.add_trace(
            go.Scatter(
                x=pf["ds"],
                y=pf["wait_min"],
                mode="lines",
                name="Forecast",
                line=dict(color="#2563eb", width=3),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.14)",
                hovertemplate="%{x|%H:%M}<br>Predicted wait: %{y:.1f} min<br>Lanes: " + str(selected_lanes) + "<extra></extra>",
            )
        )
    fig_wait.add_vline(
        x=_now_local_ts().timestamp() * 1000,
        line=dict(color="#0f172a", width=1.5, dash="dot"),
        annotation_text="Now",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#0f172a"),
    )
    fig_wait.add_hline(
        y=WAIT_BUSY_MIN,
        line=dict(color="#ea580c", width=1.5, dash="dot"),
        annotation_text=f"Queue building (>{WAIT_BUSY_MIN} min wait)",
        annotation_position="top right",
        annotation_font=dict(color="#c2410c", size=11),
    )
    fig_wait.add_hline(
        y=WAIT_ALERT_MIN,
        line=dict(color="#dc2626", width=1.5, dash="dot"),
        annotation_text=f"Open a lane (>{WAIT_ALERT_MIN} min wait)",
        annotation_position="top right",
        annotation_font=dict(color="#b91c1c", size=11),
    )
    if _show_live_wait_ref and _live_wait_ref_min > 0:
        fig_wait.add_hline(
            y=_live_wait_ref_min,
            line=dict(color="#0891b2", width=2, dash="dashdot"),
            annotation_text=f"Live queue ref: {_live_wait_ref_min:.1f} min ((N/lanes−1)×svc)",
            annotation_position="bottom right",
            annotation_font=dict(color="#0e7490", size=10),
        )
    _apply_chart_layout(fig_wait, height=260, y_suffix=" min", xaxis_type="date")
    fig_wait.update_layout(
        xaxis=dict(
            autorange=False,
            range=[
                _chart_x_start.strftime("%Y-%m-%d %H:%M:%S"),
                _chart_x_end.strftime("%Y-%m-%d %H:%M:%S"),
            ],
        ),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    )
    st.plotly_chart(fig_wait, width='stretch')

    if not pf.empty:

        # ── Service capacity vs arrival rate chart ──────────────────────────
        # Use raw 3-min buckets so bar count always matches FORECAST_DISPLAY_MIN / BUCKET_MIN (e.g. 3 bars).
        # 5-min grouping caused 2 or 3 bars unpredictably when buckets straddle 5-min boundaries.
        _cap_arr_df = pf_raw.copy() if not pf_raw.empty else pf.copy()

        # Per-bar capacity: average _dwell_per_slot_base over the 3-min buckets in each 5-min group
        # This keeps capacity consistent with the note (which also uses _dwell_per_slot_base)
        if _dwell_per_slot_base and not pf_raw.empty and not _cap_arr_df.empty:
            _dwell_raw_df = pd.DataFrame({
                "ds":    pf_raw["ds"].values,
                "dwell": _dwell_per_slot_base[:len(pf_raw)],
            })
            _dwell_raw_df["group_ds"] = pd.to_datetime(_dwell_raw_df["ds"]).dt.ceil("5min")
            _dwell_by_group = _dwell_raw_df.groupby("group_ds")["dwell"].mean()
            _bar_dwell_vals = [
                float(_dwell_by_group.get(pd.Timestamp(t), _eff_dwell))
                for t in _cap_arr_df["ds"]
            ]
            _bar_cap_vals = [selected_lanes * BUCKET_MIN / max(d, 0.5) for d in _bar_dwell_vals]
            _bar_cap_vals[0] = _bar_cap_vals[0] + _in_service
        else:
            _bar_cap_vals = [_effective_cap] * len(_cap_arr_df)

        _bar_colors = [
            "rgba(220, 38, 38, 0.6)" if float(a) > float(c) else "rgba(37, 99, 235, 0.5)"
            for a, c in zip(_cap_arr_df["arrivals"].fillna(0), _bar_cap_vals)
        ]
        # Queue depth from forecast_waits
        _pf_wait = _cap_arr_df["wait_min"].values if "wait_min" in _cap_arr_df.columns else None
        _queue_depth_series = None
        if _pf_wait is not None:
            _queue_depth_series = [max(0.0, float(w)) if not pd.isna(w) else 0.0 for w in _pf_wait]

        fig_cap = go.Figure()
        fig_cap.add_trace(
            go.Bar(
                x=_cap_arr_df["ds"],
                y=_cap_arr_df["arrivals"],
                name="Forecast arrivals",
                marker_color=_bar_colors,
                hovertemplate="%{x|%H:%M}<br>Arrivals: %{y:.1f}<extra></extra>",
            )
        )
        # Per-bar capacity as a step line (varies by time-of-day dwell prediction)
        _cap_label = (
            f"Effective capacity ({selected_lanes} lane{'s' if selected_lanes != 1 else ''}"
            + (f" + {_in_service} in service" if _in_service > 0 else "")
            + " — XGBoost dwell per slot)"
        )
        fig_cap.add_trace(
            go.Scatter(
                x=_cap_arr_df["ds"],
                y=_bar_cap_vals,
                name="Effective capacity",
                mode="lines+markers",
                line=dict(color="#16a34a", width=2, dash="dash"),
                marker=dict(size=5, color="#16a34a"),
                hovertemplate="%{x|%H:%M}<br>Capacity: %{y:.1f}<extra></extra>",
            )
        )
        if not pd.isna(_pred_mean):
            fig_cap.add_hline(
                y=_pred_mean,
                line=dict(color="#2563eb", width=1.5, dash="dot"),
                annotation_text=f"Mean: {_pred_mean:.1f}",
                annotation_position="top right",
                annotation_font=dict(color="#1d4ed8", size=10),
            )
        if _queue_depth_series is not None:
            fig_cap.add_trace(
                go.Scatter(
                    x=_cap_arr_df["ds"],
                    y=_queue_depth_series,
                    name="Predicted wait (min)",
                    mode="lines+markers",
                    line=dict(color="#ea580c", width=2),
                    marker=dict(size=4),
                    yaxis="y2",
                    hovertemplate="%{x|%H:%M}<br>Predicted wait: %{y:.1f} min<extra></extra>",
                )
            )
        _cap_arr_max = float(_cap_arr_df["arrivals"].max(skipna=True)) if not _cap_arr_df.empty else 0.0
        _y_max_cap = max(_cap_arr_max, max(_bar_cap_vals) if _bar_cap_vals else _effective_cap) * 1.25
        _apply_chart_layout(fig_cap, height=220, y_suffix=" arrivals", xaxis_type="date")
        _cap_x_start = _now_local_ts()
        _cap_x_end   = _cap_x_start + pd.Timedelta(minutes=FORECAST_DISPLAY_MIN)
        fig_cap.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
            title=dict(
                text="Arrivals vs effective capacity — red bars = queue growing",
                font=dict(size=11, color="#64748b"),
                x=0, y=0.98,
            ),
            yaxis=dict(range=[0, _y_max_cap]),
            xaxis=dict(range=[
                _cap_x_start.strftime("%Y-%m-%d %H:%M:%S"),
                _cap_x_end.strftime("%Y-%m-%d %H:%M:%S"),
            ]),
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                ticksuffix=" min",
                tickfont=dict(size=10, color="#ea580c"),
                title="",
            ) if _queue_depth_series is not None else {},
        )
        st.plotly_chart(fig_cap, width='stretch')

        # ── Queue simulation trace expander ────────────────────────────────────
        with st.expander("Queue simulation trace — how the wait is computed", expanded=False):
            _in_svc_now = _in_service

            # ── Metric pills ──────────────────────────────────────────────────
            pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
            pc1.metric("Active lanes", selected_lanes)
            pc2.metric("In service now", _in_svc_now, help="Customers currently at checkout (excluded from wait queue)")
            pc3.metric("Waiting (backlog)", int(max(0, queue_count - selected_lanes)), help="Customers in line, used as simulation seed")
            pc4.metric("Dwell / customer", f"{_eff_dwell:.1f} min" + (f" ({_dwell_source_label} now)" if _dwell_per_slot_base and _dwell_source_label else ""))
            pc5.metric("Capacity / bucket", f"{_effective_cap:.1f} cust",
                       help=f"{_capacity_per_bucket:.1f} lane throughput + {_in_svc_now} in service = {_effective_cap:.1f} effective")
            pc6.metric("Browsing gap", f"{BROWSING_GAP_MIN} min")

            # (dwell history chart is shown in the section above the forecast)

            # ── Re-simulate bucket by bucket to capture queue depth ────────────
            _sim_full = pred_future.merge(forecast_waits, on="ds", how="left").copy()
            _sim_full["arrivals"] = pd.to_numeric(_sim_full["arrivals"], errors="coerce").fillna(0)
            _sim_cutoff = _now_local_ts() + pd.Timedelta(minutes=FORECAST_DISPLAY_MIN)
            _sim = _sim_full[_sim_full["ds"] <= _sim_cutoff].copy()
            _q = float(_waiting_backlog)
            _qdepths = []
            for _si, (_, _r) in enumerate(_sim.iterrows()):
                _arr = max(0.0, float(_r["arrivals"]))
                _slot_dwell = (_dwell_per_slot_base[_si] if _dwell_per_slot_base and _si < len(_dwell_per_slot_base) else _eff_dwell)
                _slot_cap = selected_lanes * BUCKET_MIN / max(_slot_dwell, 0.5)
                if _si == 0:
                    _slot_cap += _in_service  # in-service customers complete in first bucket
                _q = min(max(0.0, _q + _arr - _slot_cap), float(selected_lanes * MAX_QUEUE_PER_LANE))
                _qdepths.append(_q)
            _sim["queue_depth"] = _qdepths

            # ── Queue depth chart ──────────────────────────────────────────────
            fig_s2 = go.Figure()
            _peak_q = max(_qdepths) if _qdepths else 0.0
            _y_max_s2 = max(_peak_q * 1.3, WAIT_ALERT_MIN + 5, 15.0)
            fig_s2.add_hrect(y0=0,  y1=5,         fillcolor="rgba(22,163,74,0.07)",  line_width=0)
            fig_s2.add_hrect(y0=5,  y1=10,        fillcolor="rgba(234,88,12,0.07)",  line_width=0)
            fig_s2.add_hrect(y0=10, y1=_y_max_s2, fillcolor="rgba(220,38,38,0.07)",  line_width=0)
            fig_s2.add_trace(go.Scatter(
                x=_sim["ds"], y=_sim["queue_depth"],
                mode="lines", fill="tozeroy",
                name="Queue depth",
                line=dict(color="#7c3aed", width=2),
                fillcolor="rgba(124,58,237,0.15)",
                hovertemplate="%{x|%H:%M}<br>Queue: %{y:.1f} waiting<extra></extra>",
            ))
            fig_s2.add_hline(y=5,  line=dict(color="#ea580c", width=1, dash="dot"),
                             annotation_text="Busy threshold", annotation_font=dict(size=10, color="#c2410c"))
            fig_s2.add_hline(y=10, line=dict(color="#dc2626", width=1, dash="dot"),
                             annotation_text="Alert threshold", annotation_font=dict(size=10, color="#b91c1c"))
            for _idx, _label, _color in [
                (WAIT_15M_INDEX, "+15 min", "#0891b2"),
                (WAIT_30M_INDEX, "+30 min", "#7c3aed"),
            ]:
                if _idx < len(_sim):
                    _pt = _sim.iloc[_idx]
                    fig_s2.add_vline(
                        x=pd.Timestamp(_pt["ds"]).timestamp() * 1000,
                        line=dict(color=_color, width=1.5, dash="dot"),
                        annotation_text=_label,
                        annotation_font=dict(size=10, color=_color),
                    )
            _apply_chart_layout(fig_s2, height=200, y_suffix=" ppl", xaxis_type="date")
            fig_s2.update_layout(
                showlegend=False,
                title=dict(text="Simulated queue depth (people waiting, bucket by bucket)", font=dict(size=11, color="#64748b"), x=0),
                yaxis=dict(range=[0, _y_max_s2]),
            )
            st.plotly_chart(fig_s2, width='stretch')

            st.markdown(
                '<div class="detail-note">'
                f"Formula: <b>queue[t] = max(0, queue[t-1] + arrivals[t] − capacity[t])</b>. "
                f"Capacity per slot = {selected_lanes} lane{'s' if selected_lanes > 1 else ''} × {BUCKET_MIN} min ÷ dwell ({_dwell_source_label or 'forecast'}, varies per slot, now {_eff_dwell:.1f} min) "
                f"= ~{_capacity_per_bucket:.1f}/bucket. "
                f"Slot 0 adds {_in_service} in-service → <b>{_effective_cap:.1f} effective capacity</b>. "
                f"Wait = (queue ÷ capacity) × {BUCKET_MIN} min. "
                f"Non-linear because dwell varies by time of day and queue is floored at 0."
                "</div>",
                unsafe_allow_html=True,
            )

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
            width='stretch',
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

_section_title("Lane Scenarios", f"wait at +10 min across 1–{MAX_LANES} lane options")
lane_summary = (
    f"Detected queue state is {_lane_phrase(detected_lanes)}. Dashboard forecast parameter is {_lane_phrase(selected_lanes)}."
)
st.markdown(f'<div class="detail-note">{lane_summary}</div>', unsafe_allow_html=True)

if lane_waits:
    scenario_lane_counts = list(range(1, MAX_LANES + 1))
    lane_cols = st.columns(len(scenario_lane_counts))
    for col, lane_count in zip(lane_cols, scenario_lane_counts):
        with col:
            wait_value = lane_waits.get(lane_count, {}).get("wait_10m")
            is_active = lane_count == selected_lanes
            is_default = lane_count == detected_lanes and selected_lanes == detected_lanes
            st.markdown(
                _lane_card_html(lane_count, wait_value, is_active=is_active, is_default=is_default),
                unsafe_allow_html=True,
            )
else:
    st.info("Run the prediction model to populate lane scenario comparisons.")


# ── Active lanes by time of day ───────────────────────────────────────────────

_section_title("Active Lanes", "live inference window")
_lane_win_col, _lane_win_note_col = st.columns([2, 5])
with _lane_win_col:
    st.select_slider(
        "Lane inference window",
        options=[5, 15, 30, 60],
        format_func=lambda v: f"{v} min" if v < 60 else "1 hour",
        key="lane_infer_window_min",
    )
with _lane_win_note_col:
    st.markdown(
        f'<div class="detail-note" style="margin-top:8px">How far back to look in '
        f'<code>service_events</code> when counting active lanes. '
        f'Shorter = more reactive; longer = smoother but may include closed lanes. '
        f'Currently using <b>{_lane_window} min</b>.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Active Lanes by Time of Day — historical pattern", expanded=False):
    _lh = load_lane_history(_span, service_minutes, slot_minutes=_lane_window)

    if not _lh.empty and "slot_label" in _lh.columns:
        import math as _math

        _MAX_LANES = 5
        # Build full slot index for open hours (per-day schedule)
        _g_oh, _g_om, _g_ch, _g_cm, _g_open_tot, _g_close_tot = _today_hours()
        _all_slots: list[str] = []
        _h = _g_oh
        while _h <= _g_ch:
            for _m in range(0, 60, _lane_window):
                if _g_open_tot <= _h * 60 + _m < _g_close_tot:
                    _all_slots.append(f"{_h:02d}:{_m:02d}")
            _h += 1
        _full_idx = pd.DataFrame({"slot_label": _all_slots})
        _fill = {"median_lanes": 0.0, "lanes_source": "none", "n_days_observed": 0,
                 "snapshot_lanes": 0.0, "n_days_snapshot": 0}
        _lhm = _full_idx.merge(_lh, on="slot_label", how="left")
        for _c, _v in _fill.items():
            if _c not in _lhm.columns:
                _lhm[_c] = _v
        _lhm = _lhm.fillna(_fill)

        _use_p1 = _lhm["median_lanes"] > 0
        _use_p2 = (~_use_p1) & (_lhm["snapshot_lanes"] > 0)

        _best_raw = _lhm["median_lanes"].where(
            _use_p1, _lhm["snapshot_lanes"].round().where(_use_p2, float("nan"))
        ).clip(0, _MAX_LANES)
        # Interpolate gaps: forward-fill then backward-fill missing slots
        _best = _best_raw.ffill().bfill().fillna(0.0).round()

        _bar_colors = [
            "#38bdf8" if p1 else ("#94a3b8" if p2 else ("#bfdbfe" if _best.iloc[i] > 0 else "#e2e8f0"))
            for i, (p1, p2) in enumerate(zip(_use_p1, _use_p2))
        ]
        _src_labels = [
            "service events" if p1 else ("snapshot" if p2 else ("interpolated" if _best.iloc[i] > 0 else "no data"))
            for i, (p1, p2) in enumerate(zip(_use_p1, _use_p2))
        ]
        _hover = [
            f"{slot}<br><b>{int(lanes)} lane{'s' if lanes != 1 else ''}</b> ({src})<extra></extra>"
            for slot, lanes, src in zip(_lhm["slot_label"], _best, _src_labels)
        ]

        fig_lanes_hist = go.Figure()
        fig_lanes_hist.add_trace(go.Bar(
            x=_lhm["slot_label"], y=_best,
            marker_color=_bar_colors, opacity=0.85,
            hovertemplate=_hover, showlegend=False,
        ))
        # Legend proxies
        if _use_p1.any():
            fig_lanes_hist.add_trace(go.Bar(x=[], y=[], name="Service events (direct count)",
                                            marker_color="#38bdf8"))
        if _use_p2.any():
            fig_lanes_hist.add_trace(go.Bar(x=[], y=[], name="Snapshot (observed)",
                                            marker_color="#94a3b8"))
        # Current inferred value as reference line
        if _inferred_lanes is not None:
            fig_lanes_hist.add_hline(
                y=_inferred_lanes,
                line=dict(color="#7c3aed", width=2, dash="dot"),
                annotation_text=f"Now: {_inferred_lanes} lane{'s' if _inferred_lanes != 1 else ''} (inferred)",
                annotation_position="top left",
                annotation_font=dict(color="#7c3aed", size=11),
            )
        elif live_lane_source:
            fig_lanes_hist.add_hline(
                y=detected_lanes,
                line=dict(color="#64748b", width=1.5, dash="dot"),
                annotation_text=f"Now: {detected_lanes} lane{'s' if detected_lanes != 1 else ''} (snapshot)",
                annotation_position="top left",
                annotation_font=dict(color="#64748b", size=11),
            )
        fig_lanes_hist.update_layout(
            height=240, margin=dict(l=0, r=20, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
            barmode="overlay",
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#e2e8f0", dtick=1, title="lanes",
                       tickfont=dict(size=11), range=[0, _MAX_LANES]),
        )
        st.plotly_chart(fig_lanes_hist, width="stretch", key="lanes_history")
    else:
        st.info("Not enough historical data to build lane schedule. Accumulate more service_events or queue_state_snapshots.")

    st.markdown(f"""
**How lane count is estimated — and why it matters**

The dashboard estimates how many checkout lanes are open using three sources, in priority order:

| Priority | Source | Method |
|---|---|---|
| 1 (best) | `service_events.lane_id` | Counts distinct lane IDs with qualifying checkouts in the last {_lane_window} min |
| 2 | `service_events` throughput | `events ÷ (window_min ÷ dwell_min)` — infers parallel checkouts from rate |
| 3 (fallback) | `queue_state_snapshots.active_lanes` | Hardware-reported snapshot value |

**Impact on the wait prediction**

- **Service capacity per bucket** = `lanes × bucket_min ÷ dwell_min`. More lanes → more customers served per 3-min slot → shorter predicted wait.
- If lane count is **overestimated**, the model predicts shorter waits than reality — the system appears less stressed than it is.
- If lane count is **underestimated**, the model over-predicts wait times — false alerts.
- The "Now" dotted line shows the current inferred count being used for the next-30-min forecast (estimated from the last {_lane_window} min of service events).

**Impact on service time (dwell)**

Lane count does **not** directly change the per-customer dwell time estimate. Dwell comes from the XGBoost model trained on `service_events.total_dwell_sec`. Lane count and dwell are independent inputs to the queue simulation.
""")


# ── Live queue + dwell history ────────────────────────────────────────────────

_section_title(
    f"Live Queue And Dwell - Last {_hour_phrase(history_range_hours).title()}",
    "people in queue and average dwell time",
)
_dwell_correction_on = st.checkbox(
    "Show corrected dwell estimate ((queue − 1) × service time)",
    value=st.session_state.get("dwell_correction_on", False),
    key="dwell_correction_on",
    help=(
        "Overlays an estimate of expected wait per lane: (queue/lanes − 1) × service_time. "
        "Divides total queue by active lanes to get per-lane depth, then subtracts "
        "the person already at the till (in service, not waiting)."
    ),
)
_service_sec = service_minutes * 60.0

_dwell_note_correction = (
    f" A corrected estimate ((queue / lanes − 1) × {service_minutes:.1f} min) is overlaid as a dashed line — "
    "divides queue by active lanes, then subtracts the person already at the till."
    if _dwell_correction_on else ""
)
st.markdown(
    f'<div class="detail-note">Showing the last {_hour_phrase(history_range_hours)} of queue snapshots. '
    'Top panel shows queue depth in people. Bottom panel shows average dwell time in seconds from the same live snapshots, so you can see whether congestion is coming from more people, slower service, or both.'
    f'{_dwell_note_correction}</div>',
    unsafe_allow_html=True,
)

if not queue_hist.empty:
    qh = queue_hist.copy()
    qh["timestamp"] = _to_local_series(qh["timestamp"])
    qh = qh[qh["timestamp"].apply(lambda t: is_open(pd.Timestamp(t)))]
    qh["queue_count"] = pd.to_numeric(qh["queue_count"], errors="coerce")
    qh["avg_dwell_sec"] = pd.to_numeric(qh["avg_dwell_sec"], errors="coerce")
    qh = qh.set_index("timestamp").resample("1min").mean().dropna(subset=["queue_count"], how="all").reset_index()
    # Apply global prediction smoothing per-day to avoid cross-day boundary blending
    _qh_smooth_win = max(1, int(st.session_state.get("pred_smooth_min", PRED_SMOOTH_MIN)))
    if _qh_smooth_win > 1:
        qh = qh.sort_values("timestamp").reset_index(drop=True)
        qh["_date"] = qh["timestamp"].dt.date
        for _qh_col in ("queue_count", "avg_dwell_sec"):
            if _qh_col in qh.columns:
                qh[_qh_col] = (
                    qh.groupby("_date", sort=False)[_qh_col]
                    .transform(lambda x: x.rolling(window=_qh_smooth_win, center=True, min_periods=1).mean())
                )
        qh = qh.drop(columns=["_date"])

    if _dwell_correction_on:
        _lanes_snap = qh["active_lanes"].fillna(selected_lanes).clip(lower=1) if "active_lanes" in qh.columns else selected_lanes
        qh["corrected_dwell_sec"] = ((qh["queue_count"].fillna(0) / _lanes_snap - 1).clip(lower=0) * _service_sec).clip(lower=0)

    qh = _insert_overnight_gaps(qh, ds_col="timestamp")

    has_dwell = qh["avg_dwell_sec"].notna().any()
    _show_legend = has_dwell and _dwell_correction_on
    fig_q = make_subplots(
        rows=2 if has_dwell else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.66, 0.34] if has_dwell else None,
        subplot_titles=["Queue depth (people)", "Average dwell (sec)"] if has_dwell else ["Queue depth (people)"],
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
                y=qh["avg_dwell_sec"],
                name="Observed avg dwell",
                mode="lines",
                line=dict(color="#7c3aed", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124, 58, 237, 0.12)",
                hovertemplate="%{x|%H:%M}<br>Avg dwell: %{y:.0f} sec<extra></extra>",
            ),
            row=2,
            col=1,
        )
        if _dwell_correction_on and "corrected_dwell_sec" in qh.columns:
            fig_q.add_trace(
                go.Scatter(
                    x=qh["timestamp"],
                    y=qh["corrected_dwell_sec"],
                    name=f"Corrected estimate ((N/lanes−1) × {service_minutes:.1f} min)",
                    mode="lines",
                    line=dict(color="#ea580c", width=2, dash="dash"),
                    hovertemplate="%{x|%H:%M}<br>Corrected dwell: %{y:.0f} sec<extra></extra>",
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
        showlegend=_show_legend,
        legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=11)) if _show_legend else {},
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
            title="Sec",
            zeroline=False,
            tickfont=dict(size=11, color="#475569"),
            row=2,
            col=1,
        )

    st.plotly_chart(fig_q, width='stretch')
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
    fdq = fdq[fdq["timestamp"].apply(lambda t: is_open(pd.Timestamp(t)))]
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
    st.plotly_chart(fig_fd, width='stretch')
else:
    st.info("No queue data has been recorded today yet.")


# ── Entries by hour + Status breakdown (two columns) ──────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    _section_title("Entries by Hour", "traffic pattern for today")
    if not traffic_today.empty:
        tt = traffic_today.copy()
        tt["hour"] = _to_local_series(tt["hour"])
        tt = tt[tt["hour"].apply(lambda t: is_open(pd.Timestamp(t)))]
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
        st.plotly_chart(fig_t, width='stretch')
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
        st.plotly_chart(fig_donut, width='stretch')
    else:
        st.info("No prediction history is available for today yet.")


# ── Demographics ──────────────────────────────────────────────────────────────

st.markdown('<hr class="tracker-divider">', unsafe_allow_html=True)
_section_title("Customer Demographics Today", "gender and age profile of today's visitors")

_has_gender = not demo_gender.empty and demo_gender["count"].sum() > 0
_has_age    = not demo_age.empty    and demo_age["count"].sum() > 0
_has_hourly = not demo_hourly.empty

if not _has_gender and not _has_age:
    st.info("No demographic data recorded today yet. Check that the entrance camera is writing age and gender to entrance_events.")
else:
    dem_c1, dem_c2, dem_c3 = st.columns([1.2, 1.8, 1.0])

    # ── Gender donut ──────────────────────────────────────────────────────────
    with dem_c1:
        st.markdown('<div class="section-subtitle">Gender split</div>', unsafe_allow_html=True)
        if _has_gender:
            _gender_colors = {"male": "#2563eb", "female": "#db2777", "unknown": "#94a3b8"}
            _gdf = demo_gender.copy()
            _gdf["gender"] = _gdf["gender"].str.lower()
            _total_g = int(_gdf["count"].sum())
            fig_gender = go.Figure(go.Pie(
                labels=_gdf["gender"].str.capitalize(),
                values=_gdf["count"],
                hole=0.58,
                marker_colors=[_gender_colors.get(g, "#94a3b8") for g in _gdf["gender"]],
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value} people<extra></extra>",
            ))
            fig_gender.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white",
                showlegend=False,
                font=dict(color="#0f172a"),
                annotations=[dict(
                    text=f"<b>{_total_g}<br>total</b>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="#0f172a"),
                )],
            )
            st.plotly_chart(fig_gender, width='stretch')
        else:
            st.info("No gender data today.")

    # ── Age group bar ─────────────────────────────────────────────────────────
    with dem_c2:
        st.markdown('<div class="section-subtitle">Age group distribution</div>', unsafe_allow_html=True)
        if _has_age:
            _adf = demo_age.copy()
            _age_order = ["18-30", "30-50", "50+"]
            _adf["age_group"] = pd.Categorical(_adf["age_group"], categories=_age_order, ordered=True)
            _adf = _adf.sort_values("age_group")
            _age_colors = {"18-30": "#0ea5e9", "30-50": "#6366f1", "50+": "#f59e0b"}
            _total_a = int(_adf["count"].sum())
            fig_age = go.Figure(go.Bar(
                x=_adf["age_group"],
                y=_adf["count"],
                marker_color=[_age_colors.get(g, "#94a3b8") for g in _adf["age_group"]],
                text=_adf["count"],
                textposition="outside",
                hovertemplate="%{x}: %{y} people<extra></extra>",
            ))
            _apply_chart_layout(fig_age, height=240)
            st.plotly_chart(fig_age, width='stretch')
        else:
            st.info("No age data today.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    with dem_c3:
        st.markdown('<div class="section-subtitle">Summary</div>', unsafe_allow_html=True)
        if _has_gender:
            _gdf2 = demo_gender.copy()
            _gdf2["gender"] = _gdf2["gender"].str.lower()
            _female_n = int(_gdf2.loc[_gdf2["gender"] == "female", "count"].sum())
            _male_n   = int(_gdf2.loc[_gdf2["gender"] == "male",   "count"].sum())
            _total_n  = _female_n + _male_n
            _female_pct = round(_female_n / _total_n * 100) if _total_n > 0 else 0
            st.metric("Female", f"{_female_pct}%", f"{_female_n} visitors")
            st.metric("Male",   f"{100 - _female_pct}%", f"{_male_n} visitors")
        if _has_age and not demo_hourly.empty:
            _avg_age_today = demo_hourly["avg_age"].mean()
            if pd.notna(_avg_age_today):
                st.metric("Avg age", f"{_avg_age_today:.0f} yrs")

    # ── Hourly demographics chart ─────────────────────────────────────────────
    if _has_hourly and _has_gender:
        with st.expander("Demographics by hour of day", expanded=False):
            _hdf = demo_hourly.copy()
            _hdf = _hdf[_hdf["total"] > 0].copy()
            _hdf["hour_label"] = _hdf["hour"].apply(lambda h: f"{int(h):02d}:00")
            _hdf["female_pct"] = (_hdf["female_count"] / _hdf["total"].clip(lower=1) * 100).round(1)
            _hdf["avg_age"]    = pd.to_numeric(_hdf["avg_age"], errors="coerce")

            fig_hourly = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.55, 0.45],
                vertical_spacing=0.08,
                subplot_titles=["Female % by hour", "Average age by hour"],
            )
            fig_hourly.add_trace(go.Bar(
                x=_hdf["hour_label"],
                y=_hdf["female_pct"],
                name="Female %",
                marker_color="#db2777",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ), row=1, col=1)
            fig_hourly.add_hline(
                y=50, line_dash="dot", line_color="#94a3b8", row=1, col=1,
                annotation_text="50%", annotation_position="right",
            )
            fig_hourly.add_trace(go.Scatter(
                x=_hdf["hour_label"],
                y=_hdf["avg_age"],
                name="Avg age",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=2),
                marker=dict(size=6),
                hovertemplate="%{x}: %{y:.1f} yrs<extra></extra>",
            ), row=2, col=1)
            fig_hourly.update_layout(
                height=360,
                margin=dict(l=0, r=20, t=30, b=0),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                font=dict(color="#0f172a", size=11),
            )
            fig_hourly.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            fig_hourly.update_xaxes(showgrid=False, row=2, col=1)
            st.plotly_chart(fig_hourly, width='stretch')
            st.caption(
                "Female % above 50% = more women than men in that hour. "
                "Average age shows whether the morning vs afternoon crowd is older or younger."
            )


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Auto-refresh every {REFRESH_SEC}s · Busy threshold: {WAIT_BUSY_MIN} min · "
    f"Alert threshold: {WAIT_ALERT_MIN} min · Default lanes: {DEFAULT_LANES}"
)
