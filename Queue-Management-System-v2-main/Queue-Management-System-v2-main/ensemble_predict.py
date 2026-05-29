from __future__ import annotations

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pickle
import time as _time
from pathlib import Path
import sys
import psycopg2
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

from prediction.core import (  # noqa: E402
    BUCKET_MINUTES,
    DEFAULT_DWELL_MIN,
    DEFAULT_LANES,
    DWELL_MAX_CAP,
    DWELL_MIN_FLOOR,
    FORECAST_STEPS,
    LAG_HOUR_STEPS,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    MIN_REAL_DAYS_FOR_SIM,
    ROLLING_WINDOW_STEPS,
    SEQUENCE_LEN,
    SHOP_OPEN_TOT,
    SNAPSHOT_MAX_AGE_MIN,
    WAIT_45M_INDEX,
    add_closed_zeros,
    build_sim_history,
    compute_wait_estimates,
    future_open_timestamps,
    get_where_clause,
    is_open,
)
from prediction.pipeline import DB_CONFIG  # noqa: E402

W_PROPHET           = float(os.getenv("W_PROPHET",          0.40))
W_LSTM              = float(os.getenv("W_LSTM",             0.30))
W_XGB               = float(os.getenv("W_XGB",             0.30))
MODEL_MAX_AGE_HOURS = int(os.getenv("MODEL_MAX_AGE_HOURS",  24))
LSTM_OPENING_BLEND_MAX = float(os.getenv("LSTM_OPENING_BLEND_MAX", 0.85))
LSTM_OPENING_BLEND_MIN = float(os.getenv("LSTM_OPENING_BLEND_MIN", 0.25))
LSTM_OPENING_BLEND_MINUTES = int(os.getenv("LSTM_OPENING_BLEND_MINUTES", 60))
LSTM_OPENING_FLOOR_SHARE_MAX = float(os.getenv("LSTM_OPENING_FLOOR_SHARE_MAX", 0.65))
LSTM_OPENING_FLOOR_SHARE_MIN = float(os.getenv("LSTM_OPENING_FLOOR_SHARE_MIN", 0.20))
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
LEGACY_MODEL_DIRS = [Path.cwd() / "models", ROOT_DIR / "models"]
MODELS_DIR.mkdir(exist_ok=True)


def _resolve_model_path(filename: str) -> Path:
    preferred = MODELS_DIR / filename
    if preferred.exists():
        return preferred
    for model_dir in LEGACY_MODEL_DIRS:
        candidate = Path(model_dir) / filename
        if candidate.exists():
            return candidate
    return preferred


LSTM_PATH         = _resolve_model_path(f'lstm_queue_{BUCKET_MINUTES}m.keras')
SCALER_PATH       = _resolve_model_path(f'lstm_scaler_{BUCKET_MINUTES}m.pkl')
XGB_PATH          = _resolve_model_path(f'xgb_queue_{BUCKET_MINUTES}m.json')
PROPHET_PATH      = _resolve_model_path(f'prophet_queue_{BUCKET_MINUTES}m.pkl')
PROPHET_META_PATH = _resolve_model_path(f'prophet_queue_{BUCKET_MINUTES}m_meta.json')
DWELL_MODEL_PATH  = _resolve_model_path(f'xgb_dwell_{BUCKET_MINUTES}m.json')

DWELL_FEATURE_COLS = ["hour", "minute_of_hour", "day_of_week", "is_weekend"]


def _build_lstm_opening_profile(history_df: pd.DataFrame, scaler: MinMaxScaler) -> dict[int, float]:
    """Build a historical opening-hour target in scaled space for early-day LSTM stabilization."""
    if history_df.empty:
        return {}

    opening_end_tot = SHOP_OPEN_TOT + max(BUCKET_MINUTES, LSTM_OPENING_BLEND_MINUTES)
    profile_df = history_df.copy()
    profile_df["ds"] = pd.to_datetime(profile_df["ds"], errors="coerce")
    profile_df["y"] = pd.to_numeric(profile_df["y"], errors="coerce").fillna(0.0)
    profile_df = profile_df.dropna(subset=["ds"])

    if profile_df.empty:
        return {}

    profile_df["minute_of_day"] = profile_df["ds"].dt.hour * 60 + profile_df["ds"].dt.minute
    profile_df = profile_df[
        profile_df["ds"].map(is_open)
        & profile_df["minute_of_day"].between(SHOP_OPEN_TOT, opening_end_tot - 1)
    ]

    if profile_df.empty:
        return {}

    opening_profile = (
        profile_df.groupby("minute_of_day", as_index=True)["y"]
        .median()
        .sort_index()
    )

    scaled_profile = scaler.transform(opening_profile.to_numpy(dtype=float).reshape(-1, 1)).flatten()
    return {
        int(minute_of_day): float(scaled_value)
        for minute_of_day, scaled_value in zip(opening_profile.index.tolist(), scaled_profile.tolist())
    }


def _build_lstm_opening_profile_raw(history_df: pd.DataFrame) -> dict[int, float]:
    """Build a raw-space opening profile for a conservative morning floor."""
    if history_df.empty:
        return {}

    opening_end_tot = SHOP_OPEN_TOT + max(BUCKET_MINUTES, LSTM_OPENING_BLEND_MINUTES)
    profile_df = history_df.copy()
    profile_df["ds"] = pd.to_datetime(profile_df["ds"], errors="coerce")
    profile_df["y"] = pd.to_numeric(profile_df["y"], errors="coerce").fillna(0.0)
    profile_df = profile_df.dropna(subset=["ds"])

    if profile_df.empty:
        return {}

    profile_df["minute_of_day"] = profile_df["ds"].dt.hour * 60 + profile_df["ds"].dt.minute
    profile_df = profile_df[
        profile_df["ds"].map(is_open)
        & profile_df["minute_of_day"].between(SHOP_OPEN_TOT, opening_end_tot - 1)
    ]

    if profile_df.empty:
        return {}

    opening_profile = (
        profile_df.groupby("minute_of_day", as_index=True)["y"]
        .median()
        .sort_index()
    )
    return {int(minute_of_day): float(value) for minute_of_day, value in opening_profile.items()}


def _models_are_fresh() -> bool:
    paths = [LSTM_PATH, SCALER_PATH, XGB_PATH, PROPHET_PATH]
    if not all(os.path.exists(p) for p in paths):
        return False
    oldest = min(os.path.getmtime(p) for p in paths)
    return (_time.time() - oldest) < MODEL_MAX_AGE_HOURS * 3600


DATA_SPAN_DAYS_DEFAULT = int(os.getenv("DATA_SPAN_DAYS", 30))


def run_ensemble_forecast(source: str = "REAL", bootstrap: bool = False, data_span_days: int = DATA_SPAN_DAYS_DEFAULT) -> dict:
    """Train Prophet + LSTM + XGBoost ensemble and return a forecast dict."""
    where_clause = get_where_clause(source)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"[Ensemble] Connected to PostgreSQL (Source: {source}, span: {data_span_days}d) [ok]")

    clean_filter = f"timestamp >= NOW() - INTERVAL '{data_span_days} days' AND dwell_seconds >= 10"
    if where_clause:
        data_filter = f"{where_clause} AND {clean_filter}"
    else:
        data_filter = f"WHERE {clean_filter}"

    query = f"""
        SELECT
            time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS bucket,
            COUNT(*) AS entry_count
        FROM entrance_events
        {data_filter}
        GROUP BY bucket
        ORDER BY bucket
    """
    df_real = pd.read_sql(query, conn)
    conn.close()
    print(f"[Ensemble] Loaded {len(df_real)} {BUCKET_MINUTES}-min buckets from DB")

    if len(df_real) > 0:
        df_real["entry_count"] = pd.to_numeric(df_real["entry_count"], errors="coerce").fillna(0).astype(int)

    df_real_r = df_real.rename(columns={"bucket": "ds", "entry_count": "y"})
    df_real_r["ds"] = pd.to_datetime(df_real_r["ds"]).dt.tz_localize(None)
    df_real_r["y"]  = pd.to_numeric(df_real_r["y"], errors="coerce").fillna(0).astype(int)

    real_span_days = (
        (df_real_r["ds"].max() - df_real_r["ds"].min()).total_seconds() / 86400
        if len(df_real_r) > 1 else 0
    )

    # ── 2. Bootstrap ─────────────────────────────────────────────────────────
    if bootstrap and real_span_days < MIN_REAL_DAYS_FOR_SIM:
        df_sim = build_sim_history(days=7)
        df = (
            pd.concat([df_real_r, df_sim], ignore_index=True)
            .drop_duplicates(subset="ds")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        print(f"[Ensemble] Bootstrap ON — {len(df_real_r)} real + {len(df_sim)} synthetic rows")
    else:
        df = df_real_r.sort_values("ds").reset_index(drop=True)
        if bootstrap and real_span_days >= MIN_REAL_DAYS_FOR_SIM:
            print(f"[Ensemble] {real_span_days:.0f} days real — bootstrap not needed")
        else:
            print(f"[Ensemble] {real_span_days:.0f} days real — training on real data only")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce").fillna(0)
    df = df.dropna(subset=["ds"])

    if df.empty:
        raise RuntimeError("No valid data to train. Check --source filter or DB contents.")

    # ── 3. Outlier cap ───────────────────────────────────────────────────────
    non_zero = df.loc[df["y"] > 0, "y"]
    if not non_zero.empty:
        cap_value = max(float(non_zero.quantile(0.99)), 30.0)
        df["y"] = df["y"].clip(upper=cap_value)

    # ── 3b. 15-min rolling median smooth (5 × 3-min buckets) ─────────────────
    df["y"] = (
        df["y"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .round(2)
    )
    print(f"[Ensemble] Applied 15-min rolling median smooth to training data")

    # ── 4. Closed-hour zeros ─────────────────────────────────────────────────
    df = add_closed_zeros(df[["ds", "y"]].copy(), days=data_span_days)

    print(f"[Ensemble] DataFrame shape: {df.shape}")

    future_timestamps = future_open_timestamps()
    future_df = pd.DataFrame({"ds": future_timestamps})
    n_steps = len(future_timestamps)

    # ── MODEL 1 — PROPHET ────────────────────────────────────────────────────
    _prophet_meta = {}
    if os.path.exists(PROPHET_META_PATH):
        try:
            import json as _json
            with open(PROPHET_META_PATH) as _mf:
                _prophet_meta = _json.load(_mf)
        except Exception:
            _prophet_meta = {}
    _prophet_days_changed = _prophet_meta.get("data_span_days") != data_span_days

    if _models_are_fresh() and not _prophet_days_changed:
        print("\n[Prophet] Loading saved model...")
        with open(PROPHET_PATH, "rb") as _f:
            prophet_model = pickle.load(_f)
        print("[Prophet] Loaded [ok]")
    else:
        _reason = "days changed" if _prophet_days_changed else "model stale"
        print(f"\n[Prophet] Training ({_reason}, span={data_span_days}d)...")
        _cps = 0.30 if data_span_days <= 10 else (0.15 if data_span_days <= 20 else 0.1)
        prophet_model = Prophet(
            daily_seasonality=False,   # replaced by custom higher-order below
            weekly_seasonality=(data_span_days >= 7),
            yearly_seasonality=False,
            changepoint_prior_scale=_cps,
            seasonality_prior_scale=10.0,
            interval_width=0.80,
        )
        # Higher Fourier order → captures sharp morning/lunch/evening transitions
        prophet_model.add_seasonality(name='daily',  period=1,   fourier_order=10)
        if data_span_days >= 7:
            prophet_model.add_seasonality(name='weekly', period=7, fourier_order=5)
        prophet_model.fit(df)
        with open(PROPHET_PATH, "wb") as _f:
            pickle.dump(prophet_model, _f)
        import json as _json
        with open(PROPHET_META_PATH, "w") as _mf:
            _json.dump({"data_span_days": data_span_days}, _mf)
        print("[Prophet] Trained and saved [ok]")
    forecast       = prophet_model.predict(future_df)
    prophet_preds  = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    prophet_vals   = prophet_preds["yhat"].clip(lower=0).values

    # ── MODEL 2 — LSTM ───────────────────────────────────────────────────────
    print("\n[LSTM] Preparing sequences...")

    def _make_sequences(data, seq_len):
        X, y_out = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i: i + seq_len])
            y_out.append(data[i + seq_len])
        return np.array(X), np.array(y_out)

    y_values = pd.to_numeric(df["y"], errors="coerce").fillna(0).values
    y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Use open-hours-only data for LSTM training and seed.
    # Closed-hour zeros contaminate the sequence window and cause the model to
    # predict near-zero when it runs outside shop hours (zero-seed problem).
    _open_mask_lstm = np.array([is_open(pd.Timestamp(t)) for t in df["ds"].values])
    y_open_values   = y_values[_open_mask_lstm]

    if _models_are_fresh():
        print("[LSTM] Loading saved model and scaler...")
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
        with open(SCALER_PATH, "rb") as _f:
            scaler = pickle.load(_f)
        y_open_scaled = scaler.transform(y_open_values.reshape(-1, 1))
        print("[LSTM] Loaded [ok]")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(df_real_r) > 0:
            real_y_all  = pd.to_numeric(df_real_r["y"], errors="coerce").fillna(0).values
            _real_open  = np.array([is_open(pd.Timestamp(t)) for t in df_real_r["ds"].values])
            real_y_open = real_y_all[_real_open]
            scaler.fit((real_y_open if len(real_y_open) > 0 else real_y_all).reshape(-1, 1))
        else:
            scaler.fit(y_open_values.reshape(-1, 1))
        y_open_scaled = scaler.transform(y_open_values.reshape(-1, 1))

        X_lstm, y_lstm = _make_sequences(y_open_scaled, SEQUENCE_LEN)
        print(f"[LSTM] Training on {len(X_lstm)} open-hour sequences (epochs=20)...")
        lstm_model = Sequential([
            LSTM(64, input_shape=(SEQUENCE_LEN, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0)
        lstm_model.save(LSTM_PATH)
        with open(SCALER_PATH, "wb") as _f:
            pickle.dump(scaler, _f)
        print("[LSTM] Trained and saved [ok]")

    opening_profile_source = df_real_r if len(df_real_r) > 0 else df[["ds", "y"]]
    opening_profile_scaled = _build_lstm_opening_profile(opening_profile_source, scaler)
    opening_profile_raw = _build_lstm_opening_profile_raw(opening_profile_source)
    if opening_profile_scaled:
        print(
            "[LSTM] Built opening profile "
            f"({len(opening_profile_scaled)} buckets over first {LSTM_OPENING_BLEND_MINUTES} min)"
        )

    # Seed from the last SEQUENCE_LEN open-hour values — not the raw tail
    # which may end in overnight zeros and cause near-zero rollout predictions.
    last_seq = y_open_scaled[-SEQUENCE_LEN:].copy() if len(y_open_scaled) >= SEQUENCE_LEN \
               else y_open_scaled.copy()
    lstm_scaled_preds = []
    opening_span = max(BUCKET_MINUTES, LSTM_OPENING_BLEND_MINUTES)
    opening_fade_span = max(1, opening_span - BUCKET_MINUTES)
    for ts in future_timestamps:
        inp      = last_seq.reshape(1, SEQUENCE_LEN, 1)
        next_val = lstm_model.predict(inp, verbose=0)[0][0]
        ts_dt = pd.Timestamp(ts)
        minute_of_day = ts_dt.hour * 60 + ts_dt.minute
        opening_target = opening_profile_scaled.get(minute_of_day)
        if opening_target is not None and SHOP_OPEN_TOT <= minute_of_day < (SHOP_OPEN_TOT + opening_span):
            progress = min(1.0, max(0.0, (minute_of_day - SHOP_OPEN_TOT) / opening_fade_span))
            blend_weight = LSTM_OPENING_BLEND_MAX + (
                (LSTM_OPENING_BLEND_MIN - LSTM_OPENING_BLEND_MAX) * progress
            )
            next_val = ((1.0 - blend_weight) * float(next_val)) + (blend_weight * opening_target)
        lstm_scaled_preds.append(next_val)
        last_seq = np.append(last_seq[1:], [[next_val]], axis=0)

    lstm_vals = scaler.inverse_transform(
        np.array(lstm_scaled_preds).reshape(-1, 1)
    ).flatten().clip(min=0)

    # ── MODEL 3 — XGBOOST ───────────────────────────────────────────────────
    print("\n[XGBoost] Building features...")

    FEATURE_COLS = [
        "hour", "minute_of_hour", "day_of_week", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_12", "rolling_mean_6",
    ]

    def _build_features(data_df):
        d = data_df.copy()
        d["hour"]           = d["ds"].dt.hour
        d["minute_of_hour"] = d["ds"].dt.minute
        d["day_of_week"]    = d["ds"].dt.dayofweek
        d["is_weekend"]     = (d["day_of_week"] >= 5).astype(int)
        d["lag_1"]          = d["y"].shift(1)
        d["lag_2"]          = d["y"].shift(2)
        d["lag_3"]          = d["y"].shift(3)
        d["lag_12"]         = d["y"].shift(LAG_HOUR_STEPS)
        d["rolling_mean_6"] = d["y"].shift(1).rolling(ROLLING_WINDOW_STEPS).mean()
        return d.dropna()

    df_feat = _build_features(df)

    if _models_are_fresh():
        print("[XGBoost] Loading saved model...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(XGB_PATH)
        print("[XGBoost] Loaded [ok]")
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        xgb_model.fit(df_feat[FEATURE_COLS], df_feat["y"])
        xgb_model.save_model(XGB_PATH)
        print("[XGBoost] Trained and saved [ok]")

    # Seed from real observed data only — df["y"] ends with add_closed_zeros synthetic
    # zeros for overnight/post-close gaps, which poison lag_1/lag_2/lag_3 and
    # rolling_mean at the opening of each day and cause near-zero predictions.
    _real_seed = df_real_r.sort_values("ds").reset_index(drop=True)
    recent_y = _real_seed["y"].tolist() if len(_real_seed) > 0 else df["y"].tolist()
    xgb_vals = []
    for ts in future_timestamps:
        ts_dt = pd.Timestamp(ts)
        row = {
            "hour":           ts_dt.hour,
            "minute_of_hour": ts_dt.minute,
            "day_of_week":    ts_dt.dayofweek,
            "is_weekend":     int(ts_dt.dayofweek >= 5),
            "lag_1":          recent_y[-1],
            "lag_2":          recent_y[-2] if len(recent_y) >= 2 else recent_y[-1],
            "lag_3":          recent_y[-3] if len(recent_y) >= 3 else recent_y[-1],
            "lag_12":         recent_y[-LAG_HOUR_STEPS] if len(recent_y) >= LAG_HOUR_STEPS else recent_y[0],
            "rolling_mean_6": np.mean(recent_y[-ROLLING_WINDOW_STEPS:]) if len(recent_y) >= ROLLING_WINDOW_STEPS else np.mean(recent_y),
        }
        pred = max(0.0, float(xgb_model.predict(pd.DataFrame([row]))[0]))
        xgb_vals.append(pred)
        recent_y.append(pred)
    xgb_vals = np.array(xgb_vals)

    # Apply a conservative first-hour opening floor so the LSTM does not stay at
    # the previous evening's low regime when Prophet/XGBoost already indicate demand.
    for i, ts in enumerate(future_timestamps):
        ts_dt = pd.Timestamp(ts)
        minute_of_day = ts_dt.hour * 60 + ts_dt.minute
        if not (SHOP_OPEN_TOT <= minute_of_day < (SHOP_OPEN_TOT + opening_span)):
            continue

        progress = min(1.0, max(0.0, (minute_of_day - SHOP_OPEN_TOT) / opening_fade_span))
        floor_share = LSTM_OPENING_FLOOR_SHARE_MAX + (
            (LSTM_OPENING_FLOOR_SHARE_MIN - LSTM_OPENING_FLOOR_SHARE_MAX) * progress
        )
        opening_hist = float(opening_profile_raw.get(minute_of_day, 0.0))
        # XGBoost floor: use Prophet as baseline (XGBoost may still be low from stale lags)
        xgb_baseline = max(opening_hist, 0.6 * float(prophet_vals[i]))
        xgb_floor = max(0.0, floor_share * xgb_baseline)
        xgb_vals[i] = max(float(xgb_vals[i]), xgb_floor)
        # LSTM floor: use the now-corrected XGBoost alongside Prophet
        model_baseline = max(float(prophet_vals[i]), float(xgb_vals[i]))
        opening_baseline = max(opening_hist, 0.6 * model_baseline)
        morning_floor = max(0.0, floor_share * opening_baseline)
        lstm_vals[i] = max(float(lstm_vals[i]), morning_floor)

    # ── ENSEMBLE ─────────────────────────────────────────────────────────────
    print("\n[Ensemble] Combining predictions (Prophet 40% / LSTM 30% / XGBoost 30%)...")
    ensemble_vals = (W_PROPHET * prophet_vals + W_LSTM * lstm_vals + W_XGB * xgb_vals).clip(min=0).round(1)

    # ── Wait estimates ────────────────────────────────────────────────────────
    print("\n[Wait] Loading current queue state from snapshots...")
    _conn_snap = psycopg2.connect(**DB_CONFIG)
    try:
        snap_latest = pd.read_sql("""
            SELECT timestamp, queue_count, active_lanes
            FROM queue_state_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """, _conn_snap)
    finally:
        _conn_snap.close()

    snap_age_min = None
    if len(snap_latest) > 0:
        snap_ts = pd.Timestamp(snap_latest.iloc[0]["timestamp"])
        if snap_ts.tzinfo is not None:
            snap_age_min = (pd.Timestamp.now(tz="UTC") - snap_ts.tz_convert("UTC")).total_seconds() / 60.0
        else:
            snap_age_min = (pd.Timestamp.now(tz="UTC").tz_localize(None) - snap_ts).total_seconds() / 60.0

    # ── Dwell model: XGBoost on service_events → per-bucket prediction ────────
    print("\n[Dwell] Loading service events for dwell model...")
    _conn_svc = psycopg2.connect(**DB_CONFIG)
    try:
        df_svc = pd.read_sql(f"""
            SELECT timestamp, total_dwell_sec / 60.0 AS service_min
            FROM service_events
            WHERE total_dwell_sec >= {DWELL_MIN_FLOOR * 60}
              AND total_dwell_sec <= {DWELL_MAX_CAP * 60}
              AND timestamp >= NOW() - INTERVAL '{data_span_days} days'
        """, _conn_svc)
    finally:
        _conn_svc.close()

    # Flat median fallback (always compute as safety net)
    svc_median = None
    if not df_svc.empty:
        svc_vals = pd.to_numeric(df_svc["service_min"], errors="coerce").dropna()
        if not svc_vals.empty:
            svc_median = float(svc_vals.median())
    avg_dwell_min = float(np.clip(svc_median, DWELL_MIN_FLOOR, DWELL_MAX_CAP)) if svc_median else float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))

    # Train / load XGBoost dwell model
    dwell_model = None
    _dwell_fresh = os.path.exists(DWELL_MODEL_PATH) and (_time.time() - os.path.getmtime(DWELL_MODEL_PATH)) < MODEL_MAX_AGE_HOURS * 3600
    if _dwell_fresh:
        print("[Dwell] Loading saved dwell model...")
        dwell_model = xgb.XGBRegressor()
        dwell_model.load_model(DWELL_MODEL_PATH)
        print("[Dwell] Loaded [ok]")
    elif not df_svc.empty and len(df_svc) >= 30:
        print(f"[Dwell] Training dwell model on {len(df_svc)} service events...")
        _ds = df_svc.copy()
        _ds["timestamp"] = pd.to_datetime(_ds["timestamp"]).dt.tz_localize(None)
        _ds["hour"]           = _ds["timestamp"].dt.hour
        _ds["minute_of_hour"] = _ds["timestamp"].dt.minute
        _ds["day_of_week"]    = _ds["timestamp"].dt.dayofweek
        _ds["is_weekend"]     = (_ds["day_of_week"] >= 5).astype(int)
        _ds["service_min"]    = _ds["service_min"].clip(DWELL_MIN_FLOOR, DWELL_MAX_CAP)
        dwell_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        dwell_model.fit(_ds[DWELL_FEATURE_COLS], _ds["service_min"])
        dwell_model.save_model(DWELL_MODEL_PATH)
        print(f"[Dwell] Trained and saved [ok]  (median={avg_dwell_min:.2f} min)")
    else:
        print(f"[Dwell] Not enough data ({len(df_svc)} events) — using flat median fallback ({avg_dwell_min:.2f} min)")

    # Per-bucket dwell predictions for future timestamps
    if dwell_model is not None:
        _dwell_rows = [{"hour": pd.Timestamp(ts).hour, "minute_of_hour": pd.Timestamp(ts).minute,
                        "day_of_week": pd.Timestamp(ts).dayofweek, "is_weekend": int(pd.Timestamp(ts).dayofweek >= 5)}
                       for ts in future_timestamps]
        _dwell_raw = dwell_model.predict(pd.DataFrame(_dwell_rows))
        per_bucket_dwell = np.clip(_dwell_raw, DWELL_MIN_FLOOR, DWELL_MAX_CAP).tolist()
        print(f"[Dwell] Per-bucket predictions — min={min(per_bucket_dwell):.2f} max={max(per_bucket_dwell):.2f} mean={np.mean(per_bucket_dwell):.2f} min")
    else:
        per_bucket_dwell = avg_dwell_min  # scalar fallback

    if len(snap_latest) > 0 and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN):
        _raw_queue    = int(snap_latest.iloc[0]["queue_count"])
        active_lanes  = max(1, int(snap_latest.iloc[0]["active_lanes"] or DEFAULT_LANES))
        current_queue = max(0, _raw_queue - active_lanes)
        print(f"[Wait] raw_queue={_raw_queue}, in_service={active_lanes}, waiting={current_queue}, service={avg_dwell_min:.1f}min/customer, lanes={active_lanes}")
    else:
        current_queue = 0
        active_lanes  = DEFAULT_LANES
        print("[Wait] No snapshot — using defaults")

    # ── Browsing-gap shift ────────────────────────────────────────────────────
    # A customer who enters the store at time T reaches checkout at T + gap.
    # So checkout arrivals at future step i = entrance arrivals from (i - lag) steps ago.
    # For the initial `lag` steps, we pull actual recent entrance counts from DB.
    BROWSING_GAP_MIN = int(os.getenv("BROWSING_GAP_MIN", 25))
    browsing_lag_steps = max(1, round(BROWSING_GAP_MIN / BUCKET_MINUTES))
    print(f"[Wait] Browsing gap = {BROWSING_GAP_MIN} min ({browsing_lag_steps} buckets) — shifting entrance → checkout...")

    _conn_hist = psycopg2.connect(**DB_CONFIG)
    try:
        df_hist = pd.read_sql(f"""
            SELECT
                time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS bucket,
                COUNT(*) AS entry_count
            FROM entrance_events
            WHERE timestamp >= NOW() - INTERVAL '{BROWSING_GAP_MIN + BUCKET_MINUTES} minutes'
            GROUP BY bucket
            ORDER BY bucket ASC
        """, _conn_hist)
    finally:
        _conn_hist.close()

    df_hist["bucket"] = pd.to_datetime(df_hist["bucket"]).dt.tz_localize(None)
    hist_by_time = dict(
        zip(df_hist["bucket"].dt.floor(f"{BUCKET_MINUTES}min"),
            df_hist["entry_count"].astype(float))
    )

    checkout_arrivals = np.zeros(n_steps)
    now_floored = pd.Timestamp.now().replace(second=0, microsecond=0,
                  minute=(pd.Timestamp.now().minute // BUCKET_MINUTES) * BUCKET_MINUTES)
    for i in range(n_steps):
        entrance_idx = i - browsing_lag_steps
        if entrance_idx < 0:
            entrance_time = now_floored + pd.Timedelta(minutes=BUCKET_MINUTES * entrance_idx)
            checkout_arrivals[i] = hist_by_time.get(entrance_time, 0.0)
        else:
            checkout_arrivals[i] = ensemble_vals[entrance_idx]

    wait_frame = prophet_preds[["ds"]].copy()
    wait_frame["yhat"] = checkout_arrivals

    wait_rows, wait_15m, wait_30m, service_per_bucket = compute_wait_estimates(
        wait_frame,
        current_queue=current_queue,
        avg_dwell_min=per_bucket_dwell,
        active_lanes=active_lanes,
        max_queue_per_lane=MAX_QUEUE_PER_LANE,
        max_wait_min=MAX_WAIT_MIN,
    )
    wait_estimates = [float(r["wait_min"]) for r in wait_rows]

    # 45-minute horizon
    wait_45m = float(wait_rows[WAIT_45M_INDEX]["wait_min"]) if len(wait_rows) > WAIT_45M_INDEX else wait_30m

    # Lane scenario comparisons — wait at +15 min for 1, 2, 3 lanes
    lane_waits_15m: dict[int, float] = {}
    for n_lanes in [1, 2, 3]:
        _rows, _w15, _, _ = compute_wait_estimates(
            wait_frame,
            current_queue=current_queue,
            avg_dwell_min=per_bucket_dwell,
            active_lanes=n_lanes,
            max_queue_per_lane=MAX_QUEUE_PER_LANE,
            max_wait_min=MAX_WAIT_MIN,
        )
        lane_waits_15m[n_lanes] = float(_w15) if _w15 is not None else 0.0
    print(f"[Wait] Lane scenarios @ +15m — 1 lane: {lane_waits_15m[1]:.1f} min | "
          f"2 lanes: {lane_waits_15m[2]:.1f} min | 3 lanes: {lane_waits_15m[3]:.1f} min")

    return {
        "source":             source,
        "prophet_preds":      prophet_preds,
        "prophet_vals":       prophet_vals,
        "lstm_vals":          lstm_vals,
        "xgb_vals":           xgb_vals,
        "ensemble_vals":      ensemble_vals,
        "checkout_arrivals":  checkout_arrivals,
        "wait_estimates":     wait_estimates,
        "wait_15m":           wait_15m,
        "wait_30m":           wait_30m,
        "wait_45m":           wait_45m,
        "lane_waits_15m":     lane_waits_15m,
        "current_queue":      current_queue,
        "avg_dwell_min":      avg_dwell_min,
        "per_bucket_dwell":   per_bucket_dwell,
        "active_lanes":       active_lanes,
        "service_per_bucket": service_per_bucket,
        "real_span_days":     real_span_days,
        "browsing_gap_min":   BROWSING_GAP_MIN,
    }


def _print_results(result: dict) -> None:
    prophet_preds      = result["prophet_preds"]
    prophet_vals       = result["prophet_vals"]
    lstm_vals          = result["lstm_vals"]
    xgb_vals           = result["xgb_vals"]
    ensemble_vals      = result["ensemble_vals"]
    checkout_arrivals  = result["checkout_arrivals"]
    wait_estimates     = result["wait_estimates"]
    service_per_bucket = result["service_per_bucket"]
    active_lanes       = result["active_lanes"]
    browsing_gap_min   = result["browsing_gap_min"]

    print("\n" + "=" * 85)
    print("  ENSEMBLE PREDICTED CUSTOMER ENTRIES — NEXT 60 MINUTES")
    print("=" * 85)
    print(f"  {'Time':<10} {'Prophet':>10} {'LSTM':>10} {'XGBoost':>10} {'Ensemble':>10} {'→Checkout':>12}")
    print("-" * 85)
    for i, row in prophet_preds.iterrows():
        print(
            f"  {row['ds'].strftime('%H:%M'):<10}"
            f" {prophet_vals[i]:>10.1f}"
            f" {lstm_vals[i]:>10.1f}"
            f" {xgb_vals[i]:>10.1f}"
            f" {ensemble_vals[i]:>10.1f}"
            f" {checkout_arrivals[i]:>12.1f}"
        )
    print("=" * 85)
    print(f"  (→Checkout = entrance shifted {browsing_gap_min} min forward for browsing gap)")

    print("\n" + "=" * 75)
    print("  QUEUE WAIT TIME ESTIMATES  (checkout arrivals + queue model)")
    print(f"  Service rate: {service_per_bucket:.1f} customers/{BUCKET_MINUTES}min  |  Lanes: {active_lanes}")
    print("=" * 75)
    print(f"  {'Time':<10} {'Checkout':>10} {'Est. Wait':>12}   Status")
    print("-" * 75)
    for i, row in prophet_preds.iterrows():
        wait   = wait_estimates[i]
        status = "OK" if wait < 5 else "BUSY" if wait < 10 else "ALERT"
        print(f"  {row['ds'].strftime('%H:%M'):<10} {checkout_arrivals[i]:>10.1f} {wait:>8} min   {status}")
    print("=" * 75)
    print(f"\n  Wait @ +15 min: {result['wait_15m']} min  |  Wait @ +30 min: {result['wait_30m']} min")


def _save_to_db(result: dict) -> None:
    prophet_preds  = result["prophet_preds"]
    prophet_vals   = result["prophet_vals"]
    lstm_vals      = result["lstm_vals"]
    xgb_vals       = result["xgb_vals"]
    ensemble_vals  = result["ensemble_vals"]
    wait_estimates = result["wait_estimates"]

    print("\n[DB] Saving predictions to queue_predictions...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queue_predictions (
            id                BIGSERIAL,
            predicted_at      TIMESTAMPTZ NOT NULL,
            prediction_for    TIMESTAMPTZ NOT NULL,
            prophet_yhat      NUMERIC(8,2),
            lstm_yhat         NUMERIC(8,2),
            xgb_yhat          NUMERIC(8,2),
            ensemble_yhat     NUMERIC(8,2),
            est_wait_minutes  NUMERIC(8,2),
            wait_15m          NUMERIC(8,2),
            wait_30m          NUMERIC(8,2),
            status            VARCHAR(10)
        )
    """)
    cur.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'queue_predictions'
                AND column_name = 'predicted_at'
                AND data_type = 'timestamp without time zone'
            ) THEN
                ALTER TABLE queue_predictions
                    ALTER COLUMN predicted_at   TYPE TIMESTAMPTZ USING predicted_at   AT TIME ZONE 'UTC',
                    ALTER COLUMN prediction_for TYPE TIMESTAMPTZ USING prediction_for AT TIME ZONE 'UTC';
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'queue_predictions' AND column_name = 'wait_15m'
            ) THEN
                ALTER TABLE queue_predictions
                    ADD COLUMN wait_15m NUMERIC(8,2),
                    ADD COLUMN wait_30m NUMERIC(8,2);
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'queue_predictions' AND column_name = 'wait_45m'
            ) THEN
                ALTER TABLE queue_predictions
                    ADD COLUMN wait_45m        NUMERIC(8,2),
                    ADD COLUMN wait_1lane_15m  NUMERIC(8,2),
                    ADD COLUMN wait_2lane_15m  NUMERIC(8,2),
                    ADD COLUMN wait_3lane_15m  NUMERIC(8,2);
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'queue_predictions' AND column_name = 'source'
            ) THEN
                ALTER TABLE queue_predictions
                    ADD COLUMN source VARCHAR(20) DEFAULT 'ensemble';
            END IF;
        END $$;
    """)
    cur.execute("""
        SELECT create_hypertable('queue_predictions', 'prediction_for',
            if_not_exists => TRUE, migrate_data => TRUE);
    """)
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'queue_predictions_prediction_for_key'
            ) THEN
                ALTER TABLE queue_predictions
                    ADD CONSTRAINT queue_predictions_prediction_for_key
                    UNIQUE (prediction_for);
            END IF;
        END $$;
    """)

    def _db_val(v):
        """Convert NaN/inf to None so PostgreSQL stores NULL instead of raising."""
        try:
            f = float(v)
            return None if (f != f or f == float("inf") or f == float("-inf")) else round(f, 2)
        except (TypeError, ValueError):
            return None

    predicted_at  = datetime.now()
    lw            = result["lane_waits_15m"]
    _lw1          = _db_val(lw.get(1, 0.0))
    _lw2          = _db_val(lw.get(2, 0.0))
    _lw3          = _db_val(lw.get(3, 0.0))

    _steps_15 = 15 // BUCKET_MINUTES
    _steps_30 = 30 // BUCKET_MINUTES
    _steps_45 = 45 // BUCKET_MINUTES
    _we: list[float] = [float(v) for v in wait_estimates]  # type: ignore[union-attr]
    _n  = len(_we)

    for i, row in prophet_preds.iterrows():
        wait   = round(_we[i], 2)
        status = "OK" if wait < 5 else "BUSY" if wait < 10 else "ALERT"
        _w15 = round(_we[i + _steps_15], 2) if i + _steps_15 < _n else None
        _w30 = round(_we[i + _steps_30], 2) if i + _steps_30 < _n else None
        _w45 = round(_we[i + _steps_45], 2) if i + _steps_45 < _n else None
        cur.execute("""
            INSERT INTO queue_predictions
                (predicted_at, prediction_for, prophet_yhat, lstm_yhat,
                 xgb_yhat, ensemble_yhat, est_wait_minutes,
                 wait_15m, wait_30m, wait_45m,
                 wait_1lane_15m, wait_2lane_15m, wait_3lane_15m, status, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (prediction_for) DO UPDATE SET
                predicted_at    = EXCLUDED.predicted_at,
                prophet_yhat    = EXCLUDED.prophet_yhat,
                lstm_yhat       = EXCLUDED.lstm_yhat,
                xgb_yhat        = EXCLUDED.xgb_yhat,
                ensemble_yhat   = EXCLUDED.ensemble_yhat,
                est_wait_minutes= EXCLUDED.est_wait_minutes,
                wait_15m        = EXCLUDED.wait_15m,
                wait_30m        = EXCLUDED.wait_30m,
                wait_45m        = EXCLUDED.wait_45m,
                wait_1lane_15m  = EXCLUDED.wait_1lane_15m,
                wait_2lane_15m  = EXCLUDED.wait_2lane_15m,
                wait_3lane_15m  = EXCLUDED.wait_3lane_15m,
                status          = EXCLUDED.status,
                source          = EXCLUDED.source
        """, (
            predicted_at,
            row["ds"].to_pydatetime(),
            _db_val(prophet_vals[i]),
            _db_val(lstm_vals[i]),
            _db_val(xgb_vals[i]),
            _db_val(ensemble_vals[i]),
            wait,
            _w15, _w30, _w45,
            _lw1, _lw2, _lw3,
            status,
            'ensemble',
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Saved {FORECAST_STEPS} rows to queue_predictions [ok]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="REAL", choices=["REAL", "SIM", "ALL"])
    parser.add_argument("--bootstrap", action="store_true",
                        help="Pad sparse real data with synthetic history when < 14 days available.")
    parser.add_argument("--days", type=int, default=DATA_SPAN_DAYS_DEFAULT,
                        help="Number of past days of entrance_events to use for training (default: DATA_SPAN_DAYS env var or 30).")
    args = parser.parse_args()

    result = run_ensemble_forecast(source=args.source, bootstrap=args.bootstrap, data_span_days=args.days)
    _print_results(result)
    _save_to_db(result)
    print("\n[Ensemble] All done.")
