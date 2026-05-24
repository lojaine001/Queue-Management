from __future__ import annotations

import os
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
    FORECAST_STEPS,
    LAG_HOUR_STEPS,
    MAX_QUEUE_PER_LANE,
    MAX_WAIT_MIN,
    MIN_REAL_DAYS_FOR_SIM,
    ROLLING_WINDOW_STEPS,
    SEQUENCE_LEN,
    SNAPSHOT_MAX_AGE_MIN,
    WAIT_45M_INDEX,
    add_closed_zeros,
    build_sim_history,
    compute_wait_estimates,
    get_where_clause,
)
from prediction.pipeline import DB_CONFIG  # noqa: E402

W_PROPHET           = float(os.getenv("W_PROPHET",          0.40))
W_LSTM              = float(os.getenv("W_LSTM",             0.30))
W_XGB               = float(os.getenv("W_XGB",             0.30))
MODEL_MAX_AGE_HOURS = int(os.getenv("MODEL_MAX_AGE_HOURS",  24))
MODELS_DIR     = 'models'
LSTM_PATH      = os.path.join(MODELS_DIR, f'lstm_queue_{BUCKET_MINUTES}m.keras')
SCALER_PATH    = os.path.join(MODELS_DIR, f'lstm_scaler_{BUCKET_MINUTES}m.pkl')
XGB_PATH       = os.path.join(MODELS_DIR, f'xgb_queue_{BUCKET_MINUTES}m.json')
PROPHET_PATH   = os.path.join(MODELS_DIR, f'prophet_queue_{BUCKET_MINUTES}m.pkl')
os.makedirs(MODELS_DIR, exist_ok=True)


def _models_are_fresh() -> bool:
    paths = [LSTM_PATH, SCALER_PATH, XGB_PATH, PROPHET_PATH]
    if not all(os.path.exists(p) for p in paths):
        return False
    oldest = min(os.path.getmtime(p) for p in paths)
    return (_time.time() - oldest) < MODEL_MAX_AGE_HOURS * 3600


def run_ensemble_forecast(source: str = "REAL", bootstrap: bool = False) -> dict:
    """Train Prophet + LSTM + XGBoost ensemble and return a forecast dict."""
    where_clause = get_where_clause(source)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"[Ensemble] Connected to PostgreSQL (Source: {source}) ✓")

    # Clean data only: Apr 13-15 (before overcounting) + Apr 21 onwards (current pipeline)
    clean_filter = """(
        (timestamp >= '2026-04-13' AND timestamp < '2026-04-16')
        OR timestamp >= '2026-04-21'
    ) AND dwell_seconds >= 10"""
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

    # ── 4. Closed-hour zeros ─────────────────────────────────────────────────
    df = add_closed_zeros(df[["ds", "y"]].copy(), days=30)

    print(f"[Ensemble] DataFrame shape: {df.shape}")

    now = pd.Timestamp.now()
    future_timestamps = [
        now + pd.Timedelta(minutes=BUCKET_MINUTES * (i + 1)) for i in range(FORECAST_STEPS)
    ]
    future_df = pd.DataFrame({"ds": future_timestamps})

    # ── MODEL 1 — PROPHET ────────────────────────────────────────────────────
    if _models_are_fresh():
        print("\n[Prophet] Loading saved model...")
        with open(PROPHET_PATH, "rb") as _f:
            prophet_model = pickle.load(_f)
        print("[Prophet] Loaded ✓")
    else:
        print("\n[Prophet] Training...")
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.80,
        )
        prophet_model.fit(df)
        with open(PROPHET_PATH, "wb") as _f:
            pickle.dump(prophet_model, _f)
        print("[Prophet] Trained and saved ✓")
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

    if _models_are_fresh():
        print("[LSTM] Loading saved model and scaler...")
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
        with open(SCALER_PATH, "rb") as _f:
            scaler = pickle.load(_f)
        y_scaled = scaler.transform(y_values.reshape(-1, 1))
        print("[LSTM] Loaded ✓")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(df_real_r) > 0:
            real_y = pd.to_numeric(df_real_r["y"], errors="coerce").fillna(0).values
            scaler.fit(real_y.reshape(-1, 1))
            y_scaled = scaler.transform(y_values.reshape(-1, 1))
        else:
            y_scaled = scaler.fit_transform(y_values.reshape(-1, 1))

        X_lstm, y_lstm = _make_sequences(y_scaled, SEQUENCE_LEN)
        print(f"[LSTM] Training on {len(X_lstm)} sequences (epochs=20)...")
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
        print("[LSTM] Trained and saved ✓")

    last_seq = y_scaled[-SEQUENCE_LEN:].copy()
    lstm_scaled_preds = []
    for _ in range(FORECAST_STEPS):
        inp      = last_seq.reshape(1, SEQUENCE_LEN, 1)
        next_val = lstm_model.predict(inp, verbose=0)[0][0]
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
        print("[XGBoost] Loaded ✓")
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        xgb_model.fit(df_feat[FEATURE_COLS], df_feat["y"])
        xgb_model.save_model(XGB_PATH)
        print("[XGBoost] Trained and saved ✓")

    recent_y = df["y"].tolist()
    xgb_vals = []
    for ts in future_timestamps:
        ts_dt = pd.Timestamp(ts)
        row = {
            "hour":           ts_dt.hour,
            "minute_of_hour": ts_dt.minute,
            "day_of_week":    ts_dt.dayofweek,
            "is_weekend":     int(ts_dt.dayofweek >= 5),
            "lag_1":          recent_y[-1],
            "lag_2":          recent_y[-2],
            "lag_3":          recent_y[-3],
            "lag_12":         recent_y[-LAG_HOUR_STEPS] if len(recent_y) >= LAG_HOUR_STEPS else recent_y[0],
            "rolling_mean_6": np.mean(recent_y[-ROLLING_WINDOW_STEPS:]) if len(recent_y) >= ROLLING_WINDOW_STEPS else np.mean(recent_y),
        }
        pred = max(0.0, float(xgb_model.predict(pd.DataFrame([row]))[0]))
        xgb_vals.append(pred)
        recent_y.append(pred)
    xgb_vals = np.array(xgb_vals)

    # ── ENSEMBLE ─────────────────────────────────────────────────────────────
    print("\n[Ensemble] Combining predictions (Prophet 40% / LSTM 30% / XGBoost 30%)...")
    ensemble_vals = (W_PROPHET * prophet_vals + W_LSTM * lstm_vals + W_XGB * xgb_vals).clip(min=0).round(1)

    # ── Wait estimates ────────────────────────────────────────────────────────
    print("\n[Wait] Loading current queue state from snapshots...")
    _conn_snap = psycopg2.connect(**DB_CONFIG)
    try:
        # Average the last 5 minutes of snapshots to smooth out per-frame noise.
        # Use only the most recent row for queue_count (current depth) and
        # active_lanes (current config), but average dwell over the window.
        snap_latest = pd.read_sql("""
            SELECT timestamp, queue_count, avg_dwell_sec, active_lanes
            FROM queue_state_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """, _conn_snap)
        snap_avg = pd.read_sql("""
            SELECT AVG(avg_dwell_sec) AS avg_dwell_sec
            FROM queue_state_snapshots
            WHERE timestamp >= NOW() - INTERVAL '5 minutes'
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

    if len(snap_latest) > 0 and (snap_age_min is None or snap_age_min < SNAPSHOT_MAX_AGE_MIN):
        current_queue = int(snap_latest.iloc[0]["queue_count"])
        raw_dwell_sec = (
            float(snap_avg.iloc[0]["avg_dwell_sec"])
            if len(snap_avg) > 0 and snap_avg.iloc[0]["avg_dwell_sec"] is not None
            else float(snap_latest.iloc[0]["avg_dwell_sec"])
        )
        raw_dwell_min = raw_dwell_sec / 60.0
        # Snapshot dwell reflects entrance-zone tracking time (~6s), not checkout service time.
        # Only use it if it looks like a real service time; otherwise use the configured default.
        checkout_service_min = float(os.getenv("CHECKOUT_SERVICE_MIN", DEFAULT_DWELL_MIN))
        avg_dwell_min = raw_dwell_min if raw_dwell_min >= 1.0 else checkout_service_min
        active_lanes  = max(1, int(snap_latest.iloc[0]["active_lanes"] or DEFAULT_LANES))
        print(f"[Wait] queue={current_queue}, service={avg_dwell_min:.1f}min/customer, lanes={active_lanes}")
    else:
        current_queue = 0
        avg_dwell_min = DEFAULT_DWELL_MIN
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

    checkout_arrivals = np.zeros(FORECAST_STEPS)
    now_floored = pd.Timestamp.now().floor(f"{BUCKET_MINUTES}min")
    for i in range(FORECAST_STEPS):
        entrance_idx = i - browsing_lag_steps
        if entrance_idx < 0:
            entrance_time = now_floored + pd.Timedelta(minutes=BUCKET_MINUTES * entrance_idx)
            checkout_arrivals[i] = hist_by_time.get(entrance_time, 0.0)
        else:
            checkout_arrivals[i] = ensemble_vals[entrance_idx]

    wait_frame = prophet_preds[["ds"]].copy()
    wait_frame["yhat"] = checkout_arrivals
    est_service_min = min(max(avg_dwell_min, 0.5), 10.0)
    wait_rows, wait_15m, wait_30m, service_per_bucket = compute_wait_estimates(
        wait_frame,
        current_queue=current_queue,
        avg_dwell_min=est_service_min,
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
            avg_dwell_min=est_service_min,
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
        SELECT create_hypertable('queue_predictions', 'prediction_for',
            if_not_exists => TRUE, migrate_data => TRUE);
    """)

    predicted_at  = datetime.now()
    lw            = result["lane_waits_15m"]
    _w45          = round(result["wait_45m"], 2)  if result["wait_45m"]  is not None else None
    _w15          = round(result["wait_15m"], 2)  if result["wait_15m"]  is not None else None
    _w30          = round(result["wait_30m"], 2)  if result["wait_30m"]  is not None else None
    _lw1          = round(lw.get(1, 0.0), 2)
    _lw2          = round(lw.get(2, 0.0), 2)
    _lw3          = round(lw.get(3, 0.0), 2)

    for i, row in prophet_preds.iterrows():
        wait   = round(wait_estimates[i], 2)
        status = "OK" if wait < 5 else "BUSY" if wait < 10 else "ALERT"
        cur.execute("""
            INSERT INTO queue_predictions
                (predicted_at, prediction_for, prophet_yhat, lstm_yhat,
                 xgb_yhat, ensemble_yhat, est_wait_minutes,
                 wait_15m, wait_30m, wait_45m,
                 wait_1lane_15m, wait_2lane_15m, wait_3lane_15m, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            predicted_at,
            row["ds"].to_pydatetime(),
            round(float(prophet_vals[i]),  2),
            round(float(lstm_vals[i]),     2),
            round(float(xgb_vals[i]),      2),
            round(float(ensemble_vals[i]), 2),
            wait,
            _w15, _w30, _w45,
            _lw1, _lw2, _lw3,
            status,
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Saved {FORECAST_STEPS} rows to queue_predictions ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="REAL", choices=["REAL", "SIM", "ALL"])
    parser.add_argument("--bootstrap", action="store_true",
                        help="Pad sparse real data with synthetic history when < 14 days available.")
    args = parser.parse_args()

    result = run_ensemble_forecast(source=args.source, bootstrap=args.bootstrap)
    _print_results(result)
    _save_to_db(result)
    print("\n[Ensemble] All done.")
