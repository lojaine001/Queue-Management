"""
backtest_predict.py

Run the currently trained ensemble models retroactively over historical
entrance-event data to fill gaps in queue_predictions.

Each past open-hour bucket gets:
  - Prophet  prediction  (direct model.predict on historical timestamps)
  - XGBoost  prediction  (features built from actual historical lag values)
  - LSTM     prediction  (actual historical sliding window as seed — batch)
  - Ensemble             (weighted average of the three)

Existing rows in queue_predictions are left untouched (ON CONFLICT DO NOTHING).

Usage:
    python backtest_predict.py --days 30
    python backtest_predict.py --days 7 --source REAL --overwrite
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import tensorflow as tf
import warnings
import xgboost as xgb
from prophet import Prophet  # noqa: F401  (needed for pickle.load)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

from prediction.core import (  # noqa: E402
    BUCKET_MINUTES,
    LAG_HOUR_STEPS,
    ROLLING_WINDOW_STEPS,
    SEQUENCE_LEN,
    SHOP_OPEN_TOT,
    add_closed_zeros,
    get_where_clause,
    is_open,
)
from prediction.pipeline import DB_CONFIG, to_local  # noqa: E402

W_PROPHET = float(os.getenv("W_PROPHET", 0.40))
W_LSTM    = float(os.getenv("W_LSTM",    0.30))
W_XGB     = float(os.getenv("W_XGB",     0.30))
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


LSTM_PATH    = _resolve_model_path(f"lstm_queue_{BUCKET_MINUTES}m.keras")
SCALER_PATH  = _resolve_model_path(f"lstm_scaler_{BUCKET_MINUTES}m.pkl")
XGB_PATH     = _resolve_model_path(f"xgb_queue_{BUCKET_MINUTES}m.json")
PROPHET_PATH = _resolve_model_path(f"prophet_queue_{BUCKET_MINUTES}m.pkl")

FEATURE_COLS = [
    "hour", "minute_of_hour", "day_of_week", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_12", "rolling_mean_6",
]


def _build_lstm_opening_profile_raw(history_df: pd.DataFrame) -> dict[int, float]:
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


def _add_closed_zeros_over_range(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_min: int = BUCKET_MINUTES,
) -> pd.DataFrame:
    """Fill closed-hour buckets only over the actual historical data range."""
    start = pd.Timestamp(start).floor(f"{interval_min}min")
    end = pd.Timestamp(end).ceil(f"{interval_min}min")
    rows = []
    t = start
    while t <= end:
        if not is_open(pd.Timestamp(t)):
            rows.append({"ds": t, "y": 0})
        t += pd.Timedelta(minutes=interval_min)
    if not rows:
        return df.sort_values("ds").reset_index(drop=True)
    zeros = pd.DataFrame(rows)
    return (
        pd.concat([df, zeros], ignore_index=True)
        .drop_duplicates(subset="ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _db_val(v):
    try:
        f = float(v)
        return None if (f != f or abs(f) == float("inf")) else round(f, 2)
    except (TypeError, ValueError):
        return None


def _check_models():
    missing = [
        name for path, name in [
            (LSTM_PATH, "LSTM"), (SCALER_PATH, "Scaler"),
            (XGB_PATH, "XGBoost"),
            # Prophet excluded: backtest writes prophet_yhat=NULL (leakage risk)
        ]
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing trained models: {', '.join(missing)}. "
            "Run ensemble_predict.py first to train the models."
        )


def run_backtest(days: int = 30, source: str = "REAL", overwrite: bool = False) -> int:
    """
    Fill queue_predictions with historical backtest predictions.

    Parameters
    ----------
    days      : int   How many past days to cover.
    source    : str   Data source filter ("REAL" / "SIM" / "ALL").
    overwrite : bool  If True, update existing rows; otherwise skip them.

    Returns
    -------
    int  Number of rows inserted (or updated).
    """
    _check_models()

    # ── 1. Load trained models ────────────────────────────────────────────────
    print("[Backtest] Loading trained models...")
    lstm_model = tf.keras.models.load_model(LSTM_PATH)
    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_PATH)
    print("[Backtest] Models loaded ✓")

    # ── 2. Load historical entrance data ──────────────────────────────────────
    where = get_where_clause(source)
    clean = f"timestamp >= NOW() - INTERVAL '{days} days' AND dwell_seconds >= 10"
    data_filter = f"{where} AND {clean}" if where else f"WHERE {clean}"

    conn = psycopg2.connect(**DB_CONFIG)
    df_raw = pd.read_sql(
        f"""
        SELECT
            time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS ds,
            COUNT(*) AS y
        FROM entrance_events
        {data_filter}
        GROUP BY ds
        ORDER BY ds
        """,
        conn,
    )
    conn.close()

    df_raw["ds"] = to_local(df_raw["ds"])
    df_raw["y"]  = pd.to_numeric(df_raw["y"], errors="coerce").fillna(0)
    print(f"[Backtest] {len(df_raw)} real buckets loaded ({days} days back)")

    if df_raw.empty:
        print("[Backtest] No historical data. Aborting.")
        return 0

    # Apply 15-min rolling median smooth to open-hour data only (before adding closed-hour zeros)
    df_raw["y"] = (
        df_raw["y"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .round(2)
    )
    opening_profile_raw = _build_lstm_opening_profile_raw(df_raw[["ds", "y"]])

    # Build complete time-series over the actual historical data range.
    hist_start = df_raw["ds"].min()
    hist_end = df_raw["ds"].max()
    df_full = _add_closed_zeros_over_range(df_raw[["ds", "y"]].copy(), start=hist_start, end=hist_end)
    df_full = df_full.sort_values("ds").reset_index(drop=True)

    # Slots to predict: only open-hour buckets in the past
    mask_open = df_full["ds"].apply(lambda t: is_open(pd.Timestamp(t)))
    past_slots = df_full[mask_open].reset_index(drop=True)
    n = len(past_slots)
    print(f"[Backtest] {n} open-hour slots to predict")

    # ── 3. LSTM (batch with actual sliding window) ────────────────────────────
    y_scaled   = scaler.transform(df_full["y"].values.reshape(-1, 1))
    ds_to_idx  = {row["ds"]: i for i, row in df_full.iterrows()}

    sequences   = []
    seq_indices = []
    for i, row in past_slots.iterrows():
        idx = ds_to_idx.get(row["ds"])
        if idx is not None and idx >= SEQUENCE_LEN:
            sequences.append(y_scaled[idx - SEQUENCE_LEN:idx])
            seq_indices.append(i)

    lstm_vals = np.zeros(n)
    if sequences:
        X_batch      = np.array(sequences)                          # (k, SEQ_LEN, 1)
        preds_scaled = lstm_model.predict(X_batch, verbose=0).flatten()
        preds_real   = scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).flatten().clip(min=0)
        for arr_i, slot_i in enumerate(seq_indices):
            lstm_vals[slot_i] = float(preds_real[arr_i])
    print("[Backtest] LSTM predictions done ✓")

    # ── 5. XGBoost (features from actual historical lags) ─────────────────────
    df_feat = df_full.copy()
    df_feat["hour"]           = df_feat["ds"].dt.hour
    df_feat["minute_of_hour"] = df_feat["ds"].dt.minute
    df_feat["day_of_week"]    = df_feat["ds"].dt.dayofweek
    df_feat["is_weekend"]     = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["lag_1"]          = df_feat["y"].shift(1)
    df_feat["lag_2"]          = df_feat["y"].shift(2)
    df_feat["lag_3"]          = df_feat["y"].shift(3)
    df_feat["lag_12"]         = df_feat["y"].shift(LAG_HOUR_STEPS)
    df_feat["rolling_mean_6"] = df_feat["y"].shift(1).rolling(ROLLING_WINDOW_STEPS).mean()

    merged     = past_slots.merge(df_feat[["ds"] + FEATURE_COLS], on="ds", how="left")
    valid_mask = merged[FEATURE_COLS].notna().all(axis=1)

    xgb_vals = np.zeros(n)
    if valid_mask.any():
        xgb_preds = xgb_model.predict(
            merged.loc[valid_mask, FEATURE_COLS]
        ).clip(min=0)
        xgb_vals[valid_mask.values] = xgb_preds
    print("[Backtest] XGBoost predictions done ✓")

    opening_span = max(BUCKET_MINUTES, LSTM_OPENING_BLEND_MINUTES)
    opening_fade_span = max(1, opening_span - BUCKET_MINUTES)
    for i, ts in enumerate(past_slots["ds"].values):
        ts_dt = pd.Timestamp(ts)
        minute_of_day = ts_dt.hour * 60 + ts_dt.minute
        if not (SHOP_OPEN_TOT <= minute_of_day < (SHOP_OPEN_TOT + opening_span)):
            continue

        progress = min(1.0, max(0.0, (minute_of_day - SHOP_OPEN_TOT) / opening_fade_span))
        floor_share = LSTM_OPENING_FLOOR_SHARE_MAX + (
            (LSTM_OPENING_FLOOR_SHARE_MIN - LSTM_OPENING_FLOOR_SHARE_MAX) * progress
        )
        model_baseline = float(xgb_vals[i])
        opening_hist = float(opening_profile_raw.get(minute_of_day, 0.0))
        opening_baseline = max(opening_hist, 0.6 * model_baseline)
        morning_floor = max(0.0, floor_share * opening_baseline)
        lstm_vals[i] = max(float(lstm_vals[i]), morning_floor)

    # ── 6. Ensemble (Prophet excluded — leakage risk; renormalize LSTM+XGBoost) ──
    _w_total = W_LSTM + W_XGB
    ensemble_vals = (
        (W_LSTM / _w_total) * lstm_vals + (W_XGB / _w_total) * xgb_vals
    ).clip(min=0).round(1)

    # ── 7. Save to DB ─────────────────────────────────────────────────────────
    conflict_action = (
        """DO UPDATE SET
            predicted_at  = EXCLUDED.predicted_at,
            prophet_yhat  = EXCLUDED.prophet_yhat,
            lstm_yhat     = EXCLUDED.lstm_yhat,
            xgb_yhat      = EXCLUDED.xgb_yhat,
            ensemble_yhat = EXCLUDED.ensemble_yhat"""
        if overwrite
        else "DO NOTHING"
    )

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
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
    conn.commit()
    predicted_at = datetime.now()
    saved = 0

    for i, row in past_slots.iterrows():
        cur.execute(
            f"""
            INSERT INTO queue_predictions
                (predicted_at, prediction_for,
                 prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat,
                 est_wait_minutes, status, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (prediction_for) {conflict_action}
            """,
            (
                predicted_at,
                row["ds"].to_pydatetime(),
                None,                        # prophet_yhat — NULL (leakage risk)
                _db_val(lstm_vals[i]),
                _db_val(xgb_vals[i]),
                _db_val(ensemble_vals[i]),
                None,
                None,
                'backtest',
            ),
        )
        saved += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    action_word = "updated" if overwrite else "inserted"
    print(f"[Backtest] {saved}/{n} rows {action_word} in queue_predictions ✓")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill queue_predictions with historical model predictions.")
    parser.add_argument("--days",      type=int,  default=7,    help="Days back to cover (default: 7)")
    parser.add_argument("--source",    type=str,  default="REAL", choices=["REAL", "SIM", "ALL"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing rows (default: skip existing)")
    args = parser.parse_args()

    print(f"\n[Backtest] Starting — {args.days} days back, source={args.source}, overwrite={args.overwrite}")
    n_saved = run_backtest(days=args.days, source=args.source, overwrite=args.overwrite)
    print(f"[Backtest] All done — {n_saved} rows saved.\n")
