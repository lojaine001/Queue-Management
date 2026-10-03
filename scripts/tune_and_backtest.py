import sys
from pathlib import Path

app_dir = Path("Queue-Management/Queue-Management-System-v2-main/Queue-Management-System-v2-main").resolve()
root_dir = Path("Queue-Management").resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(app_dir))

import psycopg2
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import xgboost as xgb
from prediction.pipeline import DB_CONFIG, to_local
from prediction.core import (
    BUCKET_MINUTES, LAG_HOUR_STEPS, ROLLING_WINDOW_STEPS, SEQUENCE_LEN,
    SHOP_OPEN_TOT, is_open
)
from prediction.arrival_calibration import target_definition
from prediction.model_bundle import current_bundle

models_dir = app_dir / "models"
active_bundle = current_bundle(models_dir)

lstm_model = tf.keras.models.load_model(active_bundle / f"lstm_queue_{BUCKET_MINUTES}m.keras")
with open(active_bundle / f"lstm_scaler_{BUCKET_MINUTES}m.pkl", "rb") as fh:
    scaler = pickle.load(fh)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(active_bundle / f"xgb_queue_{BUCKET_MINUTES}m.json")
with open(active_bundle / f"prophet_queue_{BUCKET_MINUTES}m.pkl", "rb") as fh:
    prophet_model = pickle.load(fh)

conn = psycopg2.connect(**DB_CONFIG)
cam_id = target_definition()['camera_id']

# 1. Load actuals for today
df_ground = pd.read_sql(f"""
    SELECT
        date_trunc('minute', timestamp AT TIME ZONE 'Europe/Paris') - 
        (EXTRACT(minute FROM timestamp AT TIME ZONE 'Europe/Paris')::integer % 3) * interval '1 minute' AS slot_local,
        COUNT(CASE WHEN camera_id = '{cam_id}' THEN 1 END) AS actual_target_cam,
        COUNT(*) AS actual_all_cams
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 10:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
    GROUP BY slot_local
    ORDER BY slot_local
""", conn)

# 2. Load historical raw entrance data for backtest simulation (past 7 days)
df_hist = pd.read_sql(f"""
    SELECT
        time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS ds,
        COUNT(*) AS y
    FROM entrance_events
    WHERE camera_id = '{cam_id}'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-20 00:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
    GROUP BY ds
    ORDER BY ds
""", conn)
conn.close()

df_hist["ds"] = to_local(df_hist["ds"])
df_hist["y"] = pd.to_numeric(df_hist["y"], errors="coerce").fillna(0)

# Build closed zero timeline
def add_closed_zeros(df):
    start = df["ds"].min().floor(f"{BUCKET_MINUTES}min")
    end = df["ds"].max().ceil(f"{BUCKET_MINUTES}min")
    rows = []
    t = start
    while t <= end:
        if not is_open(pd.Timestamp(t)):
            rows.append({"ds": t, "y": 0.0})
        t += pd.Timedelta(minutes=BUCKET_MINUTES)
    zeros = pd.DataFrame(rows)
    return pd.concat([df, zeros], ignore_index=True).drop_duplicates("ds").sort_values("ds").reset_index(drop=True)

FEATURE_COLS = ["hour", "minute_of_hour", "day_of_week", "is_weekend", "lag_1", "lag_2", "lag_3", "lag_12", "rolling_mean_6"]

def run_sim_backtest(smooth_win=5, burst_thresh=40, use_burst_bypass=True, outlier_cap=50):
    df_cur = df_hist.copy()
    if outlier_cap and outlier_cap > 0:
        df_cur["y"] = df_cur["y"].clip(upper=outlier_cap)
    
    if smooth_win > 1:
        smoothed = df_cur["y"].rolling(window=smooth_win, center=True, min_periods=1).median()
        if use_burst_bypass and burst_thresh:
            burst_mask = df_cur["y"] >= burst_thresh
            df_cur["y"] = np.where(burst_mask, df_cur["y"], smoothed)
        else:
            df_cur["y"] = smoothed
    
    df_full = add_closed_zeros(df_cur)
    
    # Target slots: today 10:00 to 13:00
    mask_today = (df_full["ds"] >= "2026-09-27 10:00:00") & (df_full["ds"] <= "2026-09-27 13:00:00") & df_full["ds"].map(is_open)
    target_slots = df_full[mask_today].reset_index(drop=True)
    n = len(target_slots)
    
    # LSTM
    y_scaled = scaler.transform(df_full["y"].values.reshape(-1, 1))
    ds_to_idx = {row["ds"]: i for i, row in df_full.iterrows()}
    sequences = []
    seq_indices = []
    for i, row in target_slots.iterrows():
        idx = ds_to_idx.get(row["ds"])
        if idx is not None and idx >= SEQUENCE_LEN:
            sequences.append(y_scaled[idx - SEQUENCE_LEN:idx])
            seq_indices.append(i)
    lstm_vals = np.zeros(n)
    if sequences:
        X_batch = np.array(sequences)
        preds_scaled = lstm_model.predict(X_batch, verbose=0).flatten()
        preds_real = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten().clip(min=0)
        for arr_i, slot_i in enumerate(seq_indices):
            lstm_vals[slot_i] = float(preds_real[arr_i])
            
    # XGBoost
    df_feat = df_full.copy()
    df_feat["hour"] = df_feat["ds"].dt.hour
    df_feat["minute_of_hour"] = df_feat["ds"].dt.minute
    df_feat["day_of_week"] = df_feat["ds"].dt.dayofweek
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["lag_1"] = df_feat["y"].shift(1)
    df_feat["lag_2"] = df_feat["y"].shift(2)
    df_feat["lag_3"] = df_feat["y"].shift(3)
    df_feat["lag_12"] = df_feat["y"].shift(LAG_HOUR_STEPS)
    df_feat["rolling_mean_6"] = df_feat["y"].shift(1).rolling(ROLLING_WINDOW_STEPS).mean()
    
    merged = target_slots.merge(df_feat[["ds"] + FEATURE_COLS], on="ds", how="left")
    valid_mask = merged[FEATURE_COLS].notna().all(axis=1)
    xgb_vals = np.zeros(n)
    if valid_mask.any():
        xgb_preds = xgb_model.predict(merged.loc[valid_mask, FEATURE_COLS]).clip(min=0)
        xgb_vals[valid_mask.values] = xgb_preds
        
    # Prophet
    prophet_preds = prophet_model.predict(target_slots[["ds"]])
    prophet_vals = prophet_preds["yhat"].clip(lower=0).values
    
    # Ensemble (40% Prophet, 30% LSTM, 30% XGBoost)
    ensemble_vals = (0.40 * prophet_vals + 0.30 * lstm_vals + 0.30 * xgb_vals).clip(min=0).round(1)
    
    res = target_slots[["ds"]].copy().rename(columns={"ds": "slot_local"})
    res["sim_prophet"] = prophet_vals
    res["sim_lstm"] = lstm_vals
    res["sim_xgb"] = xgb_vals
    res["sim_ensemble"] = ensemble_vals
    return res

# Metrics helper
def get_metrics(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs(yt - yp) / np.maximum(yt, 1.0)) * 100
    bias = np.mean(yp - yt)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE(%)": round(mape, 1), "Bias": round(bias, 3)}

# Run experiments
configs = [
    ("1. Default (Smooth=5, No Burst Bypass, Cap=30)", dict(smooth_win=5, burst_thresh=None, use_burst_bypass=False, outlier_cap=30)),
    ("2. Smooth=3, Cap=50, No Burst Bypass",          dict(smooth_win=3, burst_thresh=None, use_burst_bypass=False, outlier_cap=50)),
    ("3. Smooth=3, Burst-Bypass @ 15, Cap=None",       dict(smooth_win=3, burst_thresh=15, use_burst_bypass=True, outlier_cap=None)),
    ("4. Raw/No-Smooth (Smooth=1, Cap=None)",          dict(smooth_win=1, burst_thresh=None, use_burst_bypass=False, outlier_cap=None)),
]

print("=" * 75)
print("TUNING EXPERIMENT: PREDICTION ACCURACY ACROSS SMOOTHING/BURST SETTINGS")
print("Target window: Sunday 2026-09-27 10:00 - 13:00")
print("=" * 75)

summary_rows = []
for name, cfg in configs:
    sim_res = run_sim_backtest(**cfg)
    m = pd.merge(df_ground, sim_res, on="slot_local", how="inner")
    
    met_target = get_metrics(m["actual_target_cam"], m["sim_ensemble"])
    met_target["Config"] = name
    met_target["Evaluated_Against"] = f"Target Cam ({cam_id})"
    met_target["Mean_Pred"] = round(float(m["sim_ensemble"].mean()), 2)
    met_target["Sum_Pred"] = round(float(m["sim_ensemble"].sum()), 1)
    summary_rows.append(met_target)

df_summary = pd.DataFrame(summary_rows)[["Config", "Evaluated_Against", "MAE", "RMSE", "MAPE(%)", "Bias", "Mean_Pred", "Sum_Pred"]]
print(df_summary.to_string(index=False))

