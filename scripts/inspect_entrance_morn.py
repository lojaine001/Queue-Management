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
df_raw = pd.read_sql(f"""
    SELECT
        time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS ds,
        COUNT(*) AS y
    FROM entrance_events
    WHERE camera_id = '{cam_id}'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 00:00:00+02'
    GROUP BY ds
    ORDER BY ds
""", conn)
conn.close()

df_raw["ds"] = to_local(df_raw["ds"])
df_raw["y"] = pd.to_numeric(df_raw["y"], errors="coerce").fillna(0)

# Check burst values
burst_mask = df_raw["y"] >= 40
print(f"Raw buckets: {len(df_raw)}, burst count: {burst_mask.sum()}, max y: {df_raw['y'].max()}, mean y: {df_raw['y'].mean():.2f}")
print("Values from 10am to 1pm for camera", cam_id)
morn = df_raw[(df_raw["ds"].dt.hour >= 10) & (df_raw["ds"].dt.hour < 13)]
print(morn.to_string())
