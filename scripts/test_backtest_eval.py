import sys
from pathlib import Path

# Add Queue-Management paths
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
from prediction.core import BUCKET_MINUTES, LAG_HOUR_STEPS, ROLLING_WINDOW_STEPS, SEQUENCE_LEN, is_open

models_dir = app_dir / "models"
lstm_model = tf.keras.models.load_model(models_dir / f"lstm_queue_{BUCKET_MINUTES}m.keras")
with open(models_dir / f"lstm_scaler_{BUCKET_MINUTES}m.pkl", "rb") as fh:
    scaler = pickle.load(fh)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(models_dir / f"xgb_queue_{BUCKET_MINUTES}m.json")

print("[Test] Models loaded successfully from:", models_dir)

# Query today's entrance data
conn = psycopg2.connect(**DB_CONFIG)
df_raw = pd.read_sql(f"""
    SELECT
        time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS ds,
        COUNT(*) AS y
    FROM entrance_events
    WHERE camera_id = 'Bosch_Camera_Entrance'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 00:00:00+02'
    GROUP BY ds
    ORDER BY ds
""", conn)
conn.close()

df_raw["ds"] = to_local(df_raw["ds"])
df_raw["y"] = pd.to_numeric(df_raw["y"], errors="coerce").fillna(0)
print(f"Total 3-min buckets today: {len(df_raw)}")
print(f"Total arrivals recorded today: {df_raw['y'].sum()}")
print("Sample raw buckets from 10:00 to 13:00:")
morn = df_raw[(df_raw["ds"].dt.hour >= 10) & (df_raw["ds"].dt.hour < 13)]
print(morn.head(10).to_string())
