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
    SHOP_OPEN_TOT, is_open, get_where_clause
)
from prediction.model_bundle import current_bundle

models_dir = app_dir / "models"
active_bundle = current_bundle(models_dir)
print("Active bundle:", active_bundle)

lstm_model = tf.keras.models.load_model(active_bundle / f"lstm_queue_{BUCKET_MINUTES}m.keras")
with open(active_bundle / f"lstm_scaler_{BUCKET_MINUTES}m.pkl", "rb") as fh:
    scaler = pickle.load(fh)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(active_bundle / f"xgb_queue_{BUCKET_MINUTES}m.json")

with open(active_bundle / f"prophet_queue_{BUCKET_MINUTES}m.pkl", "rb") as fh:
    prophet_model = pickle.load(fh)

# Load today's entrance data
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
df_raw["y"]  = pd.to_numeric(df_raw["y"], errors="coerce").fillna(0)

# Also load ALL entrance events (including exit cam if any)
conn = psycopg2.connect(**DB_CONFIG)
df_all = pd.read_sql(f"""
    SELECT
        time_bucket('{BUCKET_MINUTES} minutes', timestamp) AS ds,
        COUNT(*) AS y
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 00:00:00+02'
    GROUP BY ds
    ORDER BY ds
""", conn)
conn.close()
df_all["ds"] = to_local(df_all["ds"])
df_all["y"]  = pd.to_numeric(df_all["y"], errors="coerce").fillna(0)

print(f"Loaded {len(df_raw)} Bosch_Camera_Entrance buckets, {len(df_all)} all-cam buckets")

# Test Prophet on today's slots
p_preds = prophet_model.predict(pd.DataFrame({"ds": df_raw["ds"]}))[["ds", "yhat"]]

merged = pd.merge(df_raw, df_all, on="ds", suffixes=("_cam_ent", "_all_cams"))
merged = pd.merge(merged, p_preds, on="ds")
print("\nSample comparison for today between 10:00 and 13:00:")
morn = merged[(merged["ds"].dt.hour >= 10) & (merged["ds"].dt.hour < 13)]
print(morn.head(20).to_string())

