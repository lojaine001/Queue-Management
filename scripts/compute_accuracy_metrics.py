import psycopg2
import pandas as pd
import numpy as np

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

# Ground truth: 3-min buckets of clean entrance events for Bosch_Camera_Entrance
df_actual_cam = pd.read_sql("""
    SELECT
        date_trunc('minute', timestamp AT TIME ZONE 'Europe/Paris') - 
        (EXTRACT(minute FROM timestamp AT TIME ZONE 'Europe/Paris')::integer % 3) * interval '1 minute' AS slot_local,
        COUNT(*) AS actual_arrivals
    FROM entrance_events
    WHERE camera_id = 'Bosch_Camera_Entrance'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 10:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
    GROUP BY slot_local
    ORDER BY slot_local
""", conn)

# Ground truth: 3-min buckets of entrance events across all cameras (NOT LIKE 'SIM_%')
df_actual_all = pd.read_sql("""
    SELECT
        date_trunc('minute', timestamp AT TIME ZONE 'Europe/Paris') - 
        (EXTRACT(minute FROM timestamp AT TIME ZONE 'Europe/Paris')::integer % 3) * interval '1 minute' AS slot_local,
        COUNT(*) AS actual_all_cams
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 10:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
    GROUP BY slot_local
    ORDER BY slot_local
""", conn)

# Current stored predictions in DB for today 10:00 to 13:00
df_preds = pd.read_sql("""
    SELECT DISTINCT ON (prediction_for)
        prediction_for AT TIME ZONE 'Europe/Paris' AS slot_local,
        predicted_at AT TIME ZONE 'Europe/Paris' AS predicted_at_local,
        prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat
    FROM queue_predictions
    WHERE prediction_for >= '2026-09-27 10:00:00+02' AND prediction_for <= '2026-09-27 13:00:00+02'
    ORDER BY prediction_for, predicted_at DESC
""", conn)

conn.close()

# Merge all three on slot_local
m = pd.merge(df_preds, df_actual_cam, on="slot_local", how="outer")
m = pd.merge(m, df_actual_all, on="slot_local", how="outer")
m["actual_arrivals"] = m["actual_arrivals"].fillna(0)
m["actual_all_cams"] = m["actual_all_cams"].fillna(0)
m = m.sort_values("slot_local").reset_index(drop=True)

# Metrics calculation helper
def calc_metrics(y_true, y_pred):
    valid = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[valid].values
    yp = y_pred[valid].values
    if len(yt) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "Bias": np.nan}
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    # Epsilon-smoothed MAPE to avoid division by zero
    mape = np.mean(np.abs(yt - yp) / np.maximum(yt, 1.0)) * 100
    bias = np.mean(yp - yt)
    return {"n": len(yt), "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE(%)": round(mape, 1), "Bias": round(bias, 3)}

print("=" * 70)
print("ACCURACY METRICS: STORED PREDICTIONS vs GROUND TRUTH (10:00 - 13:00)")
print("=" * 70)
print(f"Total 3-min slots in window: {len(m)}")
print(f"Total actual arrivals (Bosch_Camera_Entrance): {m['actual_arrivals'].sum():.0f}")
print(f"Total actual arrivals (All Cameras):           {m['actual_all_cams'].sum():.0f}")
print("\n--- METRICS VS TARGET CAMERA (Bosch_Camera_Entrance, dwell>=10s) ---")
res_target = []
for col in ["prophet_yhat", "lstm_yhat", "xgb_yhat", "ensemble_yhat"]:
    met = calc_metrics(m["actual_arrivals"], m[col])
    met["model"] = col
    res_target.append(met)
print(pd.DataFrame(res_target)[["model", "n", "MAE", "RMSE", "MAPE(%)", "Bias"]].to_string(index=False))

print("\n--- METRICS VS ALL CAMERAS (Entrance + Exit combined, as raw data plot showed) ---")
res_all = []
for col in ["prophet_yhat", "lstm_yhat", "xgb_yhat", "ensemble_yhat"]:
    met = calc_metrics(m["actual_all_cams"], m[col])
    met["model"] = col
    res_all.append(met)
print(pd.DataFrame(res_all)[["model", "n", "MAE", "RMSE", "MAPE(%)", "Bias"]].to_string(index=False))

print("\n--- BREAKDOWN BY TIME INTERVAL ---")
m["hour"] = m["slot_local"].dt.hour
for h in [10, 11, 12]:
    sub = m[m["hour"] == h]
    met_cam = calc_metrics(sub["actual_arrivals"], sub["ensemble_yhat"])
    met_all = calc_metrics(sub["actual_all_cams"], sub["ensemble_yhat"])
    print(f"Hour {h}:00 - {h+1}:00:")
    print(f"  Target Cam: Actual={sub['actual_arrivals'].sum():.0f} | Ens Pred={sub['ensemble_yhat'].sum():.1f} | MAE={met_cam['MAE']} | MAPE={met_cam['MAPE(%)']}% | Bias={met_cam['Bias']}")
    print(f"  All Cams:   Actual={sub['actual_all_cams'].sum():.0f} | Ens Pred={sub['ensemble_yhat'].sum():.1f} | MAE={met_all['MAE']} | MAPE={met_all['MAPE(%)']}% | Bias={met_all['Bias']}")

