import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

print("=== SUNDAY 2026-09-27 PREDICTIONS (10:00 to 13:00 Europe/Paris) ===")
df_pred = pd.read_sql("""
    SELECT
        prediction_for AT TIME ZONE 'Europe/Paris' as slot_local,
        predicted_at AT TIME ZONE 'Europe/Paris' as predicted_at_local,
        prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat, source
    FROM queue_predictions
    WHERE prediction_for >= '2026-09-27 10:00:00+02' AND prediction_for <= '2026-09-27 13:00:00+02'
    ORDER BY prediction_for ASC
""", conn)
print(f"Count: {len(df_pred)}")
if not df_pred.empty:
    print(df_pred.to_string())

print("\n=== SUNDAY 2026-09-27 ENTRANCE EVENTS (15-min resample) ===")
df_ent = pd.read_sql("""
    SELECT
        time_bucket('15 minutes', timestamp AT TIME ZONE 'Europe/Paris') as bucket_local,
        COUNT(*) as arrivals
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND timestamp >= '2026-09-27 00:00:00+02' AND timestamp <= '2026-09-27 23:59:59+02'
    GROUP BY bucket_local
    ORDER BY bucket_local ASC
""", conn)
print(f"Total buckets for Sunday: {len(df_ent)}")
print(df_ent.to_string())

conn.close()
