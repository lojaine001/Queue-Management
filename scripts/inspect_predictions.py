import psycopg2
import pandas as pd
from datetime import datetime

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

print("=== QUEUE PREDICTIONS 10:00 - 13:00 ===")
df_pred = pd.read_sql("""
    SELECT prediction_for, predicted_at, prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat
    FROM queue_predictions
    WHERE prediction_for >= '2026-09-27 10:00:00' AND prediction_for <= '2026-09-27 13:00:00'
    ORDER BY prediction_for ASC
""", conn)
print(f"Total prediction rows: {len(df_pred)}")
if not df_pred.empty:
    print(f"Earliest prediction_for: {df_pred['prediction_for'].min()}, Latest: {df_pred['prediction_for'].max()}")
    print(f"Earliest predicted_at: {df_pred['predicted_at'].min()}, Latest: {df_pred['predicted_at'].max()}")
    print("Summary of ensemble_yhat:")
    print(df_pred["ensemble_yhat"].describe())
    print("\nSample rows (first 10):")
    print(df_pred.head(10).to_string())

print("\n=== RAW ARRIVALS (ENTRANCE / 15-MIN BUCKETED) 10:00 - 13:00 ===")
df_events = pd.read_sql("""
    SELECT date_trunc('minute', timestamp) - (EXTRACT(minute FROM timestamp)::integer % 15) * interval '1 minute' as bucket,
           count(*) as count
    FROM entrance_events
    WHERE timestamp >= '2026-09-27 10:00:00' AND timestamp <= '2026-09-27 13:00:00'
    GROUP BY bucket
    ORDER BY bucket ASC
""", conn)
print(f"Total buckets: {len(df_events)}")
print(df_events.to_string())

conn.close()
