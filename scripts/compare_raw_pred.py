import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

df_bucketed = pd.read_sql("""
    SELECT
        date_trunc('minute', timestamp) - (EXTRACT(minute FROM timestamp)::integer % 3) * interval '1 minute' AS ds,
        COUNT(*) AS arrivals
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
      AND timestamp >= '2026-09-27 08:00:00' AND timestamp <= '2026-09-27 11:00:00'
    GROUP BY ds
    ORDER BY ds
""", conn)

df_preds = pd.read_sql("""
    SELECT DISTINCT ON (prediction_for)
           prediction_for AS ds,
           predicted_at,
           prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat
    FROM queue_predictions
    WHERE prediction_for >= '2026-09-27 08:00:00' AND prediction_for <= '2026-09-27 11:00:00'
    ORDER BY prediction_for, predicted_at DESC
""", conn)

m = pd.merge(df_preds, df_bucketed, on="ds", how="outer").sort_values("ds")
print("=== 10:00 to 13:00 LOCAL TIME (08:00 to 11:00 UTC) ===")
print(m[["ds", "predicted_at", "arrivals", "prophet_yhat", "lstm_yhat", "xgb_yhat", "ensemble_yhat"]].to_string())

conn.close()
