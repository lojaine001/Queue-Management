import os
import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    print("=== dashboard_state: what the tiles actually show ===")
    cur.execute("""
        SELECT queue_now, service_min, open_lanes,
               wait_0m, wait_5m, wait_10m, wait_15m, updated_at
        FROM dashboard_state WHERE id = 1
    """)
    print(dict(cur.fetchone()))

    print("\n=== queue_predictions: raw model breakdown, latest run ===")
    cur.execute("""
        SELECT prediction_for, prophet_yhat, lstm_yhat, xgb_yhat, ensemble_yhat,
               est_wait_minutes, wait_15m, wait_30m, wait_45m
        FROM queue_predictions
        WHERE predicted_at = (SELECT MAX(predicted_at) FROM queue_predictions)
        ORDER BY prediction_for
        LIMIT 6
    """)
    for row in cur.fetchall():
        print(dict(row))

conn.close()
