import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

df = pd.read_sql("""
    SELECT camera_id, COUNT(*) as count, COUNT(CASE WHEN dwell_seconds >= 10 THEN 1 END) as gte10
    FROM entrance_events
    WHERE timestamp >= '2026-09-27 10:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
    GROUP BY camera_id
""", conn)
print("Camera breakdown between 10am and 1pm:")
print(df.to_string())

conn.close()
