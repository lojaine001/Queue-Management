import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

df = pd.read_sql("""
    SELECT
        COUNT(*) as total_events,
        COUNT(CASE WHEN dwell_seconds >= 10 THEN 1 END) as gte_10,
        COUNT(CASE WHEN dwell_seconds >= 5 THEN 1 END) as gte_5,
        COUNT(CASE WHEN dwell_seconds >= 3 THEN 1 END) as gte_3,
        COUNT(CASE WHEN dwell_seconds < 10 THEN 1 END) as lt_10,
        AVG(dwell_seconds) as avg_dwell,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dwell_seconds) as median_dwell,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY dwell_seconds) as p90_dwell
    FROM entrance_events
    WHERE camera_id = 'Bosch_Camera_Entrance'
      AND timestamp >= '2026-09-27 10:00:00+02' AND timestamp <= '2026-09-27 13:00:00+02'
""", conn)
print("Entrance events dwell_seconds stats between 10am and 1pm today:")
print(df.to_string())

conn.close()
