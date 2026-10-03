import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="iqms", user="postgres", password="0000", host="localhost", port=5432)

df = pd.read_sql("""
    SELECT
        DATE(timestamp AT TIME ZONE 'Europe/Paris') as date,
        COUNT(*) as total_events,
        COUNT(CASE WHEN dwell_seconds >= 10 THEN 1 END) as valid_events
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
    GROUP BY date
    ORDER BY date DESC
    LIMIT 40
""", conn)
print("Entrance events by date (last 40 days):")
print(df.to_string())

conn.close()
