import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT DATE(timestamp) AS day, COUNT(*) AS total,
           COUNT(age_estimate) AS with_age
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
      AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY 1 ORDER BY 1 DESC
""")
for row in cur.fetchall():
    print(row)
cur.close(); conn.close()
