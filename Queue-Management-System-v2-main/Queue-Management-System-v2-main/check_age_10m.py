import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT COUNT(*), COUNT(age_estimate)
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND timestamp >= NOW() - INTERVAL '10 minutes'
""")
print(cur.fetchone())
conn.close()
