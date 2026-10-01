import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT
        COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '1 hour') AS last_hour,
        COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '2 hours'
                          AND timestamp < NOW() - INTERVAL '1 hour') AS prev_hour
    FROM entrance_events
    WHERE camera_id = 'Bosch_Camera_Entrance'
      AND timestamp >= NOW() - INTERVAL '2 hours'
""")
print("last_hour, prev_hour:", cur.fetchone())
cur.execute("""
    SELECT timestamp, track_id FROM entrance_events
    WHERE camera_id = 'Bosch_Camera_Entrance'
    ORDER BY timestamp DESC LIMIT 5
""")
print("Most recent entrance rows:")
for row in cur.fetchall():
    print(" ", row)
cur.close(); conn.close()
