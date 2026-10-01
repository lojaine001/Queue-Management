import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT timestamp, camera_id, NOW() AS db_now, NOW() - timestamp AS age
    FROM entrance_events
    ORDER BY timestamp DESC
    LIMIT 10
""")
for row in cur.fetchall():
    print(row)
cur.close(); conn.close()
