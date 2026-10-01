import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT timestamp, camera_id, NOW() - timestamp AS age
    FROM entrance_events
    WHERE camera_id = %s
    ORDER BY timestamp DESC
    LIMIT 10
""", ("Bosch_Camera_Entrance",))
rows = cur.fetchall()
if not rows:
    print("NO ROWS AT ALL for Bosch_Camera_Entrance")
for row in rows:
    print(row)
cur.close(); conn.close()
