import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()

cur.execute("""
    SELECT COUNT(*), camera_id
    FROM entrance_events
    WHERE timestamp >= '2026-04-23 15:00:00+02'
    AND timestamp <  '2026-04-23 15:12:00+02'
    AND camera_id NOT LIKE 'SIM_%'
    GROUP BY camera_id
""")

rows = cur.fetchall()
print("Results:")
for r in rows:
    print(f"  camera: {r[1]}  |  count: {r[0]}")
if not rows:
    print("  No entries found in that time window.")

conn.close()
