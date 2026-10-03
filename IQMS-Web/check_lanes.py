import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("""
    SELECT camera_id, queue_count, active_lanes, timestamp
    FROM (
        SELECT DISTINCT ON (camera_id)
            camera_id, queue_count, active_lanes, timestamp
        FROM queue_state_snapshots
        WHERE camera_id NOT LIKE 'SIM_%'
        ORDER BY camera_id, timestamp DESC
    ) latest
""")
for row in cur.fetchall():
    print(row)
