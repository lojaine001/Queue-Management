import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("SET timezone = 'Europe/Paris'")
cur.execute("SELECT timestamp, queue_count, lane_counts, NOW() FROM queue_state_snapshots WHERE camera_id = 'Bosch_Camera_exit' ORDER BY timestamp DESC LIMIT 1")
print(cur.fetchone())
