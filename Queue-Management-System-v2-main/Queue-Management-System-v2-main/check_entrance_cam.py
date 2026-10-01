import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("SET timezone = 'Europe/Paris'")
cur.execute("SELECT camera_id, COUNT(*), MAX(timestamp) FROM entrance_events WHERE timestamp >= NOW() - INTERVAL '2 hours' GROUP BY camera_id")
for row in cur.fetchall():
    print(row)
