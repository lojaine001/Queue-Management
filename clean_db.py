import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
conn.autocommit = True
cur = conn.cursor()
cur.execute("DELETE FROM queue_state_snapshots WHERE camera_id LIKE 'SIM_normal_day'")
cur.execute("DELETE FROM entrance_events WHERE camera_id LIKE 'SIM_normal_day'")
cur.execute("DELETE FROM service_events WHERE camera_id LIKE 'SIM_normal_day'")
print("Deleted runaway sim data")
