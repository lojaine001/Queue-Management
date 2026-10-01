import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
for table in ["entrance_events", "service_events", "queue_state_snapshots"]:
    print(f"--- {table} ---")
    cur.execute(f"""
        SELECT camera_id, COUNT(*), MIN(timestamp), MAX(timestamp)
        FROM {table}
        GROUP BY camera_id
        ORDER BY COUNT(*) DESC
    """)
    for row in cur.fetchall():
        print(" ", row)
cur.close(); conn.close()
