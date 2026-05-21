import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("""
    SELECT COUNT(*) FROM entrance_events
    WHERE timestamp >= '2026-05-19 16:00:00'::TIMESTAMPTZ
      AND timestamp <  '2026-05-19 18:00:00'::TIMESTAMPTZ
""")
print('Entries 16:00-18:00:', cur.fetchone()[0])
conn.close()