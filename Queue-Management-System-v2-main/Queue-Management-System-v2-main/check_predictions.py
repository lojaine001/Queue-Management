import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("SET timezone = 'Europe/Paris'")
cur.execute("SELECT predicted_at, prediction_for, NOW() FROM queue_predictions ORDER BY predicted_at DESC LIMIT 8")
for row in cur.fetchall():
    print(row)
