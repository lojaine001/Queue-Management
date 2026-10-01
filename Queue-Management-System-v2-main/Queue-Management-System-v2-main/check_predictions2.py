import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("SET timezone = 'Europe/Paris'")
cur.execute("SELECT COUNT(*), MIN(prediction_for), MAX(prediction_for) FROM queue_predictions WHERE predicted_at = (SELECT MAX(predicted_at) FROM queue_predictions)")
print("Latest batch:", cur.fetchone())
cur.execute("SELECT prediction_for FROM queue_predictions WHERE predicted_at = (SELECT MAX(predicted_at) FROM queue_predictions) ORDER BY prediction_for")
print("All prediction_for in latest batch:")
for row in cur.fetchall():
    print(" ", row[0])
cur.execute("SELECT COUNT(*) FROM queue_predictions WHERE prediction_for >= NOW() AND prediction_for <= NOW() + INTERVAL '\''60 minutes'\''")
print("Rows matching /forecast-chart window:", cur.fetchone())
