import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("SELECT MAX(predicted_at) FROM queue_predictions")
latest = cur.fetchone()[0]
print("Latest predicted_at:", latest)
cur.execute("""
    SELECT prediction_for, est_wait_minutes, status
    FROM queue_predictions
    WHERE predicted_at = %s
    ORDER BY prediction_for ASC
    LIMIT 15
""", (latest,))
for row in cur.fetchall():
    print(row)
cur.close(); conn.close()
