import json
import psycopg2
from psycopg2.extras import RealDictCursor

DB = dict(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
out = {'db': DB}
try:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT max(predicted_at) AS max_predicted_at FROM queue_predictions")
    out['max_predicted_at'] = cur.fetchone()['max_predicted_at']
    cur.execute("SELECT count(*) AS cnt_last_hour FROM queue_predictions WHERE predicted_at >= NOW() - INTERVAL '60 minutes'")
    out['cnt_last_hour'] = cur.fetchone()['cnt_last_hour']
    cur.execute("SELECT * FROM queue_predictions ORDER BY predicted_at DESC LIMIT 10")
    rows = cur.fetchall()
    out['sample_cols'] = list(rows[0].keys()) if rows else []
    out['sample_rows'] = rows
    cur.close()
    conn.close()
except Exception as e:
    out['error'] = str(e)

path = 'c:/Users/arnau/Documents/project/rtsp/Retail-Wait-Prediction/scripts/query_recent_queue_predictions.out.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, default=str, indent=2)
print('WROTE', path)
