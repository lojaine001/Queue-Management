import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from prediction.pipeline import DB_CONFIG
import psycopg2
from datetime import datetime

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT max(predicted_at), count(*) FROM queue_predictions")
r = cur.fetchone()
print('max(predicted_at)=', r[0], 'count=', r[1])
cur.execute("SELECT predicted_at, checkout_arrival, predicted_queue FROM queue_predictions ORDER BY predicted_at DESC LIMIT 10")
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()
conn.close()
