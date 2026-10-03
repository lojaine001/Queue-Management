from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prediction.pipeline import DB_CONFIG
import psycopg2
import pandas as pd
from datetime import datetime


def _connect():
    conn = psycopg2.connect(**{**DB_CONFIG, "connect_timeout": 10})
    return conn


def inspect(days=14):
    conn = _connect()
    try:
        query = f"""
        SELECT date(timezone('UTC', timestamp AT TIME ZONE 'UTC') AT TIME ZONE %(tz)s) AS day,
               time_bucket('3 minutes', timestamp) AS bucket,
               COUNT(*) AS cnt
        FROM entrance_events
        WHERE timestamp >= NOW() - INTERVAL '%(days)s days'
          AND camera_id = %(cam)s
        GROUP BY day, bucket
        ORDER BY day, bucket
        """
        params = {"days": days, "cam": 'Bosch_Camera_Entrance', "tz": 'UTC'}
        df = pd.read_sql(query, conn, params=params)
        if df.empty:
            print('No rows found')
            return
        # Filter buckets between 10:00 and 12:00 local
        df['local_time'] = pd.to_datetime(df['bucket'])
        df['hour'] = df['local_time'].dt.hour
        df_mid = df[(df['hour'] >= 10) & (df['hour'] < 12)]
        summary = df_mid.groupby('day')['cnt'].sum().reset_index()
        print('Day, total_entries_10-12')
        for _, r in summary.iterrows():
            print(f"{r['day']}: {int(r['cnt'])}")
    finally:
        conn.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    args = p.parse_args()
    inspect(days=args.days)
