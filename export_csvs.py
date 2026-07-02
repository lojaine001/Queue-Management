"""
export_csvs.py — dump production tables to CSV for generate_analysis.py.

Run on Arnaud's machine:
    python export_csvs.py

Output: service_events.csv, entrance_events.csv, queue_predictions.csv
Then run: python generate_analysis.py
"""
import os, sys
import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

OUT = os.path.dirname(os.path.abspath(__file__))

def connect():
    try:
        return psycopg2.connect(**DB)
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        sys.exit(1)

def export(conn, query, filename, label):
    print(f"  Exporting {label}...")
    df = pd.read_sql(query, conn)
    path = os.path.join(OUT, filename)
    df.to_csv(path, index=False)
    print(f"    {len(df):,} rows → {filename}")
    return df

def main():
    print(f"Connecting to {DB['host']}:{DB['port']}/{DB['dbname']} ...")
    conn = connect()
    print("Connected.\n")

    export(conn, """
        SELECT *
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%'
        ORDER BY timestamp
    """, "service_events.csv", "service_events (real camera only)")

    export(conn, """
        SELECT *
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%'
        ORDER BY timestamp
    """, "entrance_events.csv", "entrance_events (real camera only)")

    export(conn, """
        SELECT *
        FROM queue_predictions
        ORDER BY predicted_at
    """, "queue_predictions.csv", "queue_predictions (all sources)")

    conn.close()
    print("\nDone. Now run: python generate_analysis.py")

if __name__ == "__main__":
    main()
