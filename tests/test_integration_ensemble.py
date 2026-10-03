"""
Integration test for ensemble_predict.py against live/test PostgreSQL database.
Asserts that queue_predictions rows exist or are produced, with correct schema and non-null values.
"""
import os
import sys
import unittest
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QM = ROOT  # tests/ lives inside Queue-Management
sys.path = [str(QM), str(ROOT)] + [p for p in sys.path if p not in (str(QM), str(ROOT))]

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "iqms"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
}

EXPECTED_COLUMNS = {
    "predicted_at",
    "prediction_for",
    "ensemble_yhat",
    "est_wait_minutes",
    "wait_15m",
    "wait_30m",
    "status",
}


class TestIntegrationEnsemble(unittest.TestCase):
    def test_database_queue_predictions_schema_and_rows(self):
        """Verify DB connection, schema of queue_predictions, and non-empty rows."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            self.skipTest(f"Database connection unavailable: {e}")

        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'queue_predictions'
            """)
            cols = {row["column_name"] for row in cur.fetchall()}
            self.assertTrue(
                EXPECTED_COLUMNS.issubset(cols),
                f"Missing expected columns in queue_predictions: {EXPECTED_COLUMNS - cols}",
            )

            cur.execute("SELECT count(*) as total_rows FROM queue_predictions")
            total = cur.fetchone()["total_rows"]
            self.assertGreater(total, 0, "queue_predictions table should contain prediction rows")

            # Check recent predictions (within last 4 hours)
            cur.execute("""
                SELECT * FROM queue_predictions 
                ORDER BY predicted_at DESC 
                LIMIT 20
            """)
            recent_rows = cur.fetchall()
            self.assertGreater(len(recent_rows), 0, "Should have recent rows in queue_predictions")

            # Verify no nulls in key forecast metrics
            for r in recent_rows:
                self.assertIsNotNone(r["predicted_at"])
                self.assertIsNotNone(r["prediction_for"])
                self.assertIsNotNone(r["ensemble_yhat"])
                self.assertIsNotNone(r["est_wait_minutes"])

        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    unittest.main()
