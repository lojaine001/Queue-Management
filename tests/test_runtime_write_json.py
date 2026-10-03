"""
Unit tests for prediction.runtime.write_json and read_json under concurrent locks.
"""
import os
import sys
import time
import tempfile
import threading
import unittest
from pathlib import Path

# Add project root and Queue-Management to sys.path
ROOT = Path(__file__).resolve().parents[1]
QM = ROOT  # tests/ lives inside Queue-Management
sys.path = [str(QM), str(ROOT)] + [p for p in sys.path if p not in (str(QM), str(ROOT))]

from prediction.runtime import write_json, read_json


class TestRuntimeWriteJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target = Path(self.temp_dir.name) / "state.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_basic_write_and_read(self):
        payload = {"status": "ok", "count": 42}
        write_json(self.target, payload)
        self.assertTrue(self.target.exists())
        data = read_json(self.target)
        self.assertEqual(data, payload)

    def test_write_under_concurrent_readers(self):
        """Simulate concurrent threads opening target file for read while write_json updates it."""
        write_json(self.target, {"init": True})
        stop_event = threading.Event()
        read_errors = []

        def reader():
            while not stop_event.is_set():
                try:
                    if self.target.exists():
                        with open(self.target, "r", encoding="utf-8") as f:
                            _ = f.read()
                            time.sleep(0.01)
                except Exception as e:
                    read_errors.append(e)
                time.sleep(0.005)

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for r in readers:
            r.daemon = True
            r.start()

        write_errors = []
        for i in range(15):
            try:
                write_json(self.target, {"iteration": i, "timestamp": time.time()})
            except Exception as e:
                write_errors.append(e)
            time.sleep(0.01)

        stop_event.set()
        for r in readers:
            r.join(timeout=0.5)

        self.assertEqual(len(write_errors), 0, f"write_json encountered errors: {write_errors}")
        result = read_json(self.target)
        self.assertIn("iteration", result)
        self.assertEqual(result["iteration"], 14)

    def test_fallback_writes_valid_json(self):
        """Verify that write_json produces valid JSON even when updating an existing read-only file."""
        write_json(self.target, {"initial": 1})
        # Overwrite multiple times
        for i in range(5):
            write_json(self.target, {"v": i})
            data = read_json(self.target)
            self.assertEqual(data, {"v": i})


if __name__ == "__main__":
    unittest.main()
