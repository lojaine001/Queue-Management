"""
The services layer (iqms_api/services) without HTTP and without a real database.

- each service refuses bad requests with BadRequest / Conflict (turned into 400 / 409
  by the route), before touching the database
- what it writes goes through its repositories in one transaction, committed once
- the route turns those refusals into the same answers as before

Run:  python -m unittest tests.test_services -v
"""
import unittest
from unittest import mock

from tests.test_api_auth import load_api


class FakeCursor:
    """Records every statement; answers fetchone() with the next queued row."""
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, args=None): self.conn.statements.append((" ".join(sql.split()), args))
    def fetchone(self): return self.conn.rows.pop(0) if self.conn.rows else None
    def fetchall(self): return self.conn.rows.pop(0) if self.conn.rows else []


class FakeConnection:
    def __init__(self, rows=None):
        self.rows, self.statements, self.commits, self.used = list(rows or []), [], 0, False
    def __enter__(self): self.used = True; return self
    def __exit__(self, *a): return False
    def cursor(self, *a, **k): return FakeCursor(self)
    def commit(self): self.commits += 1


class ServiceTestCase(unittest.TestCase):
    service_module = None  # e.g. "administration"

    @classmethod
    def setUpClass(cls):
        load_api(f"api_services_test_{cls.__name__}")
        import importlib
        cls.errors = importlib.import_module("iqms_api.errors")
        cls.service = importlib.import_module(f"iqms_api.services.{cls.service_module}")

    def use_connection(self, conn):
        p = mock.patch.object(self.service, "_conn", return_value=conn)
        p.start()
        self.addCleanup(p.stop)
        return conn


class AdministrationServiceTests(ServiceTestCase):
    service_module = "administration"

    def test_lanes_out_of_range_refused_without_database(self):
        conn = self.use_connection(FakeConnection())
        for lanes in (0, 5, -1):
            with self.assertRaises(self.errors.BadRequest) as caught:
                self.service.set_lanes(lanes)
            self.assertEqual(caught.exception.detail, "lanes must be between 1 and 4")
        self.assertFalse(conn.used)

    def test_lane_change_refused_without_a_forecast(self):
        conn = self.use_connection(FakeConnection(rows=[{}]))  # no forecast published yet
        with self.assertRaises(self.errors.Conflict):
            self.service.set_lanes(2)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(len(conn.statements), 1)  # only the locking read, no update
        self.assertIn("FOR UPDATE", conn.statements[0][0])

    def test_unknown_alert_response_refused_with_a_stable_message(self):
        conn = self.use_connection(FakeConnection())
        with self.assertRaises(self.errors.BadRequest) as caught:
            self.service.record_alert_response("nonsense", None)
        self.assertEqual(caught.exception.detail,
                         "response must be one of {'opening_lane', 'cannot_open', 'false_alarm'}")
        self.assertFalse(conn.used)

    def test_alert_response_recorded(self):
        conn = self.use_connection(FakeConnection())
        self.assertEqual(self.service.record_alert_response("false_alarm", "2"),
                         {"status": "recorded", "response": "false_alarm"})
        self.assertEqual(conn.statements,
                         [("INSERT INTO alert_responses (response, lane_id) VALUES (%s, %s)", ("false_alarm", "2"))])
        self.assertEqual(conn.commits, 1)


class NotificationServiceTests(ServiceTestCase):
    service_module = "notifications"

    def test_subscribe_stores_the_device_setting(self):
        conn = self.use_connection(FakeConnection())
        self.assertEqual(self.service.subscribe("https://push/x", "p", "a", 7.5, 10), {"status": "subscribed"})
        sql, args = conn.statements[0]
        self.assertIn("ON CONFLICT (endpoint) DO UPDATE", sql)
        self.assertEqual(args, ("https://push/x", "p", "a", 7.5, 10))
        self.assertEqual(conn.commits, 1)

    def test_unsubscribe(self):
        conn = self.use_connection(FakeConnection())
        self.assertEqual(self.service.unsubscribe("https://push/x"), {"status": "unsubscribed"})
        self.assertEqual(conn.statements, [("DELETE FROM push_subscriptions WHERE endpoint = %s", ("https://push/x",))])
        self.assertEqual(conn.commits, 1)


class ErrorMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api("api_services_test_errors")
        from iqms_api import db, errors
        cls.db, cls.errors = db, errors

    def test_service_refusals_become_their_status(self):
        for exc, code in ((self.errors.BadRequest("bad"), 400), (self.errors.Conflict("busy"), 409)):
            http = self.db._http_error(exc)
            self.assertEqual((http.status_code, http.detail), (code, exc.detail))

    def test_other_failures_unchanged(self):
        self.assertEqual(self.db._http_error(self.db.PoolBusy("x")).status_code, 503)
        self.assertEqual(self.db._http_error(RuntimeError("boom")).status_code, 500)


if __name__ == "__main__":
    unittest.main()
