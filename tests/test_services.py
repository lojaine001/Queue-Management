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


class ExportServiceTests(ServiceTestCase):
    service_module = "exports"

    def test_period_bounds(self):
        from datetime import date
        b = self.service.period_bounds
        self.assertEqual(b("day", "2026-11-12"), (date(2026, 11, 12), date(2026, 11, 13)))
        self.assertEqual(b("week", "2026-11-15"), (date(2026, 11, 9), date(2026, 11, 16)))  # Sunday -> its Monday
        self.assertEqual(b("week", "2026-11-09"), (date(2026, 11, 9), date(2026, 11, 16)))  # Monday itself
        self.assertEqual(b("month", "2026-12-31"), (date(2026, 12, 1), date(2027, 1, 1)))   # year rollover
        self.assertEqual(b("month", "2028-02-10"), (date(2028, 2, 1), date(2028, 3, 1)))

    def test_bad_requests_refused_in_the_same_order_without_database(self):
        conn = self.use_connection(FakeConnection())
        cases = [  # (args, expected detail): the first mistake in this order is reported
            (("year", "bad-date", "bogus", "xlsx", "fr"), "Only format=csv is supported currently"),
            (("year", "bad-date", "bogus", "csv", "fr"), "period must be day, week, or month"),
            (("week", "2026-13-01", "bogus", "csv", "fr"), "date must be YYYY-MM-DD"),
            (("week", "2026-11-12", "customers,bogus", "csv", "fr"), "Unknown KPI(s): ['bogus']"),
        ]
        for args, detail in cases:
            with self.assertRaises(self.errors.BadRequest) as caught:
                self.service.csv_file(*args)
            self.assertEqual(caught.exception.detail, detail)
        with self.assertRaises(self.errors.BadRequest):
            self.service.preview("day", "2026-11-12", "nope", "en")
        self.assertFalse(conn.used)


class KpiServiceTests(ServiceTestCase):
    service_module = "kpis"

    def test_day_series_has_every_hour_even_when_data_is_sparse(self):
        from datetime import date, datetime
        conn = FakeConnection(rows=[
            [{"bucket": datetime(2026, 11, 12, 9), "avg_wait_min": 1.5, "max_wait_min": 4.0}],  # by hour
            {"avg_wait_min": 1.5, "max_wait_min": 4.0},                                         # overall
        ])
        with conn.cursor() as cur:
            result = self.service.kpi_wait(cur, date(2026, 11, 12), date(2026, 11, 13), "hour")
        self.assertEqual(len(result["by_hour"]), 24)
        self.assertEqual(result["by_hour"][9], {"hour": "09:00", "avg_wait_min": 1.5, "max_wait_min": 4.0})
        self.assertEqual(result["by_hour"][10], {"hour": "10:00", "avg_wait_min": None, "max_wait_min": None})
        self.assertEqual((result["avg_wait_min"], result["max_wait_min"]), (1.5, 4.0))

    def test_no_day_means_no_query(self):
        conn = FakeConnection()
        with conn.cursor() as cur:
            self.assertEqual(self.service.kpi_queue(cur, None, None)["by_hour"], [])
        self.assertEqual(conn.statements, [])


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
