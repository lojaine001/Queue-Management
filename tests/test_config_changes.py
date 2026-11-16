"""
Configuration change log (config_changes, migration 0002): who changed which setting,
when, from what to what.

- a real change writes one row; setting a value to what it already was writes nothing
- forecast settings: one row per setting that changed, with the new settings revision
- open lanes from the API (/set-lanes) and from the dashboard are recorded with who
  changed them; a refused change (stale forecast) records nothing
- the record is written on the same cursor as the change (same transaction)
- who = the request's address and app for the API

No real database: fake connections record the statements.

Run:  python -m unittest tests.test_config_changes -v
"""
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from tests.test_api_auth import load_api

INSERT = "INSERT INTO config_changes"


class Recorder:
    """A cursor and connection in one: records statements, answers fetchone from a queue."""
    def __init__(self, answers=()):
        self.statements, self.answers, self.commits, self.closed = [], list(answers), 0, False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self, *a, **k): return self
    def execute(self, sql, args=None): self.statements.append((" ".join(sql.split()), args))
    def fetchone(self): return self.answers.pop(0) if self.answers else None
    def fetchall(self): return self.answers.pop(0) if self.answers else []
    def commit(self): self.commits += 1
    def close(self): self.closed = True

    def inserts(self):
        return [args for sql, args in self.statements if sql.startswith(INSERT)]


def json_value(adapted):
    return None if adapted is None else adapted.adapted


def fresh_state(open_lanes=3):
    scenario = {"wait_0m": 1.0, "wait_5m": 1.5, "wait_10m": 2.0, "wait_15m": 2.5}
    return {"open_lanes": open_lanes,
            "forecast_json": {"as_of": datetime.now(timezone.utc).isoformat(), "lanes": {"2": scenario}}}


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api("api_config_changes_test")
        from prediction import config_audit
        cls.audit = config_audit

    def test_real_change_one_row(self):
        cur = Recorder()
        self.assertTrue(self.audit.record(cur, "api", "API key, 10.0.0.2, IQMSManager", "open_lanes", 3, 2))
        (args,) = cur.inserts()
        source, actor, setting, old, new, revision = args
        self.assertEqual((source, actor, setting, revision), ("api", "API key, 10.0.0.2, IQMSManager", "open_lanes", None))
        self.assertEqual((json_value(old), json_value(new)), (3, 2))

    def test_same_value_no_row(self):
        cur = Recorder()
        self.assertFalse(self.audit.record(cur, "dashboard", "x", "open_lanes", 2, 2))
        self.assertEqual(cur.statements, [])

    def test_missing_value_is_sql_null_and_long_actor_cut(self):
        cur = Recorder()
        self.audit.record(cur, "dashboard", "y" * 500, "calibration_enabled", None, True)
        (args,) = cur.inserts()
        self.assertIsNone(args[3])          # old value: NULL, not the JSON text "null"
        self.assertEqual(len(args[1]), self.audit.MAX_ACTOR_LEN)

    def test_unknown_source_refused(self):
        with self.assertRaises(ValueError):
            self.audit.record(Recorder(), "cron", None, "open_lanes", 1, 2)

    def test_only_changed_settings(self):
        cur = Recorder()
        old = {"arrival_models": ["prophet"], "calibration_enabled": False, "pred_smooth_min": 5}
        new = {"arrival_models": ["prophet"], "calibration_enabled": True, "pred_smooth_min": 15}
        self.assertEqual(self.audit.record_settings(cur, "dashboard", "d", old, new, 51),
                         ["calibration_enabled", "pred_smooth_min"])
        self.assertEqual([(a[2], json_value(a[3]), json_value(a[4]), a[5]) for a in cur.inserts()],
                         [("calibration_enabled", False, True, 51), ("pred_smooth_min", 5, 15, 51)])


class ApiLaneChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api("api_config_changes_api_test")
        from iqms_api import errors, security
        from iqms_api.services import administration
        cls.service, cls.errors, cls.security = administration, errors, security

    def run_set_lanes(self, state, lanes, actor="API key, 10.0.0.2, IQMSManager"):
        conn = Recorder(answers=[state])
        with mock.patch.object(self.service, "_conn", return_value=conn):
            self.service.set_lanes(lanes, actor=actor)
        return conn

    def test_lane_change_recorded_in_the_same_transaction(self):
        conn = self.run_set_lanes(fresh_state(open_lanes=3), 2)
        kinds = [sql.split()[0] for sql, _ in conn.statements]
        self.assertEqual(kinds, ["SELECT", "UPDATE", "INSERT"])  # lock, change, record: one cursor
        (args,) = conn.inserts()
        self.assertEqual((args[0], args[1], args[2]), ("api", "API key, 10.0.0.2, IQMSManager", "open_lanes"))
        self.assertEqual((json_value(args[3]), json_value(args[4])), (3, 2))
        self.assertEqual(conn.commits, 1)

    def test_same_lanes_not_recorded(self):
        self.assertEqual(self.run_set_lanes(fresh_state(open_lanes=2), 2).inserts(), [])

    def test_refused_change_not_recorded(self):
        stale = fresh_state(open_lanes=3)
        stale["forecast_json"]["as_of"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        conn = Recorder(answers=[stale])
        with mock.patch.object(self.service, "_conn", return_value=conn), self.assertRaises(self.errors.Conflict):
            self.service.set_lanes(2, actor="x")
        self.assertEqual(conn.inserts(), [])

    def test_who_from_the_request(self):
        req = types.SimpleNamespace(headers={"x-forwarded-for": "192.168.1.20, 127.0.0.1",
                                             "user-agent": "IQMSManager/1.2 okhttp"},
                                    client=types.SimpleNamespace(host="127.0.0.1"))
        self.assertEqual(self.security.caller(req), "API key, 192.168.1.20, IQMSManager/1.2 okhttp")
        direct = types.SimpleNamespace(headers={}, client=types.SimpleNamespace(host="10.0.0.7"))
        self.assertEqual(self.security.caller(direct), "API key, 10.0.0.7, unknown app")
        self.assertEqual(self.security.caller(None), "API key")


class DashboardChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api("api_config_changes_dashboard_test")
        from prediction import shared_forecast_store
        cls.store = shared_forecast_store

    def test_settings_recorded_with_new_revision(self):
        current = {"arrival_models": ["prophet"], "calibration_enabled": False, "pred_smooth_min": 5}
        conn = Recorder(answers=[(dict(current),), (51,)])  # SELECT ... FOR UPDATE, UPDATE ... RETURNING
        with mock.patch.object(self.store, "connect", return_value=conn), \
                mock.patch.object(self.store, "validate_settings"):
            self.store.update_settings({"calibration_enabled": True}, actor="dashboard, browser 127.0.0.1")
        self.assertIn("RETURNING revision", conn.statements[1][0])
        (args,) = conn.inserts()
        self.assertEqual((args[0], args[1], args[2], json_value(args[3]), json_value(args[4]), args[5]),
                         ("dashboard", "dashboard, browser 127.0.0.1", "calibration_enabled", False, True, 51))
        self.assertTrue(conn.closed)

    def test_dashboard_lane_change_recorded(self):
        conn = Recorder(answers=[{"open_lanes": 4}])
        with mock.patch.object(self.store, "connect", return_value=conn):
            self.store.select_lanes(2, actor="dashboard, browser 127.0.0.1")
        (args,) = conn.inserts()
        self.assertEqual((args[0], args[2], json_value(args[3]), json_value(args[4])), ("dashboard", "open_lanes", 4, 2))


if __name__ == "__main__":
    unittest.main()
