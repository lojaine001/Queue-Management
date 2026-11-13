"""
/health/ready and /health/workers.

- ready: 200 when the database answers and is at the migration version in db/;
  503 with the reason when it is behind, was never marked, or does not answer
- workers: 200 when every part's last sign of life is recent; 503 naming the part
  that is late ("stale") or has never been seen ("missing")
- no error text from the database or the heartbeat files reaches the answer
  (these endpoints are public through the web link)
- the check logic itself: limits, time zones, the newest migration in a chain

No real database: the pool returns a fake connection; heartbeat files are written
to a temporary folder.

Run:  python -m unittest tests.test_health -v
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests.test_api_auth import TestClient, load_api

NOW = datetime.now(timezone.utc)


class FakeCursor:
    def __init__(self, answers): self.answers, self.last = answers, None
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, args=None):
        if self.answers.get("error"):
            raise self.answers["error"]
        self.last = next((k for k in self.answers if k in sql), None)
    def fetchone(self): return self.answers.get(self.last)
    def fetchall(self): return self.answers.get(self.last)


class FakeConnection:
    def __init__(self, answers): self.answers = answers
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self, *a, **k): return FakeCursor(self.answers)


def write_migration(folder, revision, down):
    (Path(folder) / f"{revision}.py").write_text(
        f"revision = {revision!r}\ndown_revision = {down!r}\n", encoding="utf-8")


@unittest.skipIf(TestClient is None, "fastapi TestClient needs httpx")
class HealthEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = load_api("api_health_test")
        from iqms_api import settings
        from iqms_api.routes import health
        cls.health, cls.checkout, cls.entrance = health, settings.CHECKOUT_CAM_ID, settings.ENTRANCE_CAM_ID
        try:
            cls.client = TestClient(cls.api.app)
        except TypeError as exc:
            raise unittest.SkipTest(f"TestClient incompatible with installed httpx ({exc})")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.runtime = Path(tmp.name) / "runtime"
        self.runtime.mkdir()
        self.migrations = Path(tmp.name) / "versions"
        self.migrations.mkdir()
        write_migration(self.migrations, "0001_baseline", None)
        for target, value in (("RUNTIME", self.runtime), ("MIGRATIONS_DIR", self.migrations)):
            p = mock.patch.object(self.health, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.answers = {}
        p = mock.patch.object(self.api.db.DB_POOL, "connection", side_effect=lambda: FakeConnection(self.answers))
        p.start()
        self.addCleanup(p.stop)

    def heartbeat(self, name, age_sec, **extra):
        state = {"heartbeat": (NOW - timedelta(seconds=age_sec)).isoformat(), **extra}
        (self.runtime / name).write_text(json.dumps(state), encoding="utf-8")

    def all_fresh(self):
        self.heartbeat("scheduler.json", 2, status="running", interval_min=3,
                       last_error="C:\\secret\\path\\boom.py line 9")
        self.heartbeat("push_alerts.json", 10, last_check="ok")
        self.answers.update({
            "queue_state_snapshots": [(self.checkout, NOW - timedelta(seconds=8)),
                                      (self.entrance, NOW - timedelta(seconds=5))],
            "wait_forecast_runs": (NOW - timedelta(minutes=2),),
        })

    # ── /health/ready ────────────────────────────────────────────────────────
    def test_ready_when_database_at_expected_version(self):
        self.answers.update({"to_regclass": (True,), "alembic_version": ("0001_baseline",)})
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["schema_version"], "0001_baseline")

    def test_not_ready_when_a_migration_is_not_applied(self):
        write_migration(self.migrations, "0002_next", "0001_baseline")
        self.answers.update({"to_regclass": (True,), "alembic_version": ("0001_baseline",)})
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 503)
        self.assertIn("upgrade head", r.json()["reason"])
        self.assertEqual(r.json()["schema_expected"], "0002_next")

    def test_not_ready_when_database_never_marked(self):
        self.answers.update({"to_regclass": (False,)})
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 503)
        self.assertIn("no migration version", r.json()["reason"])

    def test_not_ready_when_database_down_without_leaking_the_error(self):
        self.answers["error"] = RuntimeError("password authentication failed for user postgres")
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("password", r.text)

    # ── /health/workers ──────────────────────────────────────────────────────
    def test_all_workers_ok(self):
        self.all_fresh()
        r = self.client.get("/health/workers")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(set(body["workers"]),
                         {"camera_checkout", "camera_entrance", "forecast", "push_alerts", "scheduler"})
        self.assertNotIn("secret", r.text)  # scheduler last_error is never shown

    def test_frozen_camera_is_stale(self):
        self.all_fresh()
        self.answers["queue_state_snapshots"] = [(self.checkout, NOW - timedelta(minutes=5)),
                                                 (self.entrance, NOW - timedelta(seconds=5))]
        r = self.client.get("/health/workers")
        self.assertEqual(r.status_code, 503)
        w = r.json()["workers"]
        self.assertEqual(w["camera_checkout"]["status"], "stale")
        self.assertEqual(w["camera_entrance"]["status"], "ok")

    def test_camera_never_seen_is_missing(self):
        self.all_fresh()
        self.answers["queue_state_snapshots"] = [(self.checkout, NOW)]
        self.assertEqual(self.client.get("/health/workers").json()["workers"]["camera_entrance"]["status"],
                         "missing")

    def test_push_checker_without_heartbeat_is_missing(self):
        self.all_fresh()
        (self.runtime / "push_alerts.json").unlink()
        r = self.client.get("/health/workers")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["workers"]["push_alerts"]["status"], "missing")

    def test_stopped_scheduler_is_not_ok_even_with_recent_heartbeat(self):
        self.all_fresh()
        self.heartbeat("scheduler.json", 1, status="stopped", interval_min=3)
        self.assertEqual(self.client.get("/health/workers").json()["workers"]["scheduler"]["status"], "stale")

    def test_failed_forecast_run_shows_under_forecast_not_scheduler(self):
        self.all_fresh()
        self.heartbeat("scheduler.json", 1, status="failed", interval_min=3, exit_code=1)
        self.answers["wait_forecast_runs"] = (NOW - timedelta(minutes=45),)
        w = self.client.get("/health/workers").json()["workers"]
        self.assertEqual(w["scheduler"]["status"], "ok")
        self.assertEqual(w["scheduler"]["last_run"], "failed")
        self.assertEqual(w["forecast"]["status"], "stale")

    def test_forecast_not_published_for_long_is_stale(self):
        self.all_fresh()
        self.answers["wait_forecast_runs"] = (NOW - timedelta(minutes=30),)
        w = self.client.get("/health/workers").json()["workers"]
        self.assertEqual(w["forecast"]["status"], "stale")
        self.assertEqual(w["forecast"]["max_age_sec"], 600.0)  # 3 runs of 3 min, but at least 10 min

    def test_database_down_is_degraded_without_leaking_the_error(self):
        self.all_fresh()
        self.answers["error"] = RuntimeError("could not connect to server at 10.0.0.5")
        r = self.client.get("/health/workers")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("10.0.0.5", r.text)


class HealthLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_api("api_health_logic_test")
        from iqms_api import health_checks
        cls.hc = health_checks

    def test_limits(self):
        self.assertEqual(self.hc.classify(None, 60), "missing")
        self.assertEqual(self.hc.classify(60, 60), "ok")
        self.assertEqual(self.hc.classify(61, 60), "stale")

    def test_naive_time_is_utc(self):
        now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(self.hc.age_seconds("2026-11-13T11:59:00", now), 60)
        self.assertEqual(self.hc.age_seconds("2026-11-13T12:59:00+01:00", now), 60)

    def test_forecast_limit(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(self.hc.forecast_max_age_sec(3), 600)
            self.assertEqual(self.hc.forecast_max_age_sec(15), 2700)

    def test_newest_migration(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self.hc.expected_schema_version(Path(d)))
            write_migration(d, "0001_a", None)
            write_migration(d, "0002_b", "0001_a")
            write_migration(d, "0003_c", "0002_b")
            self.assertEqual(self.hc.expected_schema_version(Path(d)), "0003_c")
            write_migration(d, "0003_other", "0002_b")  # two branches: no single answer
            self.assertIsNone(self.hc.expected_schema_version(Path(d)))
        self.assertEqual(self.hc.expected_schema_version(Path("db/migrations/versions")), "0001_baseline")


if __name__ == "__main__":
    unittest.main()
