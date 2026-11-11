"""
API key on the endpoints that change data, and the production web setup.

- every POST without the key (or with a wrong key) gets 401 and never touches the database
- every POST with the key works
- read-only (GET) endpoints still work without a key
- every route that changes data has the key check (also catches future endpoints)
- the API refuses to start without IQMS_API_KEY
- Caddy adds the key only for the two push calls, listens on this machine only,
  and the web app source never contains the key; the Vite dev server is not exposed

No real database or key is used: the database is a fake object and the key is a
test value set before api.py is loaded (so .env is never read for it).

Run:  python -m unittest tests.test_api_auth -v
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main/api.py'
WEB = ROOT / 'IQMS-Web'
TEST_KEY = 'test-key-for-unit-tests'

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - httpx missing
    TestClient = None


def load_api():
    os.environ['IQMS_API_KEY'] = TEST_KEY
    spec = importlib.util.spec_from_file_location('api_auth_test', API_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # as uvicorn does; pydantic resolves the request models through it
    spec.loader.exec_module(module)
    return module


class FakeCursor:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, args=None): self.conn.statements.append(sql)
    def fetchone(self): return self.conn.state


class FakeConnection:
    def __init__(self, state=None): self.state = state; self.statements = []; self.committed = False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self, *a, **k): return FakeCursor(self)
    def commit(self): self.committed = True
    def close(self): pass


def fresh_state():
    scenario = {'wait_0m': 1.0, 'wait_5m': 1.5, 'wait_10m': 2.0, 'wait_15m': 2.5}
    return {'open_lanes': 3, 'forecast_json': {'as_of': datetime.now(timezone.utc).isoformat(),
                                               'lanes': {'2': scenario}}}


BODIES = {
    '/set-lanes': {'lanes': 2},
    '/alert-response': {'response': 'opening_lane', 'lane_id': '1'},
    '/push-subscribe': {'endpoint': 'https://push.example/abc', 'keys': {'p256dh': 'p', 'auth': 'a'},
                        'threshold_min': 5, 'horizon_min': 10},
    '/push-unsubscribe': {'endpoint': 'https://push.example/abc'},
}


@unittest.skipIf(TestClient is None, 'fastapi TestClient needs httpx')
class ApiKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = load_api()
        try:
            cls.client = TestClient(cls.api.app)
        except TypeError as exc:  # FastAPI/starlette older than the pinned versions with a newer httpx
            raise unittest.SkipTest(f'TestClient incompatible with installed httpx ({exc}); '
                                    'install the pinned requirements')

    def setUp(self):
        self.conn = FakeConnection(fresh_state())
        patcher = mock.patch.object(self.api, '_conn', return_value=self.conn)
        self.db = patcher.start()
        self.addCleanup(patcher.stop)

    def test_post_without_key_is_rejected_and_database_untouched(self):
        for path, body in BODIES.items():
            with self.subTest(path=path):
                r = self.client.post(path, json=body)
                self.assertEqual(r.status_code, 401, r.text)
        self.db.assert_not_called()

    def test_post_with_wrong_key_is_rejected(self):
        for path, body in BODIES.items():
            with self.subTest(path=path):
                r = self.client.post(path, json=body, headers={'X-API-Key': 'wrong'})
                self.assertEqual(r.status_code, 401, r.text)
        self.db.assert_not_called()

    def test_post_with_key_works(self):
        expected = {'/set-lanes': 'ok', '/alert-response': 'recorded',
                    '/push-subscribe': 'subscribed', '/push-unsubscribe': 'unsubscribed'}
        for path, body in BODIES.items():
            with self.subTest(path=path):
                r = self.client.post(path, json=body, headers={'X-API-Key': TEST_KEY})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()['status'], expected[path])
        self.assertTrue(self.conn.committed)

    def test_read_endpoints_work_without_key(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        with mock.patch.dict(os.environ, {'VAPID_PUBLIC_KEY': 'public-test-key'}):
            r = self.client.get('/vapid-public-key')
        self.assertEqual((r.status_code, r.json()['public_key']), (200, 'public-test-key'))
        with mock.patch.object(self.api, '_forecast_state', return_value=fresh_state()):
            self.assertEqual(self.client.get('/forecast').status_code, 200)

    def test_every_route_that_changes_data_requires_the_key_and_reads_do_not(self):
        from fastapi.routing import APIRoute
        changing, reading = [], []
        for route in self.api.app.routes:
            if not isinstance(route, APIRoute):
                continue
            guarded = any(d.call is self.api.require_api_key for d in route.dependant.dependencies)
            if route.methods & {'POST', 'PUT', 'PATCH', 'DELETE'}:
                changing.append(route.path)
                self.assertTrue(guarded, f'{route.path} changes data but has no API key check')
            else:
                reading.append(route.path)
                self.assertFalse(guarded, f'{route.path} is read-only but requires the key')
        self.assertEqual(sorted(changing), sorted(BODIES))
        self.assertIn('/health', reading)


class StartupTests(unittest.TestCase):
    def test_api_refuses_to_start_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != 'IQMS_API_KEY'}
        env['PYTHONPATH'] = str(ROOT)
        code = ('import importlib.util,sys; s=importlib.util.spec_from_file_location("a", sys.argv[1]); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)')
        with tempfile.TemporaryDirectory() as tmp:  # no .env in this folder or above it
            r = subprocess.run([sys.executable, '-c', code, str(API_PATH)], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('IQMS_API_KEY is not set', r.stderr)


class WebServingTests(unittest.TestCase):
    def setUp(self):
        self.caddyfile = (WEB / 'Caddyfile').read_text(encoding='utf-8')

    def test_caddy_adds_the_key_only_to_the_push_calls(self):
        self.assertIn('@push path /push-subscribe /push-unsubscribe', self.caddyfile)
        self.assertIn('request_header @push X-API-Key {env.IQMS_API_KEY}', self.caddyfile)
        config_lines = [l for l in self.caddyfile.splitlines() if not l.lstrip().startswith('#')]
        self.assertEqual(sum('X-API-Key' in l for l in config_lines), 1)  # nowhere else

    def test_caddy_listens_on_this_machine_only_and_serves_the_build(self):
        self.assertIn('bind 127.0.0.1 [::1]', self.caddyfile)
        self.assertIn('root * dist', self.caddyfile)
        self.assertIn('reverse_proxy 127.0.0.1:8000', self.caddyfile)

    def test_web_app_source_never_contains_the_key(self):
        for f in (WEB / 'src').rglob('*'):
            if f.suffix in {'.js', '.jsx', '.ts', '.tsx'}:
                text = f.read_text(encoding='utf-8')
                self.assertNotIn('IQMS_API_KEY', text, f)
                self.assertNotIn('X-API-Key', text, f)

    def test_vite_dev_server_is_not_exposed_or_started(self):
        vite = (WEB / 'vite.config.js').read_text(encoding='utf-8')
        self.assertNotIn('host: true', vite)
        self.assertNotIn('allowedHosts: true', vite)
        start = (ROOT / 'start_all.ps1').read_text(encoding='utf-8')
        self.assertNotIn('npm run dev', start)
        self.assertIn('run --config Caddyfile', start)

    @unittest.skipUnless((ROOT / 'tools/bin/caddy.exe').exists(), 'Caddy not installed (tools\\install.ps1)')
    def test_caddyfile_is_valid(self):
        r = subprocess.run([str(ROOT / 'tools/bin/caddy.exe'), 'validate', '--config', 'Caddyfile',
                            '--adapter', 'caddyfile'], cwd=WEB, capture_output=True, text=True,
                           env={**os.environ, 'IQMS_API_KEY': 'x'}, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])


if __name__ == '__main__':
    unittest.main()
