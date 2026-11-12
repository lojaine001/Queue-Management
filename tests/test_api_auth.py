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
import re
import subprocess
import sys
import tempfile
import types
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


def load_api(name='api_auth_test', **env):
    """Load the API through api.py with a test key and build an app with the given
    environment (CORS is read when the app is built). The environment is restored
    afterwards, so other tests in this process (e.g. the DST tests using the real
    DB_PASSWORD) are unaffected."""
    overrides = {'IQMS_API_KEY': TEST_KEY, 'CORS_ALLOWED_ORIGINS': '', **env}
    with mock.patch.dict(os.environ, overrides):
        os.environ.setdefault('DB_PASSWORD', 'test-password')  # never used: no DB in these tests
        spec = importlib.util.spec_from_file_location(name, API_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # api.py: the entry point uvicorn uses
        from iqms_api import db, main, security, settings
        from iqms_api.routes import forecasts
        app = main.create_app()
    # settings.API_KEY is read once per process, so use it rather than assuming TEST_KEY.
    return types.SimpleNamespace(app=app, db=db, security=security, forecasts=forecasts, key=settings.API_KEY)


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
        patcher = mock.patch.object(self.api.db.DB_POOL, 'connection', return_value=self.conn)
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
                r = self.client.post(path, json=body, headers={'X-API-Key': self.api.key})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()['status'], expected[path])
        self.assertTrue(self.conn.committed)

    def test_read_endpoints_work_without_key(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        with mock.patch.dict(os.environ, {'VAPID_PUBLIC_KEY': 'public-test-key'}):
            r = self.client.get('/vapid-public-key')
        self.assertEqual((r.status_code, r.json()['public_key']), (200, 'public-test-key'))
        with mock.patch.object(self.api.forecasts, '_forecast_state', return_value=fresh_state()):
            self.assertEqual(self.client.get('/forecast').status_code, 200)

    def test_every_route_that_changes_data_requires_the_key_and_reads_do_not(self):
        from fastapi.routing import APIRoute
        changing, reading = [], []
        for route in self.api.app.routes:
            if not isinstance(route, APIRoute):
                continue
            guarded = any(d.call is self.api.security.require_api_key for d in route.dependant.dependencies)
            if route.methods & {'POST', 'PUT', 'PATCH', 'DELETE'}:
                changing.append(route.path)
                self.assertTrue(guarded, f'{route.path} changes data but has no API key check')
            else:
                reading.append(route.path)
                self.assertFalse(guarded, f'{route.path} is read-only but requires the key')
        self.assertEqual(sorted(changing), sorted(BODIES))
        self.assertIn('/health', reading)


@unittest.skipIf(TestClient is None, 'fastapi TestClient needs httpx')
class CorsTests(unittest.TestCase):
    ALLOWED = 'https://allowed.example'
    UNKNOWN = 'https://evil.example'

    @classmethod
    def setUpClass(cls):
        try:
            cls.default = TestClient(load_api('api_cors_default').app)  # CORS_ALLOWED_ORIGINS empty
            cls.listed = TestClient(load_api('api_cors_listed', CORS_ALLOWED_ORIGINS=cls.ALLOWED + '/').app)
        except TypeError as exc:
            raise unittest.SkipTest(f'TestClient incompatible with installed httpx ({exc})')

    def preflight(self, client, origin):
        return client.options('/set-lanes', headers={'Origin': origin, 'Access-Control-Request-Method': 'POST',
                                                     'Access-Control-Request-Headers': 'content-type,x-api-key'})

    def test_unknown_origin_is_rejected(self):
        for client in (self.default, self.listed):
            r = self.preflight(client, self.UNKNOWN)
            self.assertEqual(r.status_code, 400)
            self.assertNotIn('access-control-allow-origin', r.headers)
            r = client.get('/health', headers={'Origin': self.UNKNOWN})
            self.assertNotIn('access-control-allow-origin', r.headers)  # browser will not hand it to the page

    def test_listed_origin_is_allowed_with_only_the_needed_methods_and_headers(self):
        r = self.preflight(self.listed, self.ALLOWED)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers['access-control-allow-origin'], self.ALLOWED)
        self.assertNotIn('*', r.headers['access-control-allow-origin'])
        self.assertNotIn('DELETE', r.headers['access-control-allow-methods'])
        self.assertNotIn('access-control-allow-credentials', r.headers)

    def test_same_origin_requests_are_unaffected(self):
        # The web app: same address as /api through Caddy, so the browser sends no preflight.
        self.assertEqual(self.default.get('/health').status_code, 200)


class StartupTests(unittest.TestCase):
    CODE = ('import importlib.util,sys; s=importlib.util.spec_from_file_location("a", sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)')

    def start(self, path, **env_changes):
        """Load a module in a fresh process from a folder with no .env above it."""
        env = {k: v for k, v in os.environ.items() if k not in ('IQMS_API_KEY', 'DB_PASSWORD', 'CORS_ALLOWED_ORIGINS')}
        env.update(PYTHONPATH=str(ROOT), **env_changes)
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run([sys.executable, '-c', self.CODE, str(path)], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=120)

    def test_api_refuses_to_start_without_key(self):
        r = self.start(API_PATH, DB_PASSWORD='x')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('IQMS_API_KEY is not set', r.stderr)

    def test_api_refuses_to_start_without_db_password(self):
        for value in ('', 'change-me'):
            r = self.start(API_PATH, IQMS_API_KEY='x', DB_PASSWORD=value)
            self.assertNotEqual(r.returncode, 0, value)
            self.assertIn('DB_PASSWORD is not set', r.stderr)

    def test_api_refuses_wildcard_cors(self):
        r = self.start(API_PATH, IQMS_API_KEY='x', DB_PASSWORD='x', CORS_ALLOWED_ORIGINS='*')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('not \'*\'', r.stderr)

    def test_api_starts_with_all_secrets(self):
        r = self.start(API_PATH, IQMS_API_KEY='x', DB_PASSWORD='x')
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


class DefaultCredentialTests(unittest.TestCase):
    """No password is built into the code any more; services stop clearly without one."""

    def test_no_default_db_password_anywhere_in_python(self):
        # Only the code in the repository: a machine can hold other, untracked scripts.
        listed = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '*.py'], capture_output=True, text=True)
        if listed.returncode != 0:
            self.skipTest('not a git checkout')
        offenders = []
        for f in (ROOT / p for p in listed.stdout.splitlines()):
            if f.name == Path(__file__).name or not f.exists():
                continue
            for n, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
                if 'DB_PASSWORD' in line and 'getenv' in line and not re.search(r"""getenv\(\s*["']DB_PASSWORD["']\s*(,\s*["']{2}\s*)?\)""", line):
                    offenders.append(f'{f.relative_to(ROOT)}:{n}')
        self.assertEqual(offenders, [], 'DB_PASSWORD read with a non-empty default')

    def test_no_camera_password_fallback_in_code(self):
        for rel in ('Head-Detector/main.py', 'Queue-Management-System-v2-main/Queue-Management-System-v2-main/queue_management_v2.py'):
            text = (ROOT / rel).read_text(encoding='utf-8')
            for m in re.finditer(r"get\('password',\s*'([^']*)'\)", text):
                self.assertEqual(m.group(1), '', f'{rel} has a built-in camera password')

    def test_camera_db_loggers_stop_without_db_password(self):
        for rel in ('Head-Detector/utils/db_logger.py',
                    'Queue-Management-System-v2-main/Queue-Management-System-v2-main/utils/db_logger.py'):
            spec = importlib.util.spec_from_file_location('dblogger_' + rel.split('/')[0].replace('-', '_'), ROOT / rel)
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(os.environ, {'DB_PASSWORD': ''}):
                spec.loader.exec_module(module)
                if getattr(module, 'psycopg2', True) is None:
                    continue  # psycopg2 not installed in this venv: the logger never connects
                with mock.patch.object(module.psycopg2, 'connect') as connect, \
                        self.assertRaisesRegex(RuntimeError, 'DB_PASSWORD is not set'):
                    module.DBLogger()
                connect.assert_not_called()


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
