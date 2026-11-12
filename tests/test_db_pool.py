"""
The API's database connection pool (Queue-Management-System-v2-main/.../iqms_api/db_pool.py).

[DB] tests use a real PostgreSQL from .env (TEMP tables only, nothing written to
real tables) and are skipped without a database:
- connections are reused, never more than the maximum, even under 30 parallel requests
- each connection has the store time zone and the query time limit
- an error rolls back and gives the connection back
- a full pool refuses after the wait (PoolBusy), instead of hanging
- a broken connection is replaced
API tests (no database): a busy pool answers 503 + Retry-After, /health shows the pool.

Run:  python -m unittest tests.test_db_pool -v
"""
import importlib.util
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
QM2 = ROOT / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main'
sys.path.insert(0, str(QM2))
from iqms_api.db_pool import DbPool, PoolBusy  # noqa: E402

try:
    import psycopg2
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover
    psycopg2 = None


def conn_kwargs():
    return dict(host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', 5432)),
                user=os.getenv('DB_USER', 'postgres'), password=os.getenv('DB_PASSWORD', ''),
                dbname=os.getenv('DB_NAME', 'iqms'), connect_timeout=5)


@unittest.skipIf(psycopg2 is None, 'psycopg2 not installed')
class PoolWithDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            psycopg2.connect(**conn_kwargs()).close()
        except Exception as exc:
            raise unittest.SkipTest(f'no PostgreSQL: {exc}')

    def make(self, **kw):
        options = {'minconn': 1, 'maxconn': 5, 'wait_sec': 5.0,
                   'session_options': '-c timezone=Europe/Paris -c statement_timeout=15000', **kw}
        pool = DbPool(conn_kwargs(), **options)
        self.addCleanup(pool.close)
        return pool

    @staticmethod
    def pid(conn):
        with conn.cursor() as cur:
            cur.execute('SELECT pg_backend_pid()')
            return cur.fetchone()[0]

    def test_sequential_requests_reuse_one_connection(self):
        pool = self.make()
        pids = set()
        for _ in range(50):
            with pool.connection() as conn:
                pids.add(self.pid(conn))
        self.assertEqual(len(pids), 1)
        self.assertEqual(pool.stats()['in_use'], 0)

    def test_parallel_requests_never_exceed_the_maximum(self):
        # minconn == maxconn: psycopg2 keeps only minconn connections open between requests.
        pool = self.make(minconn=5, maxconn=5)
        pids, peak, errors, lock = set(), [0], [], threading.Lock()

        def request():
            try:
                with pool.connection() as conn:
                    with lock:
                        pids.add(self.pid(conn))
                        peak[0] = max(peak[0], pool.stats()['in_use'])
                    time.sleep(0.1)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=request) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(pids), 5)
        self.assertLessEqual(peak[0], 5)
        self.assertEqual(pool.stats()['in_use'], 0)

    def test_session_has_store_time_zone_and_query_limit(self):
        pool = self.make()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute('SHOW timezone')
            self.assertEqual(cur.fetchone()[0], 'Europe/Paris')
            cur.execute('SHOW statement_timeout')
            self.assertEqual(cur.fetchone()[0], '15s')

    def test_slow_query_is_stopped_and_connection_stays_usable(self):
        pool = self.make(maxconn=1, session_options='-c statement_timeout=200')
        with self.assertRaises(psycopg2.errors.QueryCanceled):
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute('SELECT pg_sleep(2)')
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT 1')
            self.assertEqual(cur.fetchone()[0], 1)

    def test_error_rolls_back_and_returns_the_connection(self):
        pool = self.make(maxconn=1)
        with self.assertRaises(ValueError):
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute('CREATE TEMP TABLE pool_probe (x int)')
                cur.execute('INSERT INTO pool_probe VALUES (1)')
                raise ValueError('request failed')
        self.assertEqual(pool.stats()['in_use'], 0)
        with pool.connection() as conn, conn.cursor() as cur:  # same single connection
            cur.execute("SELECT to_regclass('pg_temp.pool_probe')")
            self.assertIsNone(cur.fetchone()[0], 'the failed request was not rolled back')

    def test_success_is_committed(self):
        pool = self.make(maxconn=1)
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute('CREATE TEMP TABLE pool_commit (x int)')
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('pg_temp.pool_commit') IS NOT NULL")
            self.assertTrue(cur.fetchone()[0])

    def test_full_pool_refuses_after_the_wait_instead_of_hanging(self):
        pool = self.make(maxconn=1, wait_sec=0.3)
        held, release = threading.Event(), threading.Event()

        def hold():
            with pool.connection():
                held.set()
                release.wait(10)

        t = threading.Thread(target=hold)
        t.start()
        try:
            self.assertTrue(held.wait(10))
            started = time.monotonic()
            with self.assertRaises(PoolBusy):
                with pool.connection():
                    pass
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertEqual(pool.stats()['busy_rejections'], 1)
        finally:
            release.set()
            t.join(10)
        with pool.connection():
            pass  # free again

    def test_bursts_above_the_minimum_stay_within_the_maximum(self):
        pool = self.make(minconn=1, maxconn=3)
        peak, lock = [0], threading.Lock()

        def request():
            with pool.connection():
                with lock:
                    peak[0] = max(peak[0], pool.stats()['in_use'])
                time.sleep(0.1)

        threads = [threading.Thread(target=request) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertLessEqual(peak[0], 3)
        self.assertEqual(pool.stats()['in_use'], 0)

    def test_connection_dropped_by_the_server_is_replaced(self):
        pool = self.make(maxconn=1)
        with pool.connection() as conn:
            first = self.pid(conn)
        killer = psycopg2.connect(**conn_kwargs())
        killer.autocommit = True
        with killer.cursor() as cur:  # e.g. a PostgreSQL restart
            cur.execute('SELECT pg_terminate_backend(%s)', (first,))
        killer.close()
        # The one request that meets the dead connection fails, like any query would ...
        with self.assertRaises((psycopg2.OperationalError, psycopg2.InterfaceError)):
            with pool.connection() as conn:
                self.pid(conn)
        self.assertEqual(pool.stats()['replaced'], 1)
        # ... and the next one gets a fresh connection.
        with pool.connection() as conn:
            self.assertNotEqual(self.pid(conn), first)


class PoolWithoutDatabase(unittest.TestCase):
    def test_import_does_not_connect(self):
        pool = DbPool({'host': '127.0.0.1', 'port': 1, 'connect_timeout': 1}, minconn=1, maxconn=2)
        self.assertFalse(pool.stats()['open'])

    def test_unreachable_database_frees_the_slot(self):
        pool = DbPool({'host': '127.0.0.1', 'port': 1, 'connect_timeout': 1}, minconn=1, maxconn=1, wait_sec=0.2)
        for _ in range(3):  # would raise PoolBusy on the 2nd try if the slot leaked
            with self.assertRaises(psycopg2.OperationalError if psycopg2 else Exception):
                with pool.connection():
                    pass
        self.assertEqual(pool.stats()['in_use'], 0)

    def test_invalid_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            DbPool({}, minconn=3, maxconn=2)


class ApiUsesThePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            raise unittest.SkipTest('fastapi TestClient needs httpx')
        with mock.patch.dict(os.environ, {'IQMS_API_KEY': 'test-key', 'CORS_ALLOWED_ORIGINS': ''}):
            os.environ.setdefault('DB_PASSWORD', 'test-password')
            spec = importlib.util.spec_from_file_location('api_pool_test', QM2 / 'api.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            from iqms_api import db, main, settings
            cls.db, cls.key, cls.app = db, settings.API_KEY, main.create_app()
        try:
            cls.client = TestClient(cls.app)
        except TypeError as exc:
            raise unittest.SkipTest(f'TestClient incompatible with installed httpx ({exc})')

    def busy(self):
        return mock.patch.object(self.db.DB_POOL, 'connection', side_effect=PoolBusy('all busy'))

    def test_busy_pool_gives_503_on_read_endpoints(self):
        with self.busy():
            r = self.client.get('/statistics/summary')
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get('retry-after'), '5')

    def test_busy_pool_gives_503_not_500_on_endpoints_that_wrap_errors(self):
        with self.busy():
            r = self.client.post('/alert-response', json={'response': 'false_alarm'}, headers={'X-API-Key': self.key})
        self.assertEqual(r.status_code, 503)

    def test_health_shows_the_pool(self):
        body = self.client.get('/health').json()
        self.assertEqual(set(body['db_pool']), {'in_use', 'max', 'waiting', 'busy_rejections', 'replaced', 'open'})
        self.assertEqual(body['db_pool']['max'], int(os.getenv('DB_POOL_MAX', 10)))

    def test_api_has_no_unpooled_connections_left(self):
        files = [QM2 / 'api.py'] + [f for f in (QM2 / 'iqms_api').rglob('*.py') if f.name != 'db_pool.py']
        for f in files:
            source = f.read_text(encoding='utf-8')
            self.assertNotIn('psycopg2.connect(', source, f)
            self.assertNotIn('conn.close()', source, f)


if __name__ == '__main__':
    unittest.main()
