"""
Daylight-saving tests for the store time zone (Europe/Paris).

  Spring: Sunday 29 March 2026, 02:00 -> 03:00, a 23-hour day.
  Autumn: Sunday 25 October 2026, 03:00 -> 02:00, a 25-hour day (02:00-03:00 twice).
          The forecast stopped on this date (mixed +02/+01 offsets in one window).

Covered: camera timestamp writes, hourly/3-minute buckets, forecast training
data (DB -> local wall time -> hour/day features, closed-hour fill, future
slots), forecast publication (local slot -> UTC row, shared payload, horizons)
and the scheduler (interval timing, heartbeat age).

Tests marked [DB] use a real PostgreSQL connection from .env with TEMP tables
only (nothing is written to real tables) and are skipped without a database.
The laptop has no TimescaleDB, so time_bucket() is reproduced with date_bin()
(same UTC-aligned bins for timestamptz).

Run:  python -m unittest tests.test_dst_transitions -v
"""
import ast
import os
import types
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QM2 = ROOT / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main'
STORE_TZ = 'Europe/Paris'
PARIS = ZoneInfo(STORE_TZ)
UTC = timezone.utc
os.environ.setdefault('STORE_TZ', STORE_TZ)

SPRING = date(2026, 3, 29)
AUTUMN = date(2026, 10, 25)

try:
    import psycopg2
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover
    psycopg2 = None


def local(d, hour, minute=0, fold=0):
    """Aware store-local datetime (fold=1 = second pass of the repeated hour)."""
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=PARIS, fold=fold)


def utc(d, hour, minute=0):
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)


def store_day_instants(d, step_min):
    """Every step_min minutes of the store-local calendar day d, as UTC instants."""
    start = datetime(d.year, d.month, d.day, tzinfo=PARIS).astimezone(UTC)
    nxt = d + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=PARIS).astimezone(UTC)
    out, t = [], start
    while t < end:
        out.append(t)
        t += timedelta(minutes=step_min)
    return out


def load_function(path, name, namespace):
    """Compile one top-level function from a module that cannot be imported in tests."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def module_assignment(path, name, namespace):
    """Evaluate one module-level assignment exactly as written in the source."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def ensemble_converter():
    return load_function(QM2 / 'ensemble_predict.py', '_to_local_wall_series',
                         {'pd': pd, 'datetime': datetime, 'timezone': timezone, 'STORE_TZ': STORE_TZ,
                          'ZoneInfo': ZoneInfo, 'os': os})


def machine_is_paris():
    return (datetime(2026, 7, 1, tzinfo=UTC).astimezone().utcoffset() == timedelta(hours=2)
            and datetime(2026, 1, 1, tzinfo=UTC).astimezone().utcoffset() == timedelta(hours=1))


def fixed_datetime(now_utc):
    """datetime class whose now() returns now_utc (aware with tz, naive local without)."""
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return now_utc.astimezone(PARIS).replace(tzinfo=None)
            return now_utc.astimezone(tz)
    return Fixed


def pandas_at(now_local_naive):
    """A stand-in for the pandas module whose Timestamp.now() is fixed (store-local, naive)."""
    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            ts = pd.Timestamp(now_local_naive)
            return ts.tz_localize(PARIS).tz_convert(tz) if tz else ts
    proxy = types.ModuleType('pandas_fixed')
    proxy.__getattr__ = lambda attr: getattr(pd, attr)
    proxy.Timestamp = FixedTimestamp
    return proxy


# --------------------------------------------------------------------------- calendar
class CalendarFacts(unittest.TestCase):
    def test_store_days_are_23_and_25_hours(self):
        self.assertEqual(len(store_day_instants(SPRING, 60)), 23)
        self.assertEqual(len(store_day_instants(AUTUMN, 60)), 25)
        self.assertEqual(len(store_day_instants(date(2026, 11, 2), 60)), 24)


# --------------------------------------------------------------------------- database
@unittest.skipIf(psycopg2 is None, 'psycopg2 not installed')
class DatabaseCase(unittest.TestCase):
    """[DB] Real PostgreSQL session set up like the camera workers / forecast (SET timezone)."""

    @classmethod
    def connect(cls, set_tz=True):
        kw = dict(host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', 5432)),
                  user=os.getenv('DB_USER', 'postgres'), password=os.getenv('DB_PASSWORD', ''),
                  dbname=os.getenv('DB_NAME', 'iqms'), connect_timeout=5)
        try:
            conn = psycopg2.connect(**kw)
        except psycopg2.Error:
            kw['dbname'] = 'postgres'
            conn = psycopg2.connect(**kw)
        conn.autocommit = True
        if set_tz:
            conn.cursor().execute('SET timezone = %s', (STORE_TZ,))
        return conn

    @classmethod
    def setUpClass(cls):
        try:
            cls.conn = cls.connect()
        except Exception as exc:  # no database on this machine
            raise unittest.SkipTest(f'no PostgreSQL: {exc}')
        cls.cur = cls.conn.cursor()
        cls.cur.execute('CREATE TEMP TABLE ev (ts TIMESTAMPTZ NOT NULL)')

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def load(self, values):
        self.cur.execute('TRUNCATE ev')
        self.cur.executemany('INSERT INTO ev VALUES (%s)', [(v,) for v in values])

    def read_back(self, instants, column='bucket'):
        """What pd.read_sql hands the forecast for these instants (session time zone = STORE_TZ)."""
        self.load(instants)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # pandas warns about raw DBAPI connections
            return pd.read_sql(f'SELECT ts AS {column} FROM ev ORDER BY ts', self.conn)[column]


class CameraWrites(DatabaseCase):
    """Camera workers insert entrance timestamps as naive local datetimes (datetime.now() /
    datetime.fromtimestamp()); PostgreSQL resolves them in the session time zone."""

    def stored(self, naive):
        self.load([naive])
        self.cur.execute('SELECT ts FROM ev')
        return self.cur.fetchone()[0].astimezone(UTC)

    def test_spring_times_around_the_gap(self):
        self.assertEqual(self.stored(datetime(2026, 3, 29, 1, 59)), utc(SPRING, 0, 59))
        self.assertEqual(self.stored(datetime(2026, 3, 29, 3, 0)), utc(SPRING, 1, 0))

    def test_autumn_times_outside_the_repeated_hour(self):
        self.assertEqual(self.stored(datetime(2026, 10, 25, 1, 59)), datetime(2026, 10, 24, 23, 59, tzinfo=UTC))
        self.assertEqual(self.stored(datetime(2026, 10, 25, 3, 0)), utc(AUTUMN, 2, 0))

    @unittest.expectedFailure
    def test_autumn_first_pass_of_repeated_hour(self):
        """KNOWN LIMITATION (store closed at that hour): a person entering at the FIRST 02:30
        (00:30 UTC) is stored as the second 02:30 (01:30 UTC), because the cameras pass a naive
        local time and PostgreSQL picks winter time. Fix if ever needed: pass aware UTC
        datetimes from the cameras (see test_aware_utc_write_is_exact_in_repeated_hour)."""
        self.assertEqual(self.stored(datetime(2026, 10, 25, 2, 30)), utc(AUTUMN, 0, 30))

    def test_aware_utc_write_is_exact_in_repeated_hour(self):
        """The alternative: passing an aware UTC datetime stores both 02:30s correctly."""
        self.load([utc(AUTUMN, 0, 30), utc(AUTUMN, 1, 30)])
        self.cur.execute('SELECT ts FROM ev ORDER BY ts')
        self.assertEqual([r[0].astimezone(UTC) for r in self.cur.fetchall()],
                         [utc(AUTUMN, 0, 30), utc(AUTUMN, 1, 30)])


class HourlyBuckets(DatabaseCase):
    API_HOURLY = """
        SELECT DATE_TRUNC('hour', ts AT TIME ZONE %s) AS bucket, COUNT(*) AS n
        FROM ev WHERE ts AT TIME ZONE %s >= %s AND ts AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1"""   # same pattern as api.py summary/report queries

    def hourly(self, d):
        self.load(store_day_instants(d, 15))
        self.cur.execute(self.API_HOURLY, (STORE_TZ, STORE_TZ, d, STORE_TZ, d + timedelta(days=1)))
        return {row[0].hour: row[1] for row in self.cur.fetchall()}

    def test_spring_day_has_23_hourly_buckets_and_no_0200(self):
        buckets = self.hourly(SPRING)
        self.assertEqual(len(buckets), 23)
        self.assertNotIn(2, buckets)
        self.assertEqual(set(buckets.values()), {4})
        self.assertEqual(sum(buckets.values()), 92)

    def test_autumn_day_has_24_labels_and_0200_holds_both_passes(self):
        buckets = self.hourly(AUTUMN)
        self.assertEqual(len(buckets), 24)
        self.assertEqual(buckets[2], 8)
        self.assertEqual(sum(buckets.values()), 100)

    def test_three_minute_buckets_stay_distinct_through_both_changes(self):
        # ensemble_predict uses time_bucket('3 minutes', timestamp) on timestamptz (UTC-aligned bins).
        for d, expected in ((SPRING, 23 * 20), (AUTUMN, 25 * 20)):
            self.load(store_day_instants(d, 3))
            self.cur.execute("SELECT COUNT(DISTINCT date_bin('3 minutes', ts, '2000-01-03 00:00+00')) FROM ev")
            self.assertEqual(self.cur.fetchone()[0], expected, d)

    def test_last_hour_window_is_60_real_minutes_across_change(self):
        # Rolling windows use timestamp >= NOW() - INTERVAL 'N hours' (absolute time).
        self.load(store_day_instants(AUTUMN, 1))
        self.cur.execute("SELECT COUNT(*) FROM ev WHERE ts > %s::timestamptz - INTERVAL '1 hour' AND ts <= %s",
                         (utc(AUTUMN, 1, 30), utc(AUTUMN, 1, 30)))
        self.assertEqual(self.cur.fetchone()[0], 60)


# --------------------------------------------------------------------------- training
class ForecastTrainingData(DatabaseCase):
    """DB rows -> local wall-clock training frame (ensemble_predict / prediction helpers)."""

    def daily_at(self, first, last, hour):
        d, out = first, []
        while d <= last:
            out.append(local(d, hour).astimezone(UTC))
            d += timedelta(days=1)
        return out

    def check(self, convert, rows, hour, label):
        out = convert(rows)
        self.assertEqual(sorted(set(out.dt.hour)), [hour], f'{label}: hours {list(out.dt.strftime("%m-%d %H:%M"))}')

    def test_25_oct_2026_window_reproduces_the_original_failure(self):
        """The production error: a 30-day window across 25 Oct comes back with mixed offsets."""
        rows = self.read_back(self.daily_at(date(2026, 10, 20), date(2026, 10, 30), 9))
        self.assertEqual(rows.dtype, object)
        with self.assertRaises(ValueError):
            pd.to_datetime(rows)  # what the code did before the fix

    def test_ensemble_history_across_autumn_change(self):
        rows = self.read_back(self.daily_at(date(2026, 10, 20), date(2026, 10, 30), 9))
        self.check(ensemble_converter(), rows, 9, 'autumn')

    def test_ensemble_history_across_spring_change(self):
        rows = self.read_back(self.daily_at(date(2026, 3, 25), date(2026, 4, 2), 9))
        self.check(ensemble_converter(), rows, 9, 'spring')

    def test_ensemble_history_without_change(self):
        rows = self.read_back(self.daily_at(date(2026, 11, 2), date(2026, 11, 6), 9))
        self.check(ensemble_converter(), rows, 9, 'no change')

    def test_ensemble_features_hour_and_weekday_on_change_days(self):
        instants = [local(date(2026, 10, 24), 18), local(AUTUMN, 10), local(date(2026, 10, 26), 9),
                    local(date(2026, 3, 28), 18), local(SPRING, 10), local(date(2026, 3, 30), 9)]
        out = ensemble_converter()(self.read_back([i.astimezone(UTC) for i in instants]))
        got = [(t.date(), t.hour, t.dayofweek) for t in out]
        want = [(i.date(), i.hour, i.weekday()) for i in sorted(instants, key=lambda x: x.astimezone(UTC))]
        self.assertEqual(got, want)

    def test_pipeline_to_local_across_changes(self):
        from prediction.pipeline import to_local
        for first, last in ((date(2026, 10, 20), date(2026, 10, 30)), (date(2026, 3, 25), date(2026, 4, 2))):
            self.check(to_local, self.read_back(self.daily_at(first, last, 9)), 9, f'pipeline {first}')

    def test_dwell_models_across_changes(self):
        from prediction.dwell_models import _to_local_series
        for first, last in ((date(2026, 10, 20), date(2026, 10, 30)), (date(2026, 3, 25), date(2026, 4, 2))):
            self.check(_to_local_series, self.read_back(self.daily_at(first, last, 9)), 9, f'dwell {first}')

    def test_repeated_hour_rows_share_one_wall_clock_label(self):
        """Two night rows 1 hour apart (00:30Z and 01:30Z) both read 02:30 locally. The store is
        closed then; the forecast's closed-hour fill replaces these rows with zeros anyway."""
        out = ensemble_converter()(self.read_back([utc(AUTUMN, 0, 30), utc(AUTUMN, 1, 30)]))
        self.assertEqual(list(out.dt.strftime('%H:%M')), ['02:30', '02:30'])

    @unittest.skipUnless(machine_is_paris(), 'machine clock is not Europe/Paris')
    def test_dashboard_history_across_changes(self):
        ns = {'datetime': datetime, 'timezone': timezone, 'ZoneInfo': ZoneInfo, 'os': os, 'pd': pd}
        module_assignment(QM2 / 'dashboard.py', 'LOCAL_TZ', ns)
        convert = load_function(QM2 / 'dashboard.py', '_to_local_series', ns)
        for first, last in ((date(2026, 10, 20), date(2026, 10, 30)), (date(2026, 3, 25), date(2026, 4, 2))):
            self.check(convert, self.read_back(self.daily_at(first, last, 9)), 9, f'dashboard {first}')


class ForecastTrainingCalendar(unittest.TestCase):
    """Closed-hour fill and future slots are built in store-local wall time (prediction.core)."""

    def run_at(self, now_local, fn, *args):
        import prediction.core as core
        with mock.patch.object(core, 'pd', pandas_at(now_local)):
            return fn(*args)

    def test_future_slots_on_change_days_are_open_hours_and_unambiguous(self):
        import prediction.core as core
        for d in (SPRING, AUTUMN):
            for hour in (0, 2, 7, 12):
                now = datetime(d.year, d.month, d.day, hour, 1)
                slots = self.run_at(now, core.future_open_timestamps)
                self.assertTrue(slots, (d, hour))
                for ts in slots:
                    self.assertTrue(core.is_open(ts), ts)
                    ts.tz_localize(STORE_TZ, ambiguous='raise', nonexistent='raise')  # must not raise
                gaps = {(b - a) for a, b in zip(slots, slots[1:])}
                self.assertEqual(gaps, {pd.Timedelta(minutes=core.BUCKET_MINUTES)}, (d, hour))

    def test_closed_hour_fill_over_change_has_no_duplicate_slots(self):
        import prediction.core as core
        for d in (SPRING, AUTUMN):
            now = datetime(d.year, d.month, d.day, 12, 0) + timedelta(days=1)
            open_row = pd.DataFrame({'ds': [pd.Timestamp(d) + pd.Timedelta(hours=10)], 'y': [5]})
            filled = self.run_at(now, core.add_closed_zeros, open_row, 3)
            self.assertFalse(filled['ds'].duplicated().any())
            self.assertTrue(filled['ds'].is_monotonic_increasing)
            self.assertEqual(filled.loc[filled['ds'] == pd.Timestamp(d) + pd.Timedelta(hours=10), 'y'].item(), 5)


# --------------------------------------------------------------------------- publication
class ForecastPublication(unittest.TestCase):
    """Local forecast slots -> UTC rows (queue_predictions) and the shared wait payload."""

    def save_to_db(self, ds_local, now_utc):
        import tempfile
        from prediction.runtime import write_json
        from prediction.forecast_state import status_for
        statements = []

        class Conn:
            def cursor(self): return self
            def execute(self, sql, args=None): statements.append((sql, args))
            def commit(self): pass
            def rollback(self): pass
            def close(self): pass

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = {'datetime': fixed_datetime(now_utc), 'timezone': timezone, 'ZoneInfo': ZoneInfo,
               'STORE_TZ': STORE_TZ, 'pd': pd, 'DB_CONFIG': {}, 'BUCKET_MINUTES': 3,
               'RUNTIME': Path(tmp.name), 'write_json': write_json, 'status_for': status_for,
               '_connect': lambda: Conn()}
        load_function(QM2 / 'ensemble_predict.py', '_save_to_db', env)
        n = len(ds_local)
        vals = [1.0] * n
        result = {'prophet_preds': pd.DataFrame({'ds': ds_local}), 'prophet_vals': vals, 'lstm_vals': vals,
                  'xgb_vals': vals, 'ensemble_vals': vals, 'checkout_arrivals': vals, 'wait_estimates': vals,
                  'lane_waits_15m': {1: 3, 2: 2, 3: 1}, 'source': 'REAL', 'run_id': 'dst', 'model_version': 'dst',
                  'data_cutoff': now_utc.isoformat(), 'forecast_origin': now_utc.isoformat(),
                  'run_started_at': now_utc.isoformat(), 'service_per_bucket': 3, 'active_lanes': 3,
                  'browsing_gap_min': 25, 'wait_15m': 1.0, 'wait_30m': 1.0}
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            env['_save_to_db'](result)
        return [args[1] for sql, args in statements if 'INSERT INTO queue_predictions' in sql]

    def test_published_rows_get_the_right_utc_instant_on_change_days(self):
        for d, offset in ((SPRING, 2), (AUTUMN, 1)):
            ds = [pd.Timestamp(datetime(d.year, d.month, d.day, 10, m)) for m in (0, 3, 6)]
            now = local(d, 9, 55).astimezone(UTC)
            rows = self.save_to_db(ds, now)
            self.assertEqual(rows, [utc(d, 10 - offset, m) for m in (0, 3, 6)], d)

    def test_only_future_rows_published_right_after_change(self):
        # 09:58 local on 25 Oct = 08:58 UTC; 09:57 slot is past, 10:00 is future.
        ds = [pd.Timestamp(datetime(2026, 10, 25, 9, 57)), pd.Timestamp(datetime(2026, 10, 25, 10, 0))]
        rows = self.save_to_db(ds, utc(AUTUMN, 8, 58))
        self.assertEqual(rows, [utc(AUTUMN, 9, 0)])

    def test_shared_payload_and_horizons_on_change_days(self):
        from prediction.forecast_state import build_payload, horizon_value
        for d in (SPRING, AUTUMN):
            ds = [datetime(d.year, d.month, d.day, 9, 0) + timedelta(minutes=3 * i) for i in range(20)]
            rows = [{'ds': t, 'wait_min': float(i)} for i, t in enumerate(ds)]
            as_of = local(d, 9, 0).astimezone(UTC)
            payload = build_payload({1: rows}, [0.0] * 20, 3, STORE_TZ, as_of)
            slots = payload['lanes']['1']['slots']
            self.assertEqual(datetime.fromisoformat(slots[0]['prediction_for']), as_of)
            self.assertEqual(payload['lanes']['1']['wait_15m'], 5.0)  # 15 real minutes = 5th slot
            self.assertEqual(horizon_value(rows, 30, as_of, STORE_TZ, 3), 10.0)

    def test_payload_spanning_the_night_of_the_change(self):
        """Saturday 20:27 -> Sunday 08:30 is 12h03 on the wall clock: 13h03 real time in October, 11h03 in March."""
        from prediction.forecast_state import utc_stamp
        sat = utc_stamp(datetime(2026, 10, 24, 20, 27), STORE_TZ)
        sun = utc_stamp(datetime(2026, 10, 25, 8, 30), STORE_TZ)
        self.assertEqual(sun - sat, timedelta(hours=13, minutes=3))
        sat = utc_stamp(datetime(2026, 3, 28, 20, 27), STORE_TZ)
        sun = utc_stamp(datetime(2026, 3, 29, 8, 30), STORE_TZ)
        self.assertEqual(sun - sat, timedelta(hours=11, minutes=3))

    def test_freshness_age_across_change(self):
        from prediction.forecast_state import freshness
        state = {'forecast_json': {'as_of': utc(AUTUMN, 0, 58).isoformat()}}
        out = freshness(state, max_age_seconds=600, now=utc(AUTUMN, 1, 2))  # 02:58 summer -> 02:02 winter
        self.assertEqual(out['age_seconds'], 240)
        self.assertFalse(out['stale'])


class PublicationReadBack(DatabaseCase):
    def test_wait_forecast_round_trip_over_the_night_of_the_change(self):
        """wait_forecast reads queue_predictions for the next 24 h (mixed offsets on Saturday
        evening), converts to local wall time, then back to UTC for the payload."""
        from prediction.forecast_state import utc_stamp
        for sat, sun in ((date(2026, 10, 24), AUTUMN), (date(2026, 3, 28), SPRING)):
            instants = [local(sat, 20, 0).astimezone(UTC), local(sun, 9, 0).astimezone(UTC)]
            frame = self.read_back(instants, 'ds')
            wall = pd.to_datetime(frame, utc=True).dt.tz_convert(STORE_TZ).dt.tz_localize(None)  # wait_forecast.py
            self.assertEqual([utc_stamp(t.to_pydatetime(), STORE_TZ) for t in wall], instants)


# --------------------------------------------------------------------------- dashboard lane history
@unittest.skipIf(psycopg2 is None, 'psycopg2 not installed')
class LaneHistorySlots(DatabaseCase):
    """[DB] prediction.pipeline.load_lane_history_agg (dashboard lane chart) on a dashboard-style
    connection: no SET timezone, so the server default applies."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dash = cls.connect(set_tz=False)
        cls.dash.cursor().execute('CREATE TEMP TABLE service_events (timestamp TIMESTAMPTZ, camera_id TEXT, '
                                  'lane_id INT, total_dwell_sec FLOAT)')
        # Empty stand-ins so the function's other two queries never read real tables.
        cls.dash.cursor().execute('CREATE TEMP TABLE queue_state_snapshots (timestamp TIMESTAMPTZ)')
        cls.dash.cursor().execute('CREATE TEMP TABLE entrance_events (timestamp TIMESTAMPTZ)')
        c = cls.dash.cursor()
        c.execute('SHOW timezone')
        cls.server_tz = c.fetchone()[0]

    @classmethod
    def tearDownClass(cls):
        cls.dash.close()
        super().tearDownClass()

    def test_slot_labels_use_store_local_time_on_both_sides_of_changes(self):
        from prediction.pipeline import load_lane_history_agg
        c = self.dash.cursor()
        c.execute('TRUNCATE service_events')
        for d in (date(2026, 10, 20), date(2026, 10, 28), date(2026, 3, 26), date(2026, 3, 31)):
            c.execute('INSERT INTO service_events VALUES (%s, %s, 1, 120)', (local(d, 10, 5).astimezone(UTC), 'cam'))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = load_lane_history_agg(self.dash, days=400, shop_open=8, shop_close=21,
                                        checkout_service_min=3, service_min_dwell_sec=30, slot_minutes=15)
        seen = out.loc[out['n_days_observed'].fillna(0) > 0, ['slot_label', 'n_days_observed']]
        self.assertEqual(seen.to_dict('records'), [{'slot_label': '10:00', 'n_days_observed': 4}],
                         f'server time zone {self.server_tz}')


# --------------------------------------------------------------------------- scheduler
class Scheduler(unittest.TestCase):
    def test_interval_timing_uses_monotonic_clock_only(self):
        src = (ROOT / 'run_scheduler.py').read_text(encoding='utf-8')
        self.assertIn('due = time.monotonic()', src)
        self.assertIn('next_deadline(due, args.interval * 60, time.monotonic())', src)
        self.assertNotIn('datetime.now()', src)  # every wall-clock stamp is datetime.now(timezone.utc)

    def test_every_3_minutes_through_the_repeated_hour(self):
        import run_scheduler
        due, runs = 0.0, [0.0]
        for _ in range(40):  # 2 hours of monotonic time
            due = run_scheduler.next_deadline(due, 180, due + 30)
            runs.append(due)
        self.assertEqual({b - a for a, b in zip(runs, runs[1:])}, {180.0})

    def test_heartbeat_age_across_autumn_change(self):
        from prediction import scheduler_control as sc
        heartbeat = utc(AUTUMN, 0, 59).isoformat()          # 02:59 summer time
        proc = mock.Mock(); proc.cmdline.return_value = ['python', 'run_scheduler.py']
        with mock.patch.object(sc, 'read_json', side_effect=[{'pid': 1, 'created_at': 1, 'heartbeat': heartbeat,
                                                               'owner': 'a', 'status': 'waiting'}, {}]), \
             mock.patch.object(sc, 'owned_process', return_value=proc), \
             mock.patch.object(sc, 'datetime', fixed_datetime(utc(AUTUMN, 0, 59) + timedelta(seconds=10))):
            state = sc.scheduler_state()                       # 02:59:10 summer time
        self.assertEqual(state['status'], 'waiting')
        with mock.patch.object(sc, 'read_json', side_effect=[{'pid': 1, 'created_at': 1, 'heartbeat': heartbeat,
                                                               'owner': 'a', 'status': 'waiting'}, {}]), \
             mock.patch.object(sc, 'owned_process', return_value=proc), \
             mock.patch.object(sc, 'datetime', fixed_datetime(utc(AUTUMN, 1, 0) + timedelta(seconds=10))):
            state = sc.scheduler_state()                       # 02:00:10 winter time, 70 s later
        self.assertEqual(state['status'], 'unresponsive')      # 70 s > 15 s: measured in real time

    def test_next_due_is_published_in_utc(self):
        src = (ROOT / 'run_scheduler.py').read_text(encoding='utf-8')
        self.assertIn('state["next_due"] = (datetime.now(timezone.utc)', src)


if __name__ == '__main__':
    unittest.main()
