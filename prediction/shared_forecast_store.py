"""Canonical settings and transactional storage for background wait forecasts."""
import json
import uuid
import os
from contextlib import closing
import psycopg2
import psycopg2.extras
from prediction.pipeline import DB_CONFIG
from prediction.forecast_state import finite, freshness, scenario_for

DEFAULT_SETTINGS = dict(arrival_models=['prophet','lstm','xgboost'], pred_smooth_min=5,
                        dwell_model_mode='safe', dwell_forecast_models=['xgboost','lstm'], calibration_enabled=False)


def connect():
    conn=psycopg2.connect(**{**DB_CONFIG,'connect_timeout':10,'options':'-c statement_timeout=30000'})
    with conn.cursor() as cur:cur.execute('SET timezone = %s',(os.getenv('STORE_TZ','Europe/Paris'),))
    conn.commit()
    return conn


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS forecast_settings (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK(id=1), revision BIGINT NOT NULL DEFAULT 1,
            config JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
        cur.execute('INSERT INTO forecast_settings (id,config) VALUES (1,%s) ON CONFLICT(id) DO NOTHING',
                    (psycopg2.extras.Json(DEFAULT_SETTINGS),))
        cur.execute('''CREATE TABLE IF NOT EXISTS dashboard_state (
            id INTEGER PRIMARY KEY DEFAULT 1, updated_at TIMESTAMPTZ,
            queue_now INTEGER, service_min FLOAT, wait_0m FLOAT, wait_5m FLOAT, wait_10m FLOAT, wait_15m FLOAT,
            lane1_wait_15m FLOAT,lane2_wait_15m FLOAT,lane3_wait_15m FLOAT,lane4_wait_15m FLOAT,
            open_lanes INTEGER,demographics_json TEXT,entries_hour_json TEXT,forecast_json JSONB)''')
        cur.execute('ALTER TABLE dashboard_state ADD COLUMN IF NOT EXISTS forecast_json JSONB')
        cur.execute('''CREATE TABLE IF NOT EXISTS wait_forecast_runs (
            publication_id TEXT PRIMARY KEY, published_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            settings_revision BIGINT NOT NULL, selected_lanes INTEGER NOT NULL, payload JSONB NOT NULL)''')
        cur.execute('CREATE INDEX IF NOT EXISTS wait_forecast_runs_published_idx ON wait_forecast_runs(published_at DESC)')
    conn.commit()


def validate_settings(config):
    if set(config)!=set(DEFAULT_SETTINGS):raise ValueError('Unknown or missing forecast settings')
    for field,allowed in [('arrival_models',{'prophet','lstm','xgboost'}),('dwell_forecast_models',{'lstm','xgboost'})]:
        values=config[field]
        if not isinstance(values,list) or not values or not set(values)<=allowed:raise ValueError(f'Invalid {field}')
    if config['pred_smooth_min'] not in (3,5,15,30):raise ValueError('Invalid smoothing interval')
    if config['dwell_model_mode'] not in ('safe','legacy'):raise ValueError('Invalid dwell mode')
    if not isinstance(config['calibration_enabled'],bool):raise ValueError('Invalid calibration setting')
    return config


def read_settings():
    with closing(connect()) as conn:
        ensure_schema(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT revision,config FROM forecast_settings WHERE id=1')
            row=dict(cur.fetchone())
            validate_settings(row['config'])
            return row


def update_settings(patch):
    with closing(connect()) as conn, conn:
        with conn.cursor() as cur:
            cur.execute('SELECT config FROM forecast_settings WHERE id=1 FOR UPDATE')
            config=cur.fetchone()[0]
            config.update(patch);validate_settings(config)
            cur.execute('UPDATE forecast_settings SET config=%s,revision=revision+1,updated_at=NOW() WHERE id=1',
                        (psycopg2.extras.Json(config),))


def read_state():
    with closing(connect()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM dashboard_state WHERE id=1')
            return dict(cur.fetchone() or {})


def select_lanes(lanes):
    if not 1<=int(lanes)<=int(os.getenv('MAX_LANES',4)):raise ValueError('Invalid lane count')
    with closing(connect()) as conn,conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM dashboard_state WHERE id=1 FOR UPDATE')
            state=cur.fetchone() or {};scenario=scenario_for(state,int(lanes))
            if scenario:
                cur.execute('''UPDATE dashboard_state SET open_lanes=%s,wait_0m=%s,wait_5m=%s,wait_10m=%s,wait_15m=%s WHERE id=1''',
                            (int(lanes),*(finite(scenario.get(f'wait_{m}m')) for m in (0,5,10,15))))
            else:
                cur.execute('''UPDATE dashboard_state SET open_lanes=%s WHERE id=1''', (int(lanes),))


def publish(conn,payload):
    """Settings revision is guarded; current staffing is chosen inside the transaction."""
    with conn:
        with conn.cursor() as cur:
            cur.execute('SELECT revision FROM forecast_settings WHERE id=1 FOR SHARE')
            if cur.fetchone()[0]!=payload['settings_revision']:
                raise RuntimeError('Forecast settings changed during calculation; discarded obsolete result')
            cur.execute('SELECT open_lanes FROM dashboard_state WHERE id=1 FOR UPDATE')
            row=cur.fetchone();lanes=int(row[0] or os.getenv('DEFAULT_LANES',2)) if row else int(os.getenv('DEFAULT_LANES',2))
            payload={**payload,'publication_id':payload.get('publication_id') or uuid.uuid4().hex,'selected_lanes':lanes}
            scenario=payload['lanes'].get(str(lanes),{})
            waits=[finite(scenario.get(f'wait_{m}m')) for m in (0,5,10,15)]
            alt=[finite(payload['lanes'].get(str(n),{}).get('wait_15m')) for n in range(1,5)]
            cur.execute('''INSERT INTO dashboard_state
                (id,updated_at,queue_now,service_min,open_lanes,wait_0m,wait_5m,wait_10m,wait_15m,
                 lane1_wait_15m,lane2_wait_15m,lane3_wait_15m,lane4_wait_15m,forecast_json)
                VALUES (1,clock_timestamp(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET updated_at=EXCLUDED.updated_at,queue_now=EXCLUDED.queue_now,
                service_min=EXCLUDED.service_min,open_lanes=EXCLUDED.open_lanes,
                wait_0m=EXCLUDED.wait_0m,wait_5m=EXCLUDED.wait_5m,wait_10m=EXCLUDED.wait_10m,wait_15m=EXCLUDED.wait_15m,
                lane1_wait_15m=EXCLUDED.lane1_wait_15m,lane2_wait_15m=EXCLUDED.lane2_wait_15m,
                lane3_wait_15m=EXCLUDED.lane3_wait_15m,lane4_wait_15m=EXCLUDED.lane4_wait_15m,forecast_json=EXCLUDED.forecast_json''',
                (payload['queue_now'],payload['service_min'],lanes,*waits,*alt,psycopg2.extras.Json(payload)))
            cur.execute('''INSERT INTO wait_forecast_runs(publication_id,published_at,settings_revision,selected_lanes,payload)
                VALUES (%s,clock_timestamp(),%s,%s,%s)''',
                (payload['publication_id'],payload['settings_revision'],lanes,psycopg2.extras.Json(payload)))
