"""Preserve each forecast vintage in the same transaction as live publication."""
from psycopg2.extras import Json


def save_history(cursor, result, rows, published_at):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            run_started_at TIMESTAMPTZ NOT NULL,
            data_cutoff TIMESTAMPTZ,
            forecast_origin TIMESTAMPTZ NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            model_version TEXT NOT NULL,
            source TEXT NOT NULL,
            predictions JSONB NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS forecast_runs_published_idx ON forecast_runs (published_at DESC)')
    cursor.execute('''
        INSERT INTO forecast_runs
            (run_id, run_started_at, data_cutoff, forecast_origin, published_at, model_version, source, predictions)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ''', (result['run_id'], result['run_started_at'], result['data_cutoff'], result['forecast_origin'],
          published_at, result['model_version'], result['source'], Json(rows)))
