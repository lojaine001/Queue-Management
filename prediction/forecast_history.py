"""Preserve each forecast vintage in the same transaction as live publication."""
from psycopg2.extras import Json


def save_history(cursor, result, rows, published_at):
    cursor.execute('''
        INSERT INTO forecast_runs
            (run_id, run_started_at, data_cutoff, forecast_origin, published_at, model_version, source, predictions)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ''', (result['run_id'], result['run_started_at'], result['data_cutoff'], result['forecast_origin'],
          published_at, result['model_version'], result['source'], Json(rows)))
