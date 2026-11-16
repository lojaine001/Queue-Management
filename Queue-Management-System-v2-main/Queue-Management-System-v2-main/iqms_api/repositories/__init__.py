"""Repositories: the SQL. One module per table group; each function takes an open cursor
and runs one query, so the service that calls it decides the transaction.

    dashboard.py     dashboard_state (published forecast, lanes)
    snapshots.py     queue_state_snapshots
    entrance.py      entrance_events
    predictions.py   queue_predictions
    alerts.py        alert_responses
    push.py          push_subscriptions
    kpis.py          KPI reads (entrance, checkout wait and queue, lanes, alerts)
"""
