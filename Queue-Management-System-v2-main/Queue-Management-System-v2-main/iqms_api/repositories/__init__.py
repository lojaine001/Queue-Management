"""Repositories: the SQL. One module per table group; each function takes an open cursor
and runs one query, so the service that calls it decides the transaction.

    dashboard.py     dashboard_state (published forecast, lanes), dashboard_statistics
    snapshots.py     queue_state_snapshots
    entrance.py      entrance_events
    service.py       service_events
    predictions.py   queue_predictions
    day_filter.py    the "rows of one day" condition (day passed as a parameter)
    alerts.py        alert_responses
    push.py          push_subscriptions
    kpis.py          KPI reads (entrance, checkout wait and queue, lanes, alerts)
"""
