"""Services: what each endpoint does, without HTTP. A service opens the connection,
calls the repositories inside one transaction, applies the rules, and returns the
answer's data. When it refuses a request it raises errors.BadRequest / Conflict.

    administration.py   lane changes, answers to alerts
    notifications.py    web push subscriptions
    kpis.py             checkout wait and queue KPIs, shared by the export and the recap
    exports.py          KPI export (preview and CSV)
    live.py             live lanes and alert level
    forecasts.py        published forecast: scenarios, horizons, charts
    reports.py          day recap and the background statistics
"""
