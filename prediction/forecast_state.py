"""Shared, serialized dashboard simulations for the mobile forecast endpoints."""
from datetime import datetime, timezone, timedelta
import json
import math
import os
from zoneinfo import ZoneInfo

# One wait-status definition for every screen, chart, recap and alert.
WAIT_BUSY_MIN = float(os.getenv('WAIT_BUSY_MIN', 2.0))
WAIT_ALERT_MIN = float(os.getenv('WAIT_ALERT_MIN', 5.0))


def status_for(wait_min):
    """OK / BUSY / ALERT for a predicted wait in minutes (None when unknown)."""
    wait = finite(wait_min)
    if wait is None:
        return None
    return 'ALERT' if wait >= WAIT_ALERT_MIN else 'BUSY' if wait >= WAIT_BUSY_MIN else 'OK'


def finite(value):
    try:
        number = float(value)
        return round(number, 2) if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def waiting_backlog(total, lanes):
    return max(0.0, float(total) - int(lanes))


def utc_stamp(value, store_tz="Europe/Paris"):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(store_tz))
    return value.astimezone(timezone.utc)


def horizon_row(slots, minutes, reference, bucket_minutes=3):
    if not slots:
        return None
    target = utc_stamp(reference) + timedelta(minutes=minutes)
    dates = [utc_stamp(row["prediction_for"]) for row in slots]
    if target > dates[-1] or target < dates[0] - timedelta(minutes=bucket_minutes):
        return None
    return slots[min(range(len(slots)), key=lambda i: abs((dates[i]-target).total_seconds()))]


def horizon_value(rows, minutes, reference, store_tz="Europe/Paris", bucket_minutes=3):
    slots = [dict(prediction_for=utc_stamp(row["ds"], store_tz).isoformat(), wait_min=finite(row["wait_min"])) for row in rows]
    row = horizon_row(slots, minutes, reference, bucket_minutes)
    return row["wait_min"] if row else None


def build_payload(lane_rows, arrivals, bucket_minutes, store_tz, as_of=None):
    """Keep the actual nonlinear simulation for each lane count and horizon."""
    as_of = as_of or datetime.now(timezone.utc)
    scenarios = {}
    for lane, rows in lane_rows.items():
        slots = []
        for i, row in enumerate(rows):
            when = row['ds']
            if isinstance(when, str):
                when = datetime.fromisoformat(when)
            if when.tzinfo is None:
                when = when.replace(tzinfo=ZoneInfo(store_tz))
            slots.append(dict(prediction_for=when.astimezone(timezone.utc).isoformat(),
                              wait_min=finite(row['wait_min']),
                              arrivals=finite(arrivals[i]) if i < len(arrivals) else None))
        scenario = {'slots': slots}
        for minutes in (0, 5, 10, 15):
            row = horizon_row(slots, minutes, as_of, bucket_minutes)
            scenario[f'wait_{minutes}m'] = row['wait_min'] if row else None
        scenarios[str(lane)] = scenario
    return dict(schema=1, as_of=as_of.isoformat(),
                bucket_minutes=bucket_minutes, lanes=scenarios)


def payload_from(state):
    payload = (state or {}).get('forecast_json') or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return {}
    return payload if isinstance(payload, dict) else {}


def scenario_for(state, lanes=None):
    count = lanes if lanes is not None else (state or {}).get('open_lanes')
    return payload_from(state).get('lanes', {}).get(str(count), {})


def freshness(state, max_age_seconds=600, now=None):
    payload = payload_from(state)
    stamp = payload.get('as_of') or (state or {}).get('updated_at')
    now = now or datetime.now(timezone.utc)
    def age(value):
        try:
            seconds = (now - utc_stamp(value)).total_seconds()
            return max(0, round(seconds)), seconds > max_age_seconds or seconds < -30
        except (TypeError, ValueError, AttributeError):
            return None, True
    elapsed, stale = age(stamp)
    sources = {}
    for key in ('source_published_at', 'snapshot_at', 'service_cutoff', 'data_cutoff'):
        if key in payload:
            seconds, outdated = age(payload[key])
            sources[key] = seconds
            # No recent completed transaction/entrance can just mean a quiet shop.
            # Publication and camera heartbeat, however, must stay recent.
            if key in ('source_published_at', 'snapshot_at'):
                stale = stale or outdated
    revision=payload.get('settings_revision')
    current=(state or {}).get('current_settings_revision')
    pending=current is not None and revision!=current
    return dict(stale=stale, age_seconds=elapsed, pending_settings=pending,
                settings_revision=revision,current_settings_revision=current,
                calibration=payload.get('calibration'),
                updated_at=utc_stamp(stamp).isoformat() if elapsed is not None else None,
                source_age_seconds=sources, publisher=payload.get('publisher'),
                publication_id=payload.get('publication_id'))


STATUS_COLORS = {None: 'gray', 'OK': 'green', 'BUSY': 'orange', 'ALERT': 'red'}


def lane_scenarios(state):
    current = int((state or {}).get('open_lanes') or 1)
    result = []
    for lane, scenario in sorted(payload_from(state).get('lanes', {}).items(), key=lambda item:int(item[0])):
        wait = finite(scenario.get('wait_10m'))
        color = STATUS_COLORS[status_for(wait)]
        result.append(dict(lanes=int(lane), est_wait_min=round(wait, 1) if wait is not None else None,
                           color=color, is_current=int(lane)==current))
    return result


def wait_at_horizon(state, minutes):
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError('minutes must be finite and nonnegative')
    payload = payload_from(state)
    slots = scenario_for(state).get('slots', [])
    bucket = payload.get('bucket_minutes', 3)
    reference = payload.get('as_of')
    row = horizon_row(slots, minutes, reference, bucket) if reference else None
    max_horizon = max(0, (utc_stamp(slots[-1]['prediction_for'])-utc_stamp(reference)).total_seconds()/60) if slots and reference else None
    return dict(wait_min=round(row['wait_min'],1) if row and row.get('wait_min') is not None else None,
                horizon_min=minutes, matched_for=row['prediction_for'] if row else None,
                max_horizon_min=round(max_horizon, 1) if max_horizon is not None else None,
                reference_at=reference,
                current_lanes=(state or {}).get('open_lanes'),
                message=None if row else 'Requested horizon is outside the saved dashboard forecast.',
                **freshness(state))


_HISTORY_SQL = '''
    SELECT DISTINCT ON (prediction_for) prediction_for, arrivals, wait_min
    FROM (
        SELECT (s->>'prediction_for')::timestamptz AS prediction_for,
               (s->>'arrivals')::float AS arrivals, (s->>'wait_min')::float AS wait_min, r.published_at
        FROM wait_forecast_runs r
        CROSS JOIN LATERAL jsonb_array_elements(r.payload->'lanes'->(r.selected_lanes::text)->'slots') s
        WHERE r.published_at >= %(start)s - INTERVAL '1 day' AND r.published_at < %(end)s
    ) published
    WHERE prediction_for >= %(start)s AND prediction_for < %(end)s AND prediction_for < %(cut)s
    ORDER BY prediction_for, published_at DESC'''


def published_series(conn, start, end, store_tz="Europe/Paris"):
    """The one forecast series shown everywhere, one row per slot in [start, end).

    Slots covered by the current publication come from dashboard_state with the
    lanes selected right now (exactly what the Live tab shows); earlier slots come
    from wait_forecast_runs: the last publication made before each slot, using the
    lanes selected at that time. Nothing exists before the history began (27 Sep).
    """
    start_utc, end_utc = utc_stamp(start, store_tz), utc_stamp(end, store_tz)
    with conn.cursor() as cur:
        cur.execute('SELECT forecast_json, open_lanes FROM dashboard_state WHERE id=1')
        row = cur.fetchone()
        state = dict(forecast_json=row[0], open_lanes=row[1]) if row else {}
        current = [s for s in scenario_for(state).get('slots', []) if s.get('prediction_for')]
        cut = utc_stamp(current[0]['prediction_for']) if current else datetime.now(timezone.utc)
        cur.execute(_HISTORY_SQL, dict(start=start_utc, end=end_utc, cut=cut))
        rows = [dict(prediction_for=r[0].astimezone(timezone.utc), arrivals=finite(r[1]), wait_min=finite(r[2]))
                for r in cur.fetchall()]
    for slot in current:
        when = utc_stamp(slot['prediction_for'])
        if start_utc <= when < end_utc:
            rows.append(dict(prediction_for=when, arrivals=finite(slot.get('arrivals')), wait_min=finite(slot.get('wait_min'))))
    for row in rows:
        row['status'] = status_for(row['wait_min'])
    return rows
