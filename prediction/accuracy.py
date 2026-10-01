"""Prospective arrival evaluation. No publication, settings changes or model training."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import numpy as np
import pandas as pd
from prediction.arrival_calibration import target_definition, model_weights, factors


def load_arrival_pairs(conn,days=30,lead_minutes=10):
    """One vintage per complete target; covered zero buckets count, outages do not."""
    target=target_definition()
    with conn.cursor() as cur:
        cur.execute("""WITH forecasts AS (
          SELECT r.model_version,r.published_at,r.data_cutoff,p,
            (p->>'prediction_for')::timestamptz AS bucket
          FROM forecast_runs r CROSS JOIN LATERAL jsonb_array_elements(r.predictions) p
          WHERE r.published_at>=NOW()-(%s*INTERVAL '1 day') AND r.source='REAL'
            AND p->'target'=%s::jsonb
        ), chosen AS (
          SELECT DISTINCT ON (bucket) * FROM forecasts
          WHERE published_at<=bucket-(%s*INTERVAL '1 minute')
            AND published_at>bucket-((%s+3)*INTERVAL '1 minute')
            AND data_cutoff<=published_at AND data_cutoff<bucket AND bucket+INTERVAL '3 minutes'<NOW()
            AND bucket=time_bucket('3 minutes',bucket)
          ORDER BY bucket,published_at DESC
        ), actual AS (
          SELECT time_bucket('3 minutes',timestamp) AS bucket,COUNT(*) AS n
          FROM entrance_events WHERE timestamp>=NOW()-(%s*INTERVAL '1 day')
            AND camera_id=%s AND dwell_seconds>=2 GROUP BY 1
        ), samples AS (
          SELECT time_bucket('3 minutes',timestamp) AS bucket,timestamp,
            EXTRACT(EPOCH FROM timestamp-LAG(timestamp) OVER (
              PARTITION BY time_bucket('3 minutes',timestamp) ORDER BY timestamp)) AS gap
          FROM queue_state_snapshots WHERE timestamp>=NOW()-(%s*INTERVAL '1 day') AND camera_id=%s
        ), coverage AS (
          SELECT bucket,EXTRACT(EPOCH FROM MAX(timestamp)-MIN(timestamp)) AS span,
            COUNT(*) AS n,MAX(gap) AS max_gap FROM samples GROUP BY bucket
        ) SELECT c.bucket,c.published_at,c.model_version,c.p,
            COALESCE(a.n,0) AS actual
          FROM chosen c LEFT JOIN actual a USING(bucket) JOIN coverage v USING(bucket)
          WHERE v.span>=150 AND v.n>=6 AND v.max_gap<=30 ORDER BY c.bucket""",
          (days,json.dumps(target),lead_minutes,lead_minutes,days,target['camera_id'],days,target['camera_id']))
        rows=cur.fetchall()
    records=[]
    for stamp,published,version,p,actual in rows:
        if any(p.get(n) is None or not math.isfinite(float(p[n])) for n in ('prophet','lstm','xgboost')):continue
        records.append(dict(ds=stamp,published_at=published,model_version=version,actual=float(actual),
                            **{n:float(p[n]) for n in ('prophet','lstm','xgboost')}))
    return pd.DataFrame(records)


def metrics(actual,predicted):
    error=np.asarray(predicted)-np.asarray(actual)
    return dict(n=len(error),mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),bias=float(error.mean()))


def evaluate(frame,weights,model_version,lead_minutes=10):
    report=dict(status='insufficient_data',model_version=model_version,lead_minutes=lead_minutes,
                target=target_definition(),weights=weights,calibration_published=False,
                coverage_rule='Entrance-camera snapshots: span >=150s, >=6 samples, gaps <=30s per complete 3-minute bucket',
                wait_accuracy='Unavailable without independently measured waiting times; service duration is not waiting time.')
    if frame.empty:return report,None
    frame=frame[frame.model_version==model_version].copy()
    if frame.empty:return report,None
    frame['raw']=sum(frame[n]*w for n,w in weights.items())
    frame['ds']=pd.to_datetime(frame.ds,utc=True)
    frame['date']=frame.ds.dt.date
    report.update(slots=len(frame),days=int(frame.date.nunique()),raw=metrics(frame.actual,frame.raw),
                  zero_buckets=int((frame.actual==0).sum()),start=frame.ds.min().isoformat(),end=frame.ds.max().isoformat())
    dates=sorted(frame.date.unique())
    if len(dates)<14:return report,None
    split=dates[-7];fit=frame[frame.date<split].copy();holdout=frame[frame.date>=split].copy()
    if len(fit)<100 or len(holdout)<100 or fit.raw.sum()<=0:return report,None
    k=float(fit.actual.sum()/fit.raw.sum());hourly={}
    fit['hour']=fit.ds.dt.hour
    for hour,group in fit.groupby('hour'):
        if len(group)>=30 and group.date.nunique()>=7 and group.raw.sum()>0:
            hourly[str(int(hour))]=float(group.actual.sum()/group.raw.sum())
    now=datetime.now(timezone.utc)
    artifact=dict(schema=2,target=target_definition(),model_version=model_version,weights=weights,timezone='UTC',
                  k_global=k,k_by_hour=hourly,trained_at_utc=now.isoformat(),expires_at=(now+timedelta(days=7)).isoformat(),
                  fit_start=fit.ds.min().isoformat(),fit_end=fit.ds.max().isoformat(),
                  holdout_start=holdout.ds.min().isoformat(),holdout_end=holdout.ds.max().isoformat(),lead_minutes=lead_minutes)
    raw=metrics(holdout.actual,holdout.raw)
    corrected=metrics(holdout.actual,holdout.raw*np.array(factors(holdout.ds,artifact)))
    bounded=all(math.isfinite(v) and .25<=v<=4 for v in [k,*hourly.values()])
    passed=bounded and corrected['mae']<raw['mae'] and abs(corrected['bias'])<=abs(raw['bias'])
    artifact['validation']=dict(passed=bool(passed),fit_days=int(fit.date.nunique()),holdout_days=int(holdout.date.nunique()),
        fit_slots=len(fit),holdout_slots=len(holdout),raw_mae=raw['mae'],calibrated_mae=corrected['mae'])
    artifact['artifact_id']=hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest()[:24]
    report.update(status='holdout_passed' if passed else 'holdout_failed',holdout_raw=raw,holdout_calibrated=corrected,
                  validation=artifact['validation'])
    return report,artifact if passed else None


def evaluate_wait_observations(conn,observations,lead_minutes=10):
    """Score selected-lane waits against independent queue-wait measurements.

    CSV: observed_at (aware), lanes, wait_minutes, measurement_source.
    Measured waits exclude service duration.
    """
    errors=[]
    for row in observations:
        stamp=datetime.fromisoformat(row['observed_at'])
        actual=float(row['wait_minutes']);lanes=int(row['lanes'])
        if stamp.tzinfo is None or not math.isfinite(actual) or actual<0 or lanes<1 or not row.get('measurement_source'):
            raise ValueError('Wait observations require aware timestamps, nonnegative waits, lanes and provenance')
        with conn.cursor() as cur:
            cur.execute("""SELECT s->>'wait_min' FROM wait_forecast_runs r
                CROSS JOIN LATERAL jsonb_array_elements(r.payload->'lanes'->(%s::text)->'slots') s
                WHERE r.selected_lanes=%s AND r.published_at >= (r.payload->>'as_of')::timestamptz
                  AND r.published_at<=%s-(%s*INTERVAL '1 minute')
                  AND r.published_at>%s-((%s+3)*INTERVAL '1 minute')
                  AND ABS(EXTRACT(EPOCH FROM (s->>'prediction_for')::timestamptz-%s))<=90
                ORDER BY r.published_at DESC,ABS(EXTRACT(EPOCH FROM (s->>'prediction_for')::timestamptz-%s)) LIMIT 1""",
                (str(lanes),lanes,stamp,lead_minutes,stamp,lead_minutes,stamp,stamp))
            found=cur.fetchone()
        if found and found[0] is not None:errors.append((actual,float(found[0])))
    return dict(status='scored' if errors else 'no_matching_forecasts',matched=len(errors),
                supplied=len(observations),lead_minutes=lead_minutes,
                metrics=metrics([a for a,p in errors],[p for a,p in errors]) if errors else None)
