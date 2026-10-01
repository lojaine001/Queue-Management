"""Browser-independent wait simulation, called after each arrival publication."""
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from prediction.arrival_calibration import calibration_status, factors, model_weights, target_definition
from prediction.core import (BUCKET_MINUTES, DEFAULT_DWELL_MIN, DWELL_MIN_FLOOR, DWELL_MAX_CAP,
                             MAX_QUEUE_PER_LANE, MAX_WAIT_MIN, compute_wait_estimates, is_open)
from prediction.dashboard_lstm import forecast_dwell
from prediction.dwell_models import _build_dashboard_dwell_model, _build_dashboard_dwell_lstm_model, _DWELL_FEAT
from prediction.dwell_modifier import compute_dwell_modifier_from_recent
from prediction.forecast_state import build_payload, waiting_backlog, finite, utc_stamp
from prediction.runtime import ROOT, RUNTIME, RunLock, read_json, write_json
from prediction.shared_forecast_store import connect, ensure_schema, read_settings, publish

STORE_TZ=os.getenv('STORE_TZ','Europe/Paris')
APP=ROOT/'Queue-Management-System-v2-main'/'Queue-Management-System-v2-main'


def prepare_arrivals(frame,config,calibration,entries_hour,model_version=None):
    frame=frame.copy()
    names={'prophet':'prophet_yhat','lstm':'lstm_yhat','xgboost':'xgb_yhat'}
    weights=model_weights(config['arrival_models'])
    selected=config['arrival_models'];cols=[names[n] for n in selected]
    values=frame[cols].apply(pd.to_numeric,errors='coerce')
    w=np.array([weights[n] for n in selected]);w=w if w.sum()>0 else np.ones(len(w))
    denom=values.notna().mul(w,axis=1).sum(axis=1)
    frame['arrivals']=values.mul(w,axis=1).sum(axis=1).div(denom.where(denom>0)).fillna(0).clip(lower=0)
    status=calibration_status(calibration,config['calibration_enabled'],model_version,weights)
    if status['applied'] and values.isna().any().any():
        status.update(applied=False,artifact_id=None,reason='Missing selected model output; raw fallback')
    calibrated=status['applied']
    if calibrated:frame['arrivals']*=factors(frame['ds'],calibration)
    frame.attrs['calibration']=status
    rate=entries_hour/(60/BUCKET_MINUTES)
    capped=rate>0 and frame['arrivals'].mean()>rate*5
    if capped:frame['arrivals']=frame['arrivals'].clip(upper=rate*5)
    window=max(1,round(config['pred_smooth_min']/BUCKET_MINUTES))
    frame['arrivals']=frame['arrivals'].rolling(window,center=True,min_periods=1).mean()
    return frame,calibrated,bool(capped)


def simulate_lanes(frame,queue,service_min,dwell_values,as_of,max_lanes=4):
    scenarios={};base=dwell_values[0] if dwell_values else service_min
    for lanes in range(1,max_lanes+1):
        dwell=list(dwell_values) if dwell_values else [service_min]*len(frame)
        if min(queue,lanes)==lanes and BUCKET_MINUTES>base/2 and dwell:
            dwell[0]=BUCKET_MINUTES*base/(BUCKET_MINUTES-base/2)
        rows,*_=compute_wait_estimates(frame.rename(columns={'arrivals':'yhat'}),
            current_queue=waiting_backlog(queue,lanes),avg_dwell_min=dwell or service_min,
            active_lanes=lanes,max_queue_per_lane=MAX_QUEUE_PER_LANE,max_wait_min=MAX_WAIT_MIN)
        scenarios[lanes]=rows
    return build_payload(scenarios,frame['arrivals'].tolist(),BUCKET_MINUTES,STORE_TZ,as_of)


def _service_median(conn):
    queries=[
        'SELECT total_dwell_sec/60.0 AS service_min FROM service_events WHERE total_dwell_sec>0 AND active_head_tracks_in_lane=0 AND timestamp>=NOW()-INTERVAL \'7 days\'',
        'SELECT se.total_dwell_sec/60.0 AS service_min FROM service_events se JOIN entrance_events ee ON ee.track_id=se.track_id WHERE se.total_dwell_sec>0 AND ee.active_head_tracks_in_lane=0 AND se.timestamp>=NOW()-INTERVAL \'7 days\'',
        'SELECT total_dwell_sec/60.0 AS service_min FROM service_events WHERE total_dwell_sec>0 AND timestamp>=NOW()-INTERVAL \'7 days\'']
    for index,query in enumerate(queries):
        frame=pd.read_sql(query,conn)
        if len(frame)<10 and index<2:continue
        values=pd.to_numeric(frame['service_min'],errors='coerce').dropna()
        values=values[values>=DWELL_MIN_FLOOR]
        if not values.empty:return float(np.clip(values.median(),DWELL_MIN_FLOOR,10))
    return float(os.getenv('CHECKOUT_SERVICE_MIN',DEFAULT_DWELL_MIN))


def refresh_wait_forecast(days=30):
    if days<=0:raise ValueError('days must be positive')
    with RunLock(RUNTIME/'wait_forecast.lock'):
        settings=read_settings();config=settings['config'];started=datetime.now(timezone.utc)
        with closing(connect()) as conn:
            ensure_schema(conn)
            frame=pd.read_sql('''SELECT prediction_for AS ds,predicted_at,prophet_yhat,lstm_yhat,xgb_yhat
                FROM queue_predictions WHERE source='ensemble' AND predicted_at=(SELECT MAX(predicted_at) FROM queue_predictions WHERE source='ensemble')
                AND prediction_for>=NOW() AND prediction_for<=NOW()+INTERVAL '24 hours'
                ORDER BY prediction_for LIMIT 60''',conn)
            if frame.empty:raise RuntimeError('No future arrival forecast available')
            source_published=frame['predicted_at'].min().isoformat()
            frame['ds']=pd.to_datetime(frame['ds']).dt.tz_convert(STORE_TZ).dt.tz_localize(None)
            frame=frame[frame['ds'].map(is_open)].reset_index(drop=True)
            history=pd.read_sql('''SELECT time_bucket('15 minutes',timestamp) AS ds,
                PERCENTILE_CONT(.5) WITHIN GROUP(ORDER BY total_dwell_sec/60.0) AS dwell_min,COUNT(*) AS n_events
                FROM service_events WHERE total_dwell_sec >= %s AND total_dwell_sec<=1800
                AND timestamp>=NOW()-(%s*INTERVAL '1 day') GROUP BY ds ORDER BY ds''',conn,
                params=(int(DWELL_MIN_FLOOR*60),days))
            service_min=_service_median(conn)
            meta={};predictions={}
            for name in config['dwell_forecast_models']:
                if name=='xgboost':
                    model,meta[name]=_build_dashboard_dwell_model(history,config['dwell_model_mode'])
                    if model is not None and not frame.empty:
                        stamps=frame['ds'];features=pd.DataFrame(dict(hour=stamps.dt.hour,minute_of_hour=stamps.dt.minute,day_of_week=stamps.dt.dayofweek,is_weekend=(stamps.dt.dayofweek>=5).astype(int)))
                        predictions[name]=model.predict(features[_DWELL_FEAT]).clip(DWELL_MIN_FLOOR,DWELL_MAX_CAP).tolist()
                else:
                    model,meta[name]=_build_dashboard_dwell_lstm_model(history,config['dwell_model_mode'])
                    if model is not None and not frame.empty:
                        try:
                            predictions[name]=forecast_dwell(model['history_values'],model['seq_len'],len(frame),int(os.getenv('DWELL_LSTM_EPOCHS',12)),DWELL_MIN_FLOOR,DWELL_MAX_CAP)
                            meta[name]['status']='trained'
                        except (RuntimeError,TimeoutError) as exc:
                            meta[name].update(status='fallback',reason=str(exc))
                            print(f'[Shared wait] LSTM fallback: {exc}',flush=True)
            # Slow model preparation must not age the live queue or time origin.
            as_of=datetime.now(timezone.utc)
            keep=[i for i,t in enumerate(frame['ds']) if utc_stamp(t,STORE_TZ)>=as_of]
            frame=frame.iloc[keep].reset_index(drop=True)
            predictions={name:[values[i] for i in keep] for name,values in predictions.items()}
            with conn.cursor() as cur:
                cur.execute('SELECT timestamp,queue_count FROM queue_state_snapshots WHERE camera_id=%s ORDER BY timestamp DESC LIMIT 1',
                            (os.getenv('CHECKOUT_CAM_ID','Bosch_Camera_exit'),))
                snap=cur.fetchone()
                if not snap:raise RuntimeError('No checkout snapshot available')
                cur.execute('SELECT COUNT(*) FROM entrance_events WHERE camera_id=%s AND dwell_seconds>=2 AND timestamp>=NOW()-INTERVAL \'1 hour\'',
                            (os.getenv('CAM_ID','Bosch_Camera_Entrance'),))
                entries=int(cur.fetchone()[0])
                cur.execute('SELECT run_id,model_version,data_cutoff FROM forecast_runs WHERE published_at=%s ORDER BY id DESC LIMIT 1',(source_published,))
                lineage=cur.fetchone()
                cur.execute('SELECT MAX(timestamp) FROM service_events')
                service_cutoff=cur.fetchone()[0]
            modifier=compute_dwell_modifier_from_recent(conn)
            service_min*=modifier
            frame,calibrated,capped=prepare_arrivals(frame,config,read_json(APP/'calibration.json'),entries,lineage[1] if lineage else None)
            dwell=np.mean(list(predictions.values()),axis=0).tolist() if predictions and not frame.empty else []
            payload=simulate_lanes(frame,int(snap[1] or 0),service_min,dwell,as_of,int(os.getenv('MAX_LANES',4)))
            payload.update(publisher='background',publication_id=uuid.uuid4().hex,settings_revision=settings['revision'],config=config,
                training_days=days,queue_now=int(snap[1] or 0),service_min=service_min,
                source_published_at=source_published,snapshot_at=snap[0].isoformat(),
                service_cutoff=service_cutoff.isoformat() if service_cutoff else None,
                arrival_run_id=lineage[0] if lineage else None,model_version=lineage[1] if lineage else None,
                data_cutoff=lineage[2].isoformat() if lineage and lineage[2] else None,
                dwell_by_model=predictions,dwell_values=dwell,dwell_meta=meta,
                calibration_applied=calibrated,calibration=frame.attrs.get('calibration'),target=target_definition(),arrivals_capped=capped,
                duration_seconds=round((datetime.now(timezone.utc)-started).total_seconds(),2))
            publish(conn,payload)
            write_json(RUNTIME/'wait_publication.json',{k:v for k,v in payload.items() if k not in ('lanes','dwell_by_model','dwell_values')})
            print(f"[Shared wait] Published {len(frame)} slots for {len(payload['lanes'])} lane scenarios in {payload['duration_seconds']}s",flush=True)
            return payload
