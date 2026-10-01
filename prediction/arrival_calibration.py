"""One guarded correction for shared arrivals; stored model forecasts stay raw."""
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def target_definition():
    return dict(camera_id=os.getenv('CAM_ID','Bosch_Camera_Entrance'),
                dwell_min_seconds=10,source='REAL',bucket_minutes=3)


def model_weights(selected):
    names={'prophet':'W_PROPHET','lstm':'W_LSTM','xgboost':'W_XGB'}
    defaults={'prophet':.4,'lstm':.3,'xgboost':.3}
    weights={n:float(os.getenv(names[n],defaults[n])) for n in selected}
    if any(not math.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError('Arrival weights must be finite, nonnegative and have positive total')
    total=sum(weights.values())
    return {n:round(v/total,12) for n,v in weights.items()}


def calibration_status(artifact,enabled,model_version,weights,now=None):
    result=dict(requested=bool(enabled),applied=False,artifact_id=None,reason='Disabled; raw forecast baseline')
    if not enabled:return result
    try:
        if not isinstance(artifact,dict):raise ValueError('Calibration must be an object')
        if artifact.get('schema')!=2:raise ValueError('Legacy or missing calibration; prospective validation required')
        if artifact['target']!=target_definition():raise ValueError('Calibration target does not match')
        if artifact['model_version']!=model_version or artifact['weights']!=weights:
            raise ValueError('Calibration model or weights do not match')
        ZoneInfo(artifact['timezone'])
        now=now or datetime.now(timezone.utc)
        cutoff=datetime.fromisoformat(artifact['expires_at'])
        if cutoff.tzinfo is None or cutoff<=now:raise ValueError('Calibration expired')
        evidence=artifact['validation']
        if not evidence['passed'] or evidence['holdout_days']<7 or evidence['fit_days']<7:
            raise ValueError('Insufficient independent validation')
        if not all(math.isfinite(float(evidence[k])) and float(evidence[k])>=0 for k in ('raw_mae','calibrated_mae')):
            raise ValueError('Invalid validation metrics')
        if evidence['calibrated_mae']>=evidence['raw_mae']:
            raise ValueError('Calibration did not improve holdout MAE')
        if not artifact.get('artifact_id'):raise ValueError('Missing artifact identity')
        for key,value in artifact['k_by_hour'].items():
            if not str(key).isdigit() or not 0<=int(key)<=23:raise ValueError('Invalid calibration hour')
        if any(not math.isfinite(float(k)) or not .25<=float(k)<=4 for k in [artifact['k_global'],*artifact['k_by_hour'].values()]):
            raise ValueError('Invalid calibration factor')
        result.update(applied=True,artifact_id=artifact['artifact_id'],reason='Validated correction applied')
    except (KeyError,TypeError,ValueError,OverflowError,AttributeError) as exc:
        result['reason']=str(exc)
    return result


def factors(stamps,artifact):
    tz=ZoneInfo(artifact['timezone'])
    result=[]
    for stamp in stamps:
        if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=ZoneInfo(os.getenv('STORE_TZ','Europe/Paris')))
        hour=stamp.astimezone(tz).hour
        result.append(float(artifact['k_by_hour'].get(str(hour),artifact['k_global'])))
    return result
