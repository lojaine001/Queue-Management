"""Today's entrance demographics/traffic, refreshed without browser sessions."""
import json
from contextlib import closing
from datetime import datetime,timezone
from psycopg2.extras import Json
from prediction.shared_forecast_store import connect,ensure_schema
from prediction.arrival_calibration import target_definition


def refresh_statistics():
    camera=target_definition()['camera_id']
    with closing(connect()) as conn:
        ensure_schema(conn)
        with conn,conn.cursor() as cur:
            cur.execute("""SELECT gender,COUNT(*) FROM entrance_events
                WHERE camera_id=%s AND timestamp>=CURRENT_DATE AND timestamp<=NOW()
                AND gender IS NOT NULL AND gender!='unknown' GROUP BY gender""",(camera,))
            rows=cur.fetchall();total=sum(n for _,n in rows)
            gender=[dict(key=g.lower(),label=g.capitalize(),count=n,percent=round(100*n/total) if total else 0,
                         color={'male':'#2563eb','female':'#db2777'}.get(g.lower(),'#94a3b8')) for g,n in rows]
            cur.execute("""SELECT CASE WHEN age_estimate<30 THEN '18-30' WHEN age_estimate<50 THEN '30-50' ELSE '50+' END,COUNT(*)
                FROM entrance_events WHERE camera_id=%s AND timestamp>=CURRENT_DATE AND timestamp<=NOW()
                AND age_estimate IS NOT NULL GROUP BY 1 ORDER BY 1""",(camera,))
            rows=cur.fetchall();total=sum(n for _,n in rows)
            age=[dict(group=g,count=n,percent=round(100*n/total) if total else 0,
                      color={'18-30':'#f97316','30-50':'#3fb950','50+':'#58a6ff'}[g]) for g,n in rows]
            cur.execute("""SELECT date_trunc('hour',timestamp),COUNT(*) FROM entrance_events
                WHERE camera_id=%s AND timestamp>=CURRENT_DATE AND timestamp<=NOW() GROUP BY 1 ORDER BY 1""",(camera,))
            rows=cur.fetchall();peak=max((n for _,n in rows),default=0)
            hourly=[dict(hour=t.strftime('%H:00'),count=n,is_peak=n==peak and peak>0) for t,n in rows]
            payload=dict(gender=gender,age=age)
            cur.execute('''CREATE TABLE IF NOT EXISTS dashboard_statistics (
                id INTEGER PRIMARY KEY CHECK(id=1),updated_at TIMESTAMPTZ NOT NULL,
                camera_id TEXT NOT NULL,demographics JSONB NOT NULL,entries_hour JSONB NOT NULL)''')
            cur.execute('''INSERT INTO dashboard_statistics VALUES (1,NOW(),%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET updated_at=EXCLUDED.updated_at,camera_id=EXCLUDED.camera_id,
                demographics=EXCLUDED.demographics,entries_hour=EXCLUDED.entries_hour''',(camera,Json(payload),Json(hourly)))
            cur.execute('UPDATE dashboard_state SET demographics_json=%s,entries_hour_json=%s WHERE id=1',
                        (json.dumps(payload),json.dumps(hourly)))
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),camera_id=camera,gender_total=sum(g['count'] for g in gender))
