"""Read-only prospective evaluation. Candidate output requires --candidate-output.

Never overwrites calibration.json or changes the live toggle. Historical mutable
queue_predictions and retrospective backfills are deliberately excluded.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    import argparse,json,csv
    from contextlib import closing
    from prediction.accuracy import load_arrival_pairs,evaluate,evaluate_wait_observations
    from prediction.arrival_calibration import model_weights
    from prediction.shared_forecast_store import connect
    from prediction.model_bundle import current_bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=int,default=30)
    parser.add_argument('--lead-minutes',type=int,default=10)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--wait-observations',type=Path,help='CSV of independently measured waiting times, excluding service')
    parser.add_argument('--candidate-output',type=Path)
    args=parser.parse_args()
    if args.days<1 or args.lead_minutes<0:parser.error('days must be positive; lead must be nonnegative')
    active=Path(__file__).with_name('calibration.json')
    for output in (args.report,args.candidate_output):
        if output and output.resolve()==active.resolve():parser.error('The live calibration file cannot be an evaluation output')
    with closing(connect()) as conn:
        conn.set_session(readonly=True,autocommit=True)
        with conn.cursor() as cur:
            cur.execute('SELECT config FROM forecast_settings WHERE id=1')
            config=cur.fetchone()[0]
        frame=load_arrival_pairs(conn,args.days,args.lead_minutes)
        version=current_bundle(Path(__file__).with_name('models')).name
        report,candidate=evaluate(frame,model_weights(config['arrival_models']),version,args.lead_minutes)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('wait_forecast_runs')")
            if cur.fetchone()[0]:
                cur.execute('SELECT COUNT(*),MIN(published_at),MAX(published_at) FROM wait_forecast_runs')
                count,start,end=cur.fetchone();report['retained_wait_forecasts']=dict(count=count,start=start,end=end)
        if args.wait_observations:
            with args.wait_observations.open(encoding='utf8',newline='') as stream:
                observations=list(csv.DictReader(stream))
            report['wait_accuracy']=evaluate_wait_observations(conn,observations,args.lead_minutes)
    text=json.dumps(report,indent=2,default=str)
    print(text)
    if args.report:args.report.write_text(text,encoding='utf8')
    if args.candidate_output:
        if candidate:args.candidate_output.write_text(json.dumps(candidate,indent=2),encoding='utf8')
        else:print('No candidate written: independent validation did not pass.')
    return 0


if __name__=='__main__':raise SystemExit(main())
