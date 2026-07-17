import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import CameraPlaceholder from '../components/CameraPlaceholder';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const SNAP_INTERVAL = 30000;
const WAIT_THRESHOLD_MIN = 5;

const STATUS_WORD = { open: 'openWord', busy: 'busyWord', busy_high: 'busyWord', closed: 'closedWord' };
const STATUS_COLOR = {
  open: 'var(--green)', busy: 'var(--amber)', busy_high: 'var(--amber)', closed: 'var(--text-off)',
};

function LaneRow({ lane, t }) {
  const status = lane.status ?? 'closed';
  const isClosed = status === 'closed';
  const overThreshold = !isClosed && lane.avg_wait_min > WAIT_THRESHOLD_MIN;

  return (
    <tr style={{ background: overThreshold ? 'var(--raised)' : 'transparent' }}>
      <td>{t.laneLabel(lane.lane_number)}</td>
      <td className="mono" style={{ textAlign: 'right' }}>{isClosed ? '—' : lane.waiting}</td>
      <td
        className="mono"
        style={{ textAlign: 'right', color: overThreshold ? 'var(--amber)' : 'var(--text-2)' }}
      >
        {isClosed ? '—' : `${Math.round(lane.avg_wait_min)} min`}
      </td>
      <td style={{ textAlign: 'right', color: STATUS_COLOR[status], fontWeight: 500 }}>
        {t[STATUS_WORD[status]]}
      </td>
    </tr>
  );
}

export default function LiveScreen() {
  const { t } = useLang();
  const { data, loading, error, lastUpdated } = useApi([`${API_URL}/live-lanes`]);
  const [lanesData] = data;

  const { data: snapData } = useApi([
    `${API_URL}/snapshot/entrance`,
    `${API_URL}/snapshot/checkout`,
  ], SNAP_INTERVAL);
  const [entranceSnap, checkoutSnap] = snapData;

  const lanes = lanesData?.lanes ?? [];
  const snapshot = lanesData?.snapshot ?? {};

  return (
    <div className="screen-page">
      <div className="section-header">
        <span className="section-title">{t.liveQueueStatus}</span>
        <UpdatedAgo lastUpdated={lastUpdated} />
      </div>

      {error && <div style={s.errorText}>{error}</div>}

      {loading ? (
        <Skeleton height={40} />
      ) : (
        <div className="kpi-strip" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: 24 }}>
          <div className="kpi-cell">
            <div className="micro-label">{t.inQueue}</div>
            <div className="mono" style={s.kpiValue}>{snapshot.total_in_queue ?? '—'}</div>
          </div>
          <div className="kpi-cell">
            <div className="micro-label">{t.avgWait}</div>
            <div
              className="mono"
              style={{ ...s.kpiValue, color: snapshot.avg_wait_min > WAIT_THRESHOLD_MIN ? 'var(--amber)' : 'var(--text)' }}
            >
              {snapshot.avg_wait_min != null ? `${Math.round(snapshot.avg_wait_min)} min` : '—'}
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <Skeleton height={200} />
      ) : (
        <table className="ruled-table">
          <tbody>
            {lanes.map(lane => <LaneRow key={lane.lane_id} lane={lane} t={t} />)}
          </tbody>
        </table>
      )}

      <div className="section-header">
        <span className="section-title">{t.liveCameras}</span>
      </div>
      <div className="camera-grid">
        <CameraPlaceholder
          label={t.camEntrance} camId="1"
          dataUrl={entranceSnap?.image}
        />
        <CameraPlaceholder
          label={t.camCheckout} camId="2"
          dataUrl={checkoutSnap?.image}
          metric={snapshot.total_in_queue != null ? t.inQueueChip(snapshot.total_in_queue) : null}
        />
      </div>
    </div>
  );
}

const s = {
  kpiValue: { fontSize: 24, marginTop: 6 },
  errorText: { color: 'var(--red)', fontSize: 13, padding: '8px 0' },
};
