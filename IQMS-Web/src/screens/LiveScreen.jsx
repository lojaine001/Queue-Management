import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import CameraPlaceholder from '../components/CameraPlaceholder';

const LANE_STATUS = {
  closed:    { color: '#484f58', label: 'CLOSED', bg: '#21262d' },
  open:      { color: '#3fb950', label: 'OPEN',   bg: '#1a2e22' },
  busy:      { color: '#d29922', label: 'BUSY',   bg: '#2d2a1a' },
  busy_high: { color: '#f85149', label: 'BUSY',   bg: '#2d1a1a' },
};

function statusKey(lane) {
  if (!lane || lane.status === 'closed') return 'closed';
  if (lane.fill_pct >= 80) return 'busy_high';
  if (lane.fill_pct >= 50) return 'busy';
  return 'open';
}

function LaneCard({ lane }) {
  const key = statusKey(lane);
  const st = LANE_STATUS[key];
  return (
    <div style={{ ...s.laneCard, background: st.bg, borderColor: st.color + '44' }}>
      <div style={s.laneLeft}>
        <div style={{ ...s.laneIcon, color: st.color }}>
          {key === 'closed' ? '👤' : key.startsWith('busy') ? '🟠' : '🟢'}
        </div>
        <div>
          <div style={s.laneName}>LANE {lane.lane_id}</div>
          <div style={s.laneSub}>{lane.lane_type || 'Caisse standard'}</div>
        </div>
      </div>
      <div style={s.laneRight}>
        <span style={{ ...s.laneStatus, color: st.color }}>{st.label}</span>
        <span style={s.laneCount}>
          {lane.status === 'closed' ? '0' : (lane.queue_length ?? '—')}{' '}
          <span style={s.laneCountSub}>clients</span>
        </span>
      </div>
    </div>
  );
}

export default function LiveScreen() {
  const { data, loading, error } = useApi([
    `${API_URL}/live-lanes`,
    `${API_URL}/alerts`,
  ]);
  const [lanesData, alertsData] = data;

  const lanes = lanesData?.lanes ?? [];
  const snapshot = lanesData?.snapshot ?? {};
  const openLanes = lanes.filter(l => l.status !== 'closed');
  const totalInQueue = openLanes.reduce((a, l) => a + (l.queue_length ?? 0), 0);

  return (
    <div style={s.page}>
      {/* Header */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>LIVE QUEUE STATUS</span>
        <div style={s.liveBadge}>
          <span style={s.liveDot} />
          <span style={s.liveText}>LIVE</span>
        </div>
      </div>

      {loading && <div style={s.hint}>Chargement…</div>}
      {error && <div style={s.errorHint}>{error}</div>}

      {/* Lanes */}
      <div style={s.lanesGroup}>
        {lanes.length === 0 && !loading && (
          <div style={s.hint}>Aucune donnée de file disponible.</div>
        )}
        {lanes.map(lane => <LaneCard key={lane.lane_id} lane={lane} />)}
      </div>

      {/* Snapshot */}
      {(snapshot.in_queue != null || snapshot.avg_wait_min != null) && (
        <div style={s.snapshotRow}>
          <div style={s.snapshotCard}>
            <div style={s.snapValue}>{snapshot.in_queue ?? '—'}</div>
            <div style={s.snapLabel}>EN FILE</div>
            <div style={s.snapSub}>{openLanes.length} file{openLanes.length !== 1 ? 's' : ''} ouvertes</div>
          </div>
          <div style={s.snapshotCard}>
            <div style={{ ...s.snapValue, color: snapshot.avg_wait_min > 10 ? '#f85149' : '#3fb950' }}>
              {snapshot.avg_wait_min != null ? `${Math.round(snapshot.avg_wait_min)} min` : '—'}
            </div>
            <div style={s.snapLabel}>ATTENTE MOY.</div>
          </div>
        </div>
      )}

      {/* Live cameras */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>LIVE CAMERAS</span>
      </div>
      <div style={s.cameraRow}>
        <CameraPlaceholder label="CAM 1 – ENTRÉE" />
        <CameraPlaceholder label="CAM 2 – CAISSES" />
      </div>
    </div>
  );
}

const s = {
  page: { padding: '16px 16px 80px' },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
    marginTop: 16,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: 700,
    color: '#8b949e',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  liveBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: '#3fb950',
    boxShadow: '0 0 6px #3fb950',
  },
  liveText: { fontSize: 11, fontWeight: 700, color: '#3fb950', letterSpacing: 0.8 },
  lanesGroup: { display: 'flex', flexDirection: 'column', gap: 8 },
  laneCard: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 14px',
    border: '1px solid',
    borderRadius: 8,
  },
  laneLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  laneIcon: { fontSize: 20 },
  laneName: { fontSize: 14, fontWeight: 700, color: '#e6edf3' },
  laneSub: { fontSize: 11, color: '#8b949e', marginTop: 2 },
  laneRight: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 },
  laneStatus: { fontSize: 11, fontWeight: 700, letterSpacing: 0.8 },
  laneCount: { fontSize: 22, fontWeight: 700, color: '#e6edf3' },
  laneCountSub: { fontSize: 12, fontWeight: 400, color: '#8b949e' },
  snapshotRow: { display: 'flex', gap: 10, margin: '14px 0' },
  snapshotCard: {
    flex: 1,
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '12px 14px',
  },
  snapValue: { fontSize: 26, fontWeight: 700, color: '#e6edf3' },
  snapLabel: { fontSize: 10, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginTop: 2 },
  snapSub: { fontSize: 11, color: '#484f58', marginTop: 2 },
  cameraRow: { display: 'flex', gap: 10 },
  hint: { color: '#8b949e', fontSize: 13, padding: '10px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '10px 0' },
};
