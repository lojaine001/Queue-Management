import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import CameraPlaceholder from '../components/CameraPlaceholder';

const SNAP_INTERVAL = 30000;

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

function LaneCard({ lane, t }) {
  const key = statusKey(lane);
  const st = LANE_STATUS[key];
  const isClosed = key === 'closed';
  const hasCount = !isClosed && lane.queue_length != null;
  return (
    <div style={{ ...s.laneCard, background: st.bg, borderColor: st.color + (isClosed ? '33' : '55') }}>
      <div style={s.laneLeft}>
        <div style={{ ...s.laneCircle, background: st.color + '22', border: `1.5px solid ${st.color}` }}>
          <span style={{ color: st.color, fontSize: 13 }}>👤</span>
        </div>
        <div>
          <div style={s.laneName}>LANE {lane.lane_id + 1}</div>
          <div style={s.laneSub}>{lane.lane_type || 'Caisse standard'}</div>
        </div>
      </div>
      <div style={s.laneRight}>
        <span style={{ ...s.laneStatusLabel, color: st.color }}>{st.label}</span>
        <span style={{ ...s.laneCount, color: isClosed ? '#484f58' : '#e6edf3' }}>
          {hasCount || isClosed ? (isClosed ? 0 : lane.queue_length) : <span style={s.laneDash}>—</span>}
          {' '}<span style={s.laneCountSub}>{t.clients}</span>
        </span>
      </div>
    </div>
  );
}

export default function LiveScreen() {
  const { t } = useLang();
  const { data, loading, error } = useApi([
    `${API_URL}/live-lanes`,
    `${API_URL}/alerts`,
  ]);
  const [lanesData] = data;

  const { data: snapData } = useApi([
    `${API_URL}/snapshot/entrance`,
    `${API_URL}/snapshot/checkout`,
  ], SNAP_INTERVAL);
  const [entranceSnap, checkoutSnap] = snapData;

  const lanes = lanesData?.lanes ?? [];
  const snapshot = lanesData?.snapshot ?? {};
  const openLanes = lanes.filter(l => l.status !== 'closed');

  return (
    <div className="screen-page">
      {/* ── Live grid: lanes | right panel ── */}
      <div className="live-grid">
        {/* Left: lanes */}
        <div>
          <div className="section-header">
            <span className="section-title">{t.liveQueueStatus}</span>
            <div style={s.liveBadge}>
              <span style={s.liveDot} />
              <span style={{ fontSize: 11, fontWeight: 700, color: '#3fb950', letterSpacing: 0.8 }}>{t.live}</span>
            </div>
          </div>
          {loading && <div style={s.hint}>{t.loading}</div>}
          {error && <div style={s.errorHint}>{error}</div>}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {lanes.map(lane => <LaneCard key={lane.lane_id} lane={lane} t={t} />)}
            {lanes.length === 0 && !loading && (
              <div style={s.hint}>—</div>
            )}
          </div>
        </div>

        {/* Right: snapshot + cameras */}
        <div>
          {(snapshot.in_queue != null || snapshot.avg_wait_min != null) && (
            <>
              <div className="section-header" style={{ marginTop: 0 }}>
                <span className="section-title">SNAPSHOT</span>
              </div>
              <div style={s.snapshotRow}>
                <div style={s.snapshotCard}>
                  <div style={s.snapValue}>{snapshot.in_queue ?? '—'}</div>
                  <div style={s.snapLabel}>{t.inQueue}</div>
                  <div style={s.snapSub}>{t.openLanes(openLanes.length)}</div>
                </div>
                <div style={s.snapshotCard}>
                  <div style={{ ...s.snapValue, color: (snapshot.avg_wait_min ?? 0) > 10 ? '#f85149' : '#3fb950' }}>
                    {snapshot.avg_wait_min != null ? `${Math.round(snapshot.avg_wait_min)} min` : '—'}
                  </div>
                  <div style={s.snapLabel}>{t.avgWait}</div>
                </div>
              </div>
            </>
          )}

        </div>
      </div>

      {/* ── Cameras: full-width row on desktop, stacked on mobile ── */}
      <div className="section-header">
        <span className="section-title">{t.liveCameras}</span>
      </div>
      <div className="camera-grid">
        <CameraPlaceholder label="CAM 1 – ENTRÉE"  dataUrl={entranceSnap?.image} />
        <CameraPlaceholder label="CAM 2 – CAISSES" dataUrl={checkoutSnap?.image} />
      </div>
    </div>
  );
}

const s = {
  liveBadge: { display: 'flex', alignItems: 'center', gap: 5 },
  liveDot: {
    width: 6, height: 6, borderRadius: '50%',
    background: '#3fb950', boxShadow: '0 0 6px #3fb950',
  },
  laneCard: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 20px', border: '1px solid', borderRadius: 16,
  },
  laneLeft: { display: 'flex', alignItems: 'center', gap: 14 },
  laneCircle: {
    width: 36, height: 36, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  },
  laneName: { fontSize: 14, fontWeight: 700, color: '#e6edf3' },
  laneSub: { fontSize: 11, color: '#8b949e', marginTop: 2 },
  laneRight: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 },
  laneStatusLabel: { fontSize: 11, fontWeight: 700, letterSpacing: 0.8 },
  laneCount: { fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' },
  laneCountSub: { fontSize: 12, fontWeight: 400, fontFamily: 'var(--font-ui)', color: '#8b949e' },
  laneDash: { color: '#484f58' },
  snapshotRow: { display: 'flex', gap: 14, marginBottom: 4 },
  snapshotCard: {
    flex: 1, background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    borderRadius: 16, padding: '16px 18px',
  },
  snapValue: { fontSize: 28, fontWeight: 700, color: '#e6edf3', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' },
  snapLabel: { fontSize: 10, fontWeight: 700, color: '#8b95a8', letterSpacing: 0.8, marginTop: 6, textTransform: 'uppercase' },
  snapSub: { fontSize: 11, color: '#484f58', marginTop: 4 },
  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
