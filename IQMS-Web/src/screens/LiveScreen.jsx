import { useEffect, useRef, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import { useToast } from '../context/ToastContext';
import CameraPlaceholder from '../components/CameraPlaceholder';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const SNAP_INTERVAL = 30000;
const GAUGE_MAX_MIN = 12; // top of the gauge — matches the 9-12+ top zone

// Count-driven lane color: 0 = idle/closed, 1 = green, 2-3 = orange, 4+ = red
function countColor(count) {
  if (!count) return { color: '#484f58', bg: '#21262d', borderWidth: 4 };
  if (count === 1) return { color: '#3fb950', bg: '#1a2e22', borderWidth: 4 };
  if (count <= 3) return { color: '#db6d28', bg: '#2d2218', borderWidth: 4 };
  return { color: '#ff3b30', bg: '#3a1210', borderWidth: 6 };
}

function LaneCard({ lane, t }) {
  const count = lane.waiting ?? 0;
  const isClosed = lane.status === 'closed';
  const st = countColor(isClosed ? 0 : count);

  return (
    <div style={{ ...s.laneCard, borderLeftColor: st.color, borderLeftWidth: st.borderWidth, background: st.bg }}>
      <div style={s.laneLeft}>
        <div style={{ ...s.laneIcon, background: st.color + '22', border: `1.5px solid ${st.color}` }}>
          <span style={{ color: st.color, fontSize: 13 }}>👤</span>
        </div>
        <div>
          <div style={s.laneName}>LANE {Number(lane.lane_id) + 1}</div>
          <div style={s.laneSub}>{isClosed ? t.closedLabel : (lane.lane_type || 'Caisse standard')}</div>
        </div>
      </div>
      <div style={s.laneRight}>
        <span style={{ ...s.laneCount, color: isClosed ? '#484f58' : '#e6edf3' }}>
          {isClosed ? 0 : count}
        </span>
        <span style={s.laneCountSub}>{t.clients}</span>
      </div>
    </div>
  );
}

function Toggle({ on, onChange }) {
  return (
    <button
      onClick={() => onChange(!on)}
      style={{ ...s.toggle, background: on ? '#3fb950' : '#30363d' }}
    >
      <span style={{ ...s.toggleKnob, transform: on ? 'translateX(18px)' : 'translateX(0)' }} />
    </button>
  );
}

function AlertsBar({ enabled, onToggle, threshold, onThreshold, t }) {
  return (
    <div style={s.alertsBar}>
      <span style={s.bellIcon}>🔔</span>
      <span style={s.alertsLabel}>{t.alertsLabel}</span>
      <Toggle on={enabled} onChange={onToggle} />
      <span style={s.divider} />
      <span style={s.seuilLabel}>{t.seuilLabel}</span>
      <input
        type="range"
        min={1} max={15} step={1}
        value={threshold}
        onChange={e => onThreshold(Number(e.target.value))}
        style={s.slider}
      />
      <span className="mono" style={s.seuilValue}>{threshold} min</span>
    </div>
  );
}

function Gauge({ current, threshold, t }) {
  const clamp = v => Math.max(0, Math.min(GAUGE_MAX_MIN, v ?? 0));
  const currentPct = (clamp(current) / GAUGE_MAX_MIN) * 100;
  const thresholdPct = (clamp(threshold) / GAUGE_MAX_MIN) * 100;
  const overThreshold = current != null && current >= threshold;

  return (
    <div style={s.gaugeWrap}>
      <div style={s.gaugeLabel}>{t.avgWait}</div>
      <div style={s.gaugeBody}>
        <div style={s.gaugeBar}>
          <div style={{ ...s.gaugeZone, background: '#f85149' }} />
          <div style={{ ...s.gaugeZone, background: '#db6d28' }} />
          <div style={{ ...s.gaugeZone, background: '#d29922' }} />
          <div style={{ ...s.gaugeZone, background: '#3fb950' }} />
        </div>

        {/* Live wait — red marker, value sits right next to it at its actual position */}
        <div style={{ ...s.markerGroup, left: -70, bottom: `calc(${currentPct}% - 8px)` }}>
          <span className="mono" style={{ fontSize: 14, fontWeight: 700, color: overThreshold ? '#f85149' : '#e6edf3' }}>
            {current != null ? `${Math.round(current)}m` : '—'}
          </span>
          <span style={{ ...s.marker, ...s.markerRed }} />
        </div>

        {/* Seuil threshold — white marker, value sits next to it, moves live with the slider */}
        <div style={{ ...s.markerGroup, right: -66, bottom: `calc(${thresholdPct}% - 8px)` }}>
          <span style={{ ...s.marker, ...s.markerWhite }} />
          <span className="mono" style={{ fontSize: 12, color: '#8b949e' }}>{threshold}m</span>
        </div>
      </div>
    </div>
  );
}

export default function LiveScreen() {
  const { t } = useLang();
  const showToast = useToast();

  const [alertsEnabled, setAlertsEnabled] = useState(
    () => localStorage.getItem('iqms_alerts_enabled') === 'true'
  );
  const [threshold, setThreshold] = useState(
    () => Number(localStorage.getItem('iqms_alert_threshold')) || 8
  );
  const wasOverRef = useRef(false);

  useEffect(() => {
    localStorage.setItem('iqms_alerts_enabled', String(alertsEnabled));
  }, [alertsEnabled]);

  useEffect(() => {
    localStorage.setItem('iqms_alert_threshold', String(threshold));
  }, [threshold]);

  const { data, loading, error, lastUpdated } = useApi([
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
  const avgWait = snapshot.avg_wait_min;

  // Fire a popup only on the rising edge (crossing into alert), not every
  // refresh. While disabled, keep resetting the tracker so turning alerts
  // back on always gets a fresh chance to fire if already over threshold —
  // otherwise a crossing that happened while OFF silently "used up" the
  // rising edge and nothing would ever show once you turned it back on.
  useEffect(() => {
    if (!alertsEnabled) {
      wasOverRef.current = false;
      return;
    }
    if (avgWait == null) return;
    const isOver = avgWait >= threshold;
    if (isOver && !wasOverRef.current) {
      showToast(t.alertPopupMessage(Math.round(avgWait), threshold), 'error', 6000);
    }
    wasOverRef.current = isOver;
  }, [avgWait, threshold, alertsEnabled]);

  return (
    <div className="screen-page">
      <AlertsBar
        enabled={alertsEnabled} onToggle={setAlertsEnabled}
        threshold={threshold} onThreshold={setThreshold}
        t={t}
      />

      <div className="live-grid">
        <div>
          <div className="section-header">
            <span style={s.bigTitle}>{t.liveQueueStatus}</span>
            <div style={s.liveBadgeRow}>
              <div style={s.liveBadge}>
                <span style={s.liveDot} />
                <span style={s.liveWord}>{t.live}</span>
              </div>
              <UpdatedAgo lastUpdated={lastUpdated} fontSize={14} dotSize={7} />
            </div>
          </div>
          {error && <div style={s.errorHint}>{error}</div>}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {loading
              ? [0, 1, 2, 3].map(i => <Skeleton key={i} height={64} radius={16} />)
              : <>
                  {lanes.map(lane => <LaneCard key={lane.lane_id} lane={lane} t={t} />)}
                  {lanes.length === 0 && <div style={s.hint}>—</div>}
                </>
            }
          </div>
        </div>

        <div>
          <div className="section-header" style={{ marginTop: 0 }}>
            <span className="section-title">SNAPSHOT</span>
          </div>
          <Gauge current={avgWait} threshold={threshold} t={t} />
        </div>
      </div>

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
  alertsBar: {
    display: 'flex', alignItems: 'center', gap: 12,
    background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    borderRadius: 16, padding: '14px 20px', marginBottom: 8,
  },
  bellIcon: { fontSize: 16 },
  alertsLabel: { fontSize: 15, fontWeight: 600, color: '#e6edf3', marginRight: 4 },
  toggle: {
    width: 38, height: 20, borderRadius: 10, position: 'relative',
    padding: 2, transition: 'background 0.15s',
  },
  toggleKnob: {
    display: 'block', width: 16, height: 16, borderRadius: '50%',
    background: '#fff', transition: 'transform 0.15s',
  },
  divider: { width: 1, height: 24, background: 'var(--card-border)', margin: '0 4px' },
  seuilLabel: { fontSize: 14, color: '#8b949e' },
  slider: { flex: 1, maxWidth: 240, accentColor: '#58a6ff' },
  seuilValue: { fontSize: 15, fontWeight: 700, color: '#e6edf3', minWidth: 48 },

  bigTitle: { fontSize: 20, fontWeight: 500, color: '#e6edf3' },
  liveBadgeRow: { display: 'flex', alignItems: 'center', gap: 12 },
  liveBadge: { display: 'flex', alignItems: 'center', gap: 5 },
  liveDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: '#3fb950', boxShadow: '0 0 6px #3fb950',
  },
  liveWord: { fontSize: 14, fontWeight: 700, color: '#3fb950', letterSpacing: 0.8 },

  laneCard: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 20px', borderLeft: '4px solid', borderRadius: 12,
  },
  laneLeft: { display: 'flex', alignItems: 'center', gap: 14 },
  laneIcon: {
    width: 36, height: 36, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  },
  laneName: { fontSize: 14, fontWeight: 700, color: '#e6edf3' },
  laneSub: { fontSize: 11, color: '#8b949e', marginTop: 2 },
  laneRight: { display: 'flex', alignItems: 'baseline', gap: 6 },
  laneCount: { fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' },
  laneCountSub: { fontSize: 12, fontWeight: 400, color: '#8b949e' },

  gaugeWrap: {
    background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    borderRadius: 16, padding: '20px 80px', display: 'flex', flexDirection: 'column', alignItems: 'center',
  },
  gaugeLabel: { fontSize: 13, color: '#8b949e', marginBottom: 14 },
  gaugeBody: { position: 'relative', width: 40, height: 220 },
  gaugeBar: {
    width: '100%', height: '100%', borderRadius: 20, overflow: 'hidden',
    display: 'flex', flexDirection: 'column',
  },
  gaugeZone: { flex: 1 },
  markerGroup: {
    position: 'absolute', display: 'flex', alignItems: 'center', gap: 6,
  },
  marker: { width: 0, height: 0, flexShrink: 0 },
  markerRed: {
    borderTop: '7px solid transparent', borderBottom: '7px solid transparent',
    borderLeft: '11px solid #f85149',
  },
  markerWhite: {
    borderTop: '7px solid transparent', borderBottom: '7px solid transparent',
    borderRight: '11px solid #ffffff',
  },

  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
