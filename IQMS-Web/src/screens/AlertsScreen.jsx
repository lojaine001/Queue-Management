import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';
import { useToast } from '../context/ToastContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const STYLE = {
  red:    { bg: '#2d1a1a', border: '#f85149', color: '#f85149' },
  orange: { bg: '#2d2218', border: '#db6d28', color: '#db6d28' },
  yellow: { bg: '#2d2a1a', border: '#d29922', color: '#d29922' },
};

export default function AlertsScreen() {
  const { t } = useLang();
  const showToast = useToast();
  const { data, loading, error, refresh, lastUpdated } = useApi([`${API_URL}/alerts`]);
  const [alert] = data;

  const level = alert?.level;
  const style = level ? (STYLE[level] ?? STYLE.yellow) : null;
  const label = level === 'red' ? t.critical : level === 'orange' ? t.urgent : t.warning;

  async function sendResponse(response) {
    try {
      await fetch(`${API_URL}/alert-response`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ response }),
      });
      showToast(t.responseRecorded);
      refresh();
    } catch {
      showToast(t.serverError, 'error');
    }
  }

  return (
    <div className="screen-page">
      <div className="section-header">
        <span className="section-title">{t.alerts}</span>
        <div style={s.headerRight}>
          {level && <span style={s.countBadge}>{t.active(1)}</span>}
          <UpdatedAgo lastUpdated={lastUpdated} />
        </div>
      </div>

      {error && <div style={s.errorHint}>{error}</div>}

      {loading && <Skeleton height={140} radius={16} />}

      {!loading && !level && (
        <div style={s.emptyWrap}>
          <div style={s.emptyBadge}>
            <span style={s.emptyIcon}>✓</span>
          </div>
          <div style={s.emptyText}>{t.noAlerts}</div>
          <div style={s.emptySub}>{t.allGood}</div>
        </div>
      )}

      {!loading && level && style && (
        <div style={{ ...s.card, background: style.bg, borderColor: style.border }}>
          <div style={s.cardTop}>
            <span style={{ ...s.badge, color: style.color, borderColor: style.border }}>{label}</span>
          </div>
          <div style={s.message}>{alert.message}</div>
          {alert.predicted_wait_min != null && (
            <div style={s.predicted}>{t.predictedWait(Math.round(alert.predicted_wait_min))}</div>
          )}
          <div style={s.actions}>
            <button
              onClick={() => sendResponse('opening_lane')}
              style={{ ...s.actionBtn, ...s.primaryBtn, background: style.border }}
            >
              {t.openBtn}
            </button>
            <button onClick={() => sendResponse('cannot_open')} style={s.actionBtn}>
              {t.cannotOpen}
            </button>
            <button onClick={() => sendResponse('false_alarm')} style={s.actionBtn}>
              {t.falseAlarm}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const s = {
  headerRight: { display: 'flex', alignItems: 'center', gap: 10 },
  countBadge: {
    fontSize: 11, fontWeight: 700, color: '#f85149',
    background: '#2d1a1a', border: '1px solid #f85149', borderRadius: 10, padding: '2px 8px',
  },
  card: { background: 'var(--card-bg)', border: '1px solid', borderRadius: 16, padding: '18px 20px' },
  cardTop: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  badge: { fontSize: 10, fontWeight: 700, letterSpacing: 0.8, border: '1px solid', borderRadius: 4, padding: '2px 6px' },
  message: { fontSize: 14, color: '#e6edf3', lineHeight: 1.4 },
  predicted: { fontSize: 12, color: '#8b949e', marginTop: 6, fontFamily: 'var(--font-mono)' },
  actions: { display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' },
  actionBtn: {
    background: '#1c2128', border: '1px solid #30363d', borderRadius: 8,
    padding: '8px 14px', color: '#e6edf3', fontSize: 12, fontWeight: 500,
  },
  primaryBtn: { border: 'none', color: '#0d1117', fontWeight: 700 },
  emptyWrap: { display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '70px 20px', gap: 14 },
  emptyBadge: {
    width: 64, height: 64, borderRadius: '50%',
    background: 'rgba(63, 185, 80, 0.12)', border: '1px solid rgba(63, 185, 80, 0.35)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  emptyIcon: { fontSize: 28, color: '#3fb950', fontWeight: 700 },
  emptyText: { fontSize: 16, fontWeight: 600, color: '#e6edf3' },
  emptySub: { fontSize: 13, color: '#8b949e' },
  hint: { color: '#8b949e', fontSize: 13 },
  errorHint: { color: '#f85149', fontSize: 13 },
};
