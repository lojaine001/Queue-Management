import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';

function AlertCard({ alert, t, onAction }) {
  const STYLE = {
    red:    { bg: '#2d1a1a', border: '#f85149', label: t.critical, color: '#f85149' },
    orange: { bg: '#2d2218', border: '#db6d28', label: t.urgent,   color: '#db6d28' },
    yellow: { bg: '#2d2a1a', border: '#d29922', label: t.warning,  color: '#d29922' },
  };
  const style = STYLE[alert.color] ?? STYLE.yellow;
  return (
    <div style={{ ...s.card, background: style.bg, borderColor: style.border }}>
      <div style={s.cardTop}>
        <span style={{ ...s.badge, color: style.color, borderColor: style.border }}>{style.label}</span>
        <span style={s.time}>{alert.time ?? ''}</span>
      </div>
      <div style={s.message}>{alert.message}</div>
      {alert.lane_id && <div style={s.lane}>Lane {alert.lane_id}</div>}
      {alert.actions?.length > 0 && (
        <div style={s.actions}>
          {alert.actions.map(a => (
            <button key={a.id} onClick={() => onAction(alert.id, a.id)} style={s.actionBtn}>
              {a.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function AlertsScreen() {
  const { t } = useLang();
  const { data, loading, error, refresh } = useApi([`${API_URL}/alerts`]);
  const [alertsData] = data;

  const raw = alertsData?.alerts ?? alertsData ?? [];
  const list = Array.isArray(raw) ? raw : [];

  async function handleAction(alertId, actionId) {
    try {
      await fetch(`${API_URL}/alert-response`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: alertId, action: actionId }),
      });
      refresh();
    } catch { /* ignore */ }
  }

  return (
    <div className="screen-page">
      <div className="section-header">
        <span className="section-title">{t.alerts}</span>
        {list.length > 0 && (
          <span style={s.countBadge}>{t.active(list.length)}</span>
        )}
      </div>

      {loading && <div style={s.hint}>{t.loading}</div>}
      {error && <div style={s.errorHint}>{error}</div>}

      {!loading && list.length === 0 && (
        <div style={s.emptyWrap}>
          <div style={s.emptyBadge}>
            <span style={s.emptyIcon}>✓</span>
          </div>
          <div style={s.emptyText}>{t.noAlerts}</div>
          <div style={s.emptySub}>{t.allGood}</div>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {list.map((alert, i) => (
          <AlertCard key={alert.id ?? i} alert={alert} t={t} onAction={handleAction} />
        ))}
      </div>
    </div>
  );
}

const s = {
  countBadge: {
    fontSize: 11, fontWeight: 700, color: '#f85149',
    background: '#2d1a1a', border: '1px solid #f85149', borderRadius: 10, padding: '2px 8px',
  },
  card: { background: 'var(--card-bg)', border: '1px solid', borderRadius: 16, padding: '18px 20px' },
  cardTop: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  badge: { fontSize: 10, fontWeight: 700, letterSpacing: 0.8, border: '1px solid', borderRadius: 4, padding: '2px 6px' },
  time: { fontSize: 11, color: '#484f58' },
  message: { fontSize: 14, color: '#e6edf3', lineHeight: 1.4 },
  lane: { fontSize: 11, color: '#8b949e', marginTop: 6 },
  actions: { display: 'flex', gap: 8, marginTop: 12 },
  actionBtn: {
    background: '#1c2128', border: '1px solid #30363d', borderRadius: 8,
    padding: '8px 14px', color: '#e6edf3', fontSize: 12, fontWeight: 500,
  },
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
