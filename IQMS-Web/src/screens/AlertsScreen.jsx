import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';
import { useToast } from '../context/ToastContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const LEVEL_COLOR = { red: 'var(--red)', orange: 'var(--amber)', yellow: 'var(--amber)' };

function levelWord(level, t) {
  return level === 'red' ? t.critical : level === 'orange' ? t.urgent : t.warning;
}

export default function AlertsScreen() {
  const { t } = useLang();
  const showToast = useToast();
  const { data, loading, error, refresh, lastUpdated } = useApi([
    `${API_URL}/alerts`,
    `${API_URL}/alert-history`,
  ]);
  const [alert, historyData] = data;
  const history = historyData?.history ?? [];

  const level = alert?.level;

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
        <UpdatedAgo lastUpdated={lastUpdated} />
      </div>

      {error && <div style={s.errorText}>{error}</div>}

      {loading ? (
        <Skeleton height={80} style={{ marginBottom: 24 }} />
      ) : level ? (
        <div className="hairline-top hairline-bottom" style={s.activeBlock}>
          <span style={{ ...s.marker, background: 'var(--red)' }} />
          <div style={{ flex: 1 }}>
            <div style={s.activeTitle}>{levelWord(level, t)}</div>
            <div style={s.activeMessage}>{alert.message}</div>
            {alert.predicted_wait_min != null && (
              <div className="mono" style={s.activePredicted}>{t.predictedWait(Math.round(alert.predicted_wait_min))}</div>
            )}
            <div style={s.actions}>
              <button onClick={() => sendResponse('opening_lane')} style={{ ...s.actionBtn, color: 'var(--text)' }}>
                {t.openBtn}
              </button>
              <button onClick={() => sendResponse('cannot_open')} style={s.actionBtn}>{t.cannotOpen}</button>
              <button onClick={() => sendResponse('false_alarm')} style={s.actionBtn}>{t.falseAlarm}</button>
            </div>
          </div>
        </div>
      ) : (
        <div style={s.emptyBlock}>
          <div style={s.emptyLine1}>{t.noAlerts}</div>
          <div style={s.emptyLine2}>{t.allGood}</div>
        </div>
      )}

      <div className="section-header">
        <span className="section-title">{t.alertHistory}</span>
      </div>

      {loading ? (
        <Skeleton height={160} />
      ) : history.length > 0 ? (
        <table className="ruled-table">
          <tbody>
            {history.map((h, i) => (
              <tr key={i}>
                <td className="mono" style={{ color: 'var(--text-2)' }}>{h.timestamp}</td>
                <td className="mono" style={{ textAlign: 'right', color: 'var(--text-2)' }}>{h.duration_min} min</td>
                <td style={{ textAlign: 'right', color: LEVEL_COLOR[h.level] ?? 'var(--text-2)', fontWeight: 500 }}>
                  {levelWord(h.level, t)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={s.noHistory}>{t.noAlertHistory}</div>
      )}
    </div>
  );
}

const s = {
  activeBlock: {
    display: 'flex', alignItems: 'flex-start', gap: 12,
    background: 'var(--surface)', padding: '14px 4px', marginBottom: 8,
  },
  marker: { width: 8, height: 8, marginTop: 4, flexShrink: 0 },
  activeTitle: { fontSize: 11, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase', color: 'var(--red)' },
  activeMessage: { fontSize: 14, color: 'var(--text)', marginTop: 6, lineHeight: 1.4 },
  activePredicted: { fontSize: 12, color: 'var(--text-2)', marginTop: 4 },
  actions: { display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap' },
  actionBtn: { fontSize: 12, fontWeight: 500, color: 'var(--text-2)' },
  emptyBlock: { padding: '24px 0' },
  emptyLine1: { fontSize: 14, color: 'var(--text)' },
  emptyLine2: { fontSize: 13, color: 'var(--text-2)', marginTop: 4 },
  noHistory: { fontSize: 13, color: 'var(--text-3)', padding: '16px 0' },
  errorText: { color: 'var(--red)', fontSize: 13, padding: '8px 0' },
};
