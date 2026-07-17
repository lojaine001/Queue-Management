import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';
import { useToast } from '../context/ToastContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const SCENARIO_COLOR = { red: '#f85149', orange: '#db6d28', yellow: '#d29922', green: '#3fb950' };

function CustomTooltip({ active, payload, label, t }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700 }}>
          {p.dataKey === 'wait' ? t.waitLegend : t.arrivalsLegend}: {p.value?.toFixed(1)}{p.dataKey === 'wait' ? ' min' : ''}
        </div>
      ))}
    </div>
  );
}

export default function ForecastScreen() {
  const { t } = useLang();
  const showToast = useToast();

  const { data, loading, error, refresh, lastUpdated } = useApi([
    `${API_URL}/forecast`,
    `${API_URL}/forecast-chart`,
  ]);
  const [forecastData, chartData] = data;

  const waitNow = forecastData?.wait_now_min;
  const currentLanes = forecastData?.current_lanes ?? 1;
  const scenarios = forecastData?.lane_scenarios ?? [];

  const points = (chartData?.slots ?? []).map(sl => ({
    t: sl.time,
    wait: sl.wait_min,
    arrivals: sl.arrivals,
  }));

  // Recommendation, computed client-side from the same data the lane scenarios use
  // (the API has no "recommendation" field — dashboard_state only stores raw wait numbers).
  const cw = waitNow ?? 0;
  const betterScenario = scenarios.find(sc => sc.lanes > currentLanes && sc.est_wait_min < cw - 1);
  const worseScenario  = scenarios.find(sc => sc.lanes < currentLanes && sc.est_wait_min <= 5);
  let recText = null;
  let recColor = '#8b949e';
  if (cw <= 5 && worseScenario) {
    recText = t.recLight(worseScenario.lanes);
    recColor = '#3fb950';
  } else if (cw > 5 && betterScenario) {
    const saved = Math.round(cw - betterScenario.est_wait_min);
    recText = t.recOpenMore(betterScenario.lanes - currentLanes, saved);
    recColor = '#db6d28';
  } else if (cw > 10) {
    recText = t.recHighDemand;
    recColor = '#f85149';
  } else if (cw <= 5) {
    recText = t.recOptimal;
    recColor = '#3fb950';
  }

  const setLanes = async (n) => {
    if (n === currentLanes) return;
    try {
      await fetch(`${API_URL}/set-lanes`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ lanes: n }),
      });
      showToast(n > currentLanes ? t.laneOpenedToast(n) : t.laneClosedToast(currentLanes));
      refresh();
    } catch {
      showToast(t.serverError, 'error');
    }
  };

  return (
    <div className="screen-page">
      {/* Recommendation */}
      <div className="section-header">
        <div style={s.forecastTitleRow}>
          <span style={s.clockIcon}>⏱</span>
          <span className="section-title">{t.forecast15min}</span>
        </div>
        <div style={s.headerRight}>
          <span style={s.sub}>{t.recommendation}</span>
          <UpdatedAgo lastUpdated={lastUpdated} />
        </div>
      </div>

      {error && <div style={s.errorHint}>{error}</div>}

      {loading ? (
        <>
          <Skeleton height={64} radius={16} style={{ marginBottom: 14 }} />
          <Skeleton width={180} height={14} style={{ marginBottom: 24 }} />
          <Skeleton height={200} radius={16} style={{ marginBottom: 24 }} />
          <div style={s.scenarioRow}>
            {[0, 1, 2, 3].map(i => <Skeleton key={i} height={90} radius={16} />)}
          </div>
        </>
      ) : (
        <>
          {recText && (
            <div style={{ ...s.recCard, background: recColor + '11', borderColor: recColor + '55' }}>
              <span style={{ ...s.recArrow, color: recColor }}>{cw > 5 ? '↑' : '✓'}</span>
              <span style={{ ...s.recText, color: recColor }}>{recText}</span>
            </div>
          )}

          {waitNow != null && (
            <div style={s.waitRow}>
              <span style={s.waitDot}>⏱</span>
              <span style={s.waitText}>
                {t.estimatedWait} :{' '}
                <strong style={s.waitNumber}>{Math.round(waitNow)} min</strong>
              </span>
            </div>
          )}

          {/* Chart */}
          <div className="section-header">
            <span className="section-title">{t.forecastChart}</span>
            <span style={s.sub}>{t.next60min}</span>
          </div>

          <div style={s.chartCard}>
            <ResponsiveContainer width="100%" height={200}>
              <ComposedChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
                <defs>
                  <linearGradient id="waitFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#db6d28" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#db6d28" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                <XAxis
                  dataKey="t"
                  tick={{ fill: '#8b949e', fontSize: 10 }}
                  tickLine={false} axisLine={false}
                  minTickGap={30}
                />
                <YAxis
                  domain={[0, 32]}
                  ticks={[0, 8, 16, 24, 32]}
                  tick={{ fill: '#8b949e', fontSize: 10 }}
                  tickLine={false} axisLine={false}
                  unit=" min"
                />
                <Tooltip content={<CustomTooltip t={t} />} />
                <Area type="natural" dataKey="wait" stroke="#db6d28" strokeWidth={2.5} fill="url(#waitFill)" dot={false} />
                <Line type="natural" dataKey="arrivals" stroke="#58a6ff" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={s.legendRow}>
              <span style={s.legendItem}>
                <span style={{ ...s.legendDot, background: '#db6d28' }} /> {t.waitLegend}
              </span>
              <span style={s.legendItem}>
                <span style={{ ...s.legendDot, background: '#58a6ff' }} /> {t.arrivalsLegend}
              </span>
            </div>
          </div>

          {/* Lanes — tap to open/close */}
          <div className="section-header">
            <span className="section-title">{t.laneScenarios}</span>
            <span style={s.sub}>{t.simulateScenarios}</span>
          </div>

          <div style={s.scenarioRow}>
            {scenarios.map(sc => {
              const isOpen = sc.lanes <= currentLanes;
              const color = SCENARIO_COLOR[sc.color] || '#8b949e';
              return (
                <button
                  key={sc.lanes}
                  onClick={() => setLanes(sc.lanes)}
                  style={{
                    ...s.scenBtn,
                    background: isOpen ? '#1a2e22' : '#161b22',
                    border: `1px solid ${isOpen ? '#3fb950' : '#30363d'}`,
                    color: isOpen ? '#e6edf3' : '#8b949e',
                  }}
                >
                  {isOpen && <span style={s.currentDot} />}
                  <span style={s.scenLabel}>{t.laneLabel(sc.lanes)}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
                    {Math.round(sc.est_wait_min)} min
                  </span>
                  <span style={{ ...s.currentBadge, color: isOpen ? '#3fb950' : '#484f58' }}>
                    {isOpen ? t.openBadge : t.closedBadge}
                  </span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

const s = {
  forecastTitleRow: { display: 'flex', alignItems: 'center', gap: 6 },
  clockIcon: { fontSize: 13 },
  headerRight: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 3 },
  sub: { fontSize: 11, color: '#8b95a8' },
  recCard: {
    display: 'flex', alignItems: 'center', gap: 14,
    padding: '18px 20px', border: '1px solid', borderRadius: 16, marginBottom: 14,
  },
  recArrow: { fontSize: 22, fontWeight: 700 },
  recText: { fontSize: 16, fontWeight: 600 },
  waitRow: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color: '#8b949e', fontSize: 13 },
  waitDot: { fontSize: 13 },
  waitText: { color: '#8b949e' },
  waitNumber: { fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums', color: '#e6edf3' },
  chartCard: {
    background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16,
    padding: '20px 18px 14px', marginBottom: 4,
  },
  legendRow: { display: 'flex', gap: 18, marginTop: 10, paddingLeft: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#8b949e' },
  legendDot: { display: 'inline-block', width: 8, height: 8, borderRadius: '50%' },
  scenarioRow: { display: 'flex', gap: 14 },
  scenBtn: {
    position: 'relative',
    flex: 1, padding: '14px 8px', borderRadius: 16,
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
    fontSize: 13, fontWeight: 700, transition: 'border-color 0.15s',
  },
  scenLabel: {},
  currentDot: {
    position: 'absolute', top: 8, right: 8, width: 7, height: 7,
    borderRadius: '50%', background: '#3fb950',
  },
  currentBadge: { fontSize: 9, fontWeight: 700, color: '#3fb950', letterSpacing: 0.5 },
  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
