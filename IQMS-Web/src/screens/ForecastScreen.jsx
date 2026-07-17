import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';
import { useToast } from '../context/ToastContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

const WAIT_THRESHOLD_MIN = 5;

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={s.tooltip}>
      <div style={{ fontSize: 11, color: 'var(--text-3)' }}>{label}</div>
      <div className="mono" style={{ color: 'var(--cyan)' }}>{payload[0]?.value?.toFixed(1)} min</div>
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

  const waitNow = forecastData?.wait_now_min ?? 0;
  const currentLanes = forecastData?.current_lanes ?? 1;
  const scenarios = forecastData?.lane_scenarios ?? [];
  const slots = chartData?.slots ?? [];

  const points = slots.map(sl => ({ t: sl.time, wait: sl.wait_min }));

  // WHY line, built only from real forecast-chart data — no fabricated deltas.
  const breachSlot = slots.find(sl => sl.wait_min > WAIT_THRESHOLD_MIN);
  const next15Arrivals = Math.round(slots.slice(0, 3).reduce((sum, sl) => sum + (sl.arrivals || 0), 0));
  const lanesLabel = t.filesCount(currentLanes);

  let title, why, actionNeeded;
  if (breachSlot && currentLanes < 4) {
    title = t.recActionTitle;
    why = t.whyBreach(breachSlot.time, next15Arrivals, lanesLabel);
    actionNeeded = true;
  } else if (breachSlot) {
    title = t.recHighDemandTitle;
    why = t.whyBreach(breachSlot.time, next15Arrivals, lanesLabel);
    actionNeeded = true;
  } else {
    title = t.recOptimalTitle;
    why = t.whyStable(lanesLabel);
    actionNeeded = false;
  }

  const recommended = scenarios.find(sc => sc.est_wait_min <= WAIT_THRESHOLD_MIN) ?? scenarios[scenarios.length - 1];

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
      <div className="section-header">
        <span className="section-title">{t.forecastTitle}</span>
        <UpdatedAgo lastUpdated={lastUpdated} />
      </div>

      {error && <div style={s.errorText}>{error}</div>}

      {loading ? (
        <>
          <Skeleton height={64} style={{ marginBottom: 24 }} />
          <Skeleton height={200} style={{ marginBottom: 24 }} />
          <div className="kpi-strip">{[0, 1, 2, 3].map(i => <Skeleton key={i} height={70} />)}</div>
        </>
      ) : (
        <>
          {/* Current wait — big readout */}
          <div style={s.readoutRow}>
            <span className="mono" style={s.readout}>{Math.round(waitNow)}</span>
            <span style={s.readoutUnit}>min · {t.estimatedWait.toLowerCase()}</span>
          </div>

          {/* Recommendation block */}
          <div className="hairline-top hairline-bottom" style={s.recBlock}>
            <span style={{ ...s.marker, background: actionNeeded ? 'var(--amber)' : 'var(--green)' }} />
            <div>
              <div style={s.recTitle}>{title}</div>
              <div style={s.recWhy}>{why}</div>
            </div>
          </div>

          {/* Chart */}
          <div className="section-header">
            <span className="section-title">{t.forecastChart}</span>
            <span style={s.sub}>{t.next60min}</span>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={points} margin={{ top: 8, right: 8, left: -16, bottom: 8 }}>
              <CartesianGrid stroke="var(--hairline)" strokeWidth={0.5} horizontal vertical={false} />
              <XAxis
                dataKey="t"
                tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'var(--font-num)' }}
                tickLine={false} axisLine={false}
                minTickGap={30}
              />
              <YAxis
                domain={[0, 32]}
                ticks={[0, 8, 16, 24, 32]}
                tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'var(--font-num)' }}
                tickLine={false} axisLine={false}
                unit=" min"
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={WAIT_THRESHOLD_MIN}
                stroke="var(--amber)"
                strokeDasharray="5 4"
                label={{ value: t.thresholdLabel, position: 'insideTopRight', fill: 'var(--amber)', fontSize: 11 }}
              />
              <Line type="monotone" dataKey="wait" stroke="var(--cyan)" strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} />
            </ComposedChart>
          </ResponsiveContainer>

          {/* Lane scenarios — tap to open/close */}
          <div className="section-header">
            <span className="section-title">{t.laneScenarios}</span>
            <span style={s.sub}>{t.simulateScenarios}</span>
          </div>

          <div className="kpi-strip">
            {scenarios.map(sc => {
              const isCurrent = sc.lanes === currentLanes;
              const isRecommended = !isCurrent && recommended && sc.lanes === recommended.lanes;
              return (
                <button
                  key={sc.lanes}
                  onClick={() => setLanes(sc.lanes)}
                  className="kpi-cell"
                  style={{
                    ...s.scenCell,
                    background: isRecommended ? 'var(--raised)' : 'transparent',
                    borderTop: isRecommended ? '2px solid var(--green)' : undefined,
                  }}
                >
                  <div className="micro-label">{t.laneLabel(sc.lanes)}</div>
                  <div className="mono" style={{ ...s.scenValue, color: isRecommended ? 'var(--green)' : 'var(--text)' }}>
                    {Math.round(sc.est_wait_min)} min
                  </div>
                  <div style={s.scenTag}>
                    {isCurrent ? t.currentTag : isRecommended ? t.recommendedTag : ' '}
                  </div>
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
  sub: { fontSize: 11, color: 'var(--text-3)' },
  readoutRow: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 16 },
  readout: { fontSize: 32, color: 'var(--text)' },
  readoutUnit: { fontSize: 13, color: 'var(--text-3)' },
  recBlock: {
    display: 'flex', alignItems: 'flex-start', gap: 12,
    background: 'var(--surface)', padding: '14px 4px', marginBottom: 8,
  },
  marker: { width: 8, height: 8, marginTop: 4, flexShrink: 0 },
  recTitle: { fontSize: 14, fontWeight: 500, color: 'var(--text)' },
  recWhy: { fontSize: 12, color: 'var(--text-2)', marginTop: 4, lineHeight: 1.5 },
  scenCell: { textAlign: 'left', cursor: 'pointer' },
  scenValue: { fontSize: 18, marginTop: 6 },
  scenTag: { fontSize: 11, color: 'var(--text-3)', marginTop: 4 },
  tooltip: {
    background: 'var(--surface)', border: '0.5px solid var(--hairline)', borderRadius: 4, padding: '6px 10px',
  },
  errorText: { color: 'var(--red)', fontSize: 13, padding: '8px 0' },
};
