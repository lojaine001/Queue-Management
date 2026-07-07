import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL, HEADERS } from '../config';
import { useLang } from '../context/LanguageContext';

const SCENARIO_COLOR = { red: '#f85149', orange: '#db6d28', yellow: '#d29922', green: '#3fb950' };

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      <div style={{ color: '#58a6ff', fontWeight: 700 }}>{payload[0]?.value?.toFixed(1)} min</div>
    </div>
  );
}

export default function ForecastScreen() {
  const { t } = useLang();
  const [selected, setSelected] = useState(null);

  const { data, loading, error, refresh } = useApi([
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
    setSelected(n);
    try {
      await fetch(`${API_URL}/set-lanes`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ lanes: n }),
      });
      refresh();
    } catch {
      // fall through — next refresh cycle will show the real state either way
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
        <span style={s.sub}>{t.recommendation}</span>
      </div>

      {loading && <div style={s.hint}>{t.loading}</div>}
      {error && <div style={s.errorHint}>{error}</div>}

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
            <strong style={{ color: '#e6edf3' }}>{Math.round(waitNow)} min</strong>
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
          <LineChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fill: '#8b949e', fontSize: 10 }}
              tickLine={false} axisLine={false}
              minTickGap={30}
            />
            <YAxis
              tick={{ fill: '#8b949e', fontSize: 10 }}
              tickLine={false} axisLine={false}
              unit=" min"
            />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="wait" stroke="#58a6ff" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div style={s.legendRow}>
          <span style={s.legendItem}>
            <span style={{ ...s.legendLine, borderStyle: 'solid' }} /> {t.estimation}
          </span>
        </div>
      </div>

      {/* Scenarios */}
      <div className="section-header">
        <span className="section-title">{t.laneScenarios}</span>
        <span style={s.sub}>{t.simulateScenarios}</span>
      </div>

      <div style={s.scenarioRow}>
        {scenarios.map(sc => {
          const isCurrent = sc.lanes === currentLanes;
          const isSelected = selected === sc.lanes;
          const color = SCENARIO_COLOR[sc.color] || '#8b949e';
          return (
            <button
              key={sc.lanes}
              onClick={() => setLanes(sc.lanes)}
              style={{
                ...s.scenBtn,
                background: isSelected ? '#1c2a3a' : '#161b22',
                border: `1px solid ${isSelected ? '#58a6ff' : '#30363d'}`,
                color: isSelected ? '#58a6ff' : '#e6edf3',
              }}
            >
              {isCurrent && <span style={s.currentDot} />}
              <span style={s.scenLabel}>{t.laneLabel(sc.lanes)}</span>
              <span style={{ fontSize: 12, fontWeight: 700, color }}>
                {Math.round(sc.est_wait_min)} min
              </span>
              {isCurrent && <span style={s.currentBadge}>{t.openBadge}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

const s = {
  forecastTitleRow: { display: 'flex', alignItems: 'center', gap: 6 },
  clockIcon: { fontSize: 13 },
  sub: { fontSize: 11, color: '#484f58' },
  recCard: {
    display: 'flex', alignItems: 'center', gap: 14,
    padding: '16px 18px', border: '1px solid', borderRadius: 10, marginBottom: 10,
  },
  recArrow: { fontSize: 22, fontWeight: 700 },
  recText: { fontSize: 16, fontWeight: 600 },
  waitRow: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, color: '#8b949e', fontSize: 13 },
  waitDot: { fontSize: 13 },
  waitText: { color: '#8b949e' },
  chartCard: {
    background: '#161b22', border: '1px solid #30363d', borderRadius: 10,
    padding: '14px 10px 8px', marginBottom: 4,
  },
  legendRow: { display: 'flex', gap: 18, marginTop: 8, paddingLeft: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#8b949e' },
  legendLine: { display: 'inline-block', width: 18, height: 0, border: '1.5px solid #58a6ff' },
  scenarioRow: { display: 'flex', gap: 10 },
  scenBtn: {
    position: 'relative',
    flex: 1, padding: '12px 6px', borderRadius: 10,
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
    fontSize: 13, fontWeight: 700, transition: 'border-color 0.15s',
  },
  scenLabel: {},
  currentDot: {
    position: 'absolute', top: 6, right: 6, width: 7, height: 7,
    borderRadius: '50%', background: '#3fb950',
  },
  currentBadge: { fontSize: 9, fontWeight: 700, color: '#3fb950', letterSpacing: 0.5 },
  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
