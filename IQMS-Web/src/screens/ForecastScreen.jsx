import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label} min</div>
      <div style={{ color: '#58a6ff', fontWeight: 700 }}>{payload[0]?.value?.toFixed(1)} min</div>
    </div>
  );
}

export default function ForecastScreen() {
  const { t } = useLang();
  const [scenario, setScenario] = useState(null);

  const { data, loading, error } = useApi([
    `${API_URL}/forecast`,
    `${API_URL}/forecast-chart`,
  ]);
  const [forecastData, chartData] = data;

  const rec = forecastData?.recommendation;
  const currentWait = forecastData?.current_wait_min;
  const scenarios = forecastData?.scenarios ?? [];

  const points = (chartData?.points ?? []).map((p, i) => ({
    t: i * 5,
    wait: p.wait_min ?? p,
    upper: p.upper ?? null,
    lower: p.lower ?? null,
  }));

  const activeScenario = scenario != null
    ? scenarios.find(s => s.open_lanes === scenario)
    : null;

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

      {rec && (
        <div style={{
          ...s.recCard,
          background: rec.open_lanes > 0 ? '#1a2e22' : '#1c2128',
          borderColor: rec.open_lanes > 0 ? '#3fb950' : '#30363d',
        }}>
          <span style={{ ...s.recArrow, color: rec.open_lanes > 0 ? '#3fb950' : '#8b949e' }}>
            {rec.open_lanes > 0 ? '↑' : '✓'}
          </span>
          <span style={{ ...s.recText, color: rec.open_lanes > 0 ? '#3fb950' : '#e6edf3' }}>
            {rec.open_lanes > 0
              ? t.openMoreLanes(rec.open_lanes)
              : t.optimalConditions}
          </span>
        </div>
      )}

      {currentWait != null && (
        <div style={s.waitRow}>
          <span style={s.waitDot}>⏱</span>
          <span style={s.waitText}>
            {t.estimatedWait} :{' '}
            <strong style={{ color: '#e6edf3' }}>{Math.round(currentWait)} min</strong>
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
              label={{ value: 'MIN', position: 'insideBottom', offset: -2, fill: '#484f58', fontSize: 9 }}
            />
            <YAxis
              tick={{ fill: '#8b949e', fontSize: 10 }}
              tickLine={false} axisLine={false}
              unit=" min"
            />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="wait" stroke="#58a6ff" strokeWidth={2.5} dot={false} />
            {points[0]?.upper != null && (
              <Line type="monotone" dataKey="upper" stroke="#58a6ff" strokeWidth={1} strokeDasharray="4 3" dot={false} />
            )}
            {points[0]?.lower != null && (
              <Line type="monotone" dataKey="lower" stroke="#58a6ff" strokeWidth={1} strokeDasharray="4 3" dot={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
        <div style={s.legendRow}>
          <span style={s.legendItem}>
            <span style={{ ...s.legendLine, borderStyle: 'solid' }} /> {t.estimation}
          </span>
          <span style={s.legendItem}>
            <span style={{ ...s.legendLine, borderStyle: 'dashed' }} /> {t.confidenceInterval}
          </span>
        </div>
      </div>

      {/* Scenarios */}
      <div className="section-header">
        <span className="section-title">{t.laneScenarios}</span>
        <span style={s.sub}>{t.simulateScenarios}</span>
      </div>

      <div style={s.scenarioRow}>
        {[1, 2, 3].map(n => {
          const sc = scenarios.find(s => s.open_lanes === n);
          const active = scenario === n;
          const delta = sc?.wait_delta_min;
          return (
            <button
              key={n}
              onClick={() => setScenario(active ? null : n)}
              style={{
                ...s.scenBtn,
                background: active ? '#1c2a3a' : '#161b22',
                border: `1px solid ${active ? '#58a6ff' : '#30363d'}`,
                color: active ? '#58a6ff' : '#e6edf3',
              }}
            >
              <span style={s.scenLabel}>{n} FILE{n > 1 ? 'S' : ''}</span>
              {delta != null && (
                <span style={{ fontSize: 12, fontWeight: 700, color: delta < 0 ? '#3fb950' : '#f85149' }}>
                  {delta > 0 ? '+' : ''}{Math.round(delta)} min
                </span>
              )}
            </button>
          );
        })}
      </div>

      {activeScenario && (
        <div style={s.scenDetail}>
          {t.withLanes(activeScenario.open_lanes, Math.round(activeScenario.estimated_wait_min ?? 0))}
        </div>
      )}
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
    flex: 1, padding: '12px 6px', borderRadius: 10,
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5,
    fontSize: 13, fontWeight: 700, transition: 'border-color 0.15s',
  },
  scenLabel: {},
  scenDetail: {
    marginTop: 10, background: '#1c2128', border: '1px solid #30363d',
    borderRadius: 8, padding: '10px 14px', fontSize: 13, color: '#e6edf3',
  },
  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
