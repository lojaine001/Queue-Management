import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';

const SCENARIO_BTN = [
  { lanes: 1, label: '1 FILE' },
  { lanes: 2, label: '2 FILES' },
  { lanes: 3, label: '3 FILES' },
];

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
  const [scenario, setScenario] = useState(null);

  const { data, loading, error } = useApi([
    `${API_URL}/forecast`,
    `${API_URL}/forecast-chart`,
  ]);
  const [forecastData, chartData] = data;

  const rec = forecastData?.recommendation;
  const currentWait = forecastData?.current_wait_min;
  const trend = forecastData?.trend;

  const points = (chartData?.points ?? []).map((p, i) => ({
    t: i * 5,
    wait: p.wait_min ?? p,
    lower: p.lower ?? null,
    upper: p.upper ?? null,
  }));

  const scenarios = forecastData?.scenarios ?? [];
  const activeScenario = scenario != null
    ? scenarios.find(s => s.open_lanes === scenario)
    : null;

  return (
    <div style={s.page}>
      {/* 15-min recommendation */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>PRÉVISION 15 MIN</span>
        <span style={s.sub}>Recommandation</span>
      </div>

      {loading && <div style={s.hint}>Chargement…</div>}
      {error && <div style={s.errorHint}>{error}</div>}

      {rec && (
        <div style={{ ...s.recCard, background: rec.open_lanes > 0 ? '#1a2e22' : '#21262d', borderColor: rec.open_lanes > 0 ? '#3fb950' : '#30363d' }}>
          <span style={s.recArrow}>{rec.open_lanes > 0 ? '↑' : '✓'}</span>
          <span style={s.recText}>
            {rec.open_lanes > 0
              ? `Ouvrir ${rec.open_lanes} file${rec.open_lanes > 1 ? 's' : ''} de plus`
              : 'Files optimales — aucune action requise'}
          </span>
        </div>
      )}

      {currentWait != null && (
        <div style={s.waitRow}>
          <span style={s.waitIcon}>⏱</span>
          <span style={s.waitText}>
            Temps d'attente estimé : <strong>{Math.round(currentWait)} min</strong>
          </span>
        </div>
      )}

      {/* Chart */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>PRÉVISION TEMPS D'ATTENTE</span>
        <span style={s.sub}>Prochaines 60 minutes</span>
      </div>

      <div style={s.chartWrap}>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fill: '#8b949e', fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              label={{ value: 'MINUTES', position: 'insideBottom', offset: -2, fill: '#484f58', fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: '#8b949e', fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              unit=" min"
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="wait"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={false}
              name="Estimation"
            />
            {points[0]?.upper != null && (
              <Line
                type="monotone"
                dataKey="upper"
                stroke="#58a6ff"
                strokeWidth={1}
                strokeDasharray="4 3"
                dot={false}
                name="Intervalle haut"
              />
            )}
            {points[0]?.lower != null && (
              <Line
                type="monotone"
                dataKey="lower"
                stroke="#58a6ff"
                strokeWidth={1}
                strokeDasharray="4 3"
                dot={false}
                name="Intervalle bas"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
        <div style={s.legendRow}>
          <span style={s.legendItem}><span style={{ ...s.legendLine, borderStyle: 'solid' }} /> Estimation</span>
          <span style={s.legendItem}><span style={{ ...s.legendLine, borderStyle: 'dashed' }} /> Intervalle de confiance</span>
        </div>
      </div>

      {/* Scenarios */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>SCÉNARIOS FILES</span>
        <span style={s.sub}>Simuler l'ouverture de files supplémentaires</span>
      </div>
      <div style={s.scenarioRow}>
        {SCENARIO_BTN.map(b => {
          const sc = scenarios.find(s => s.open_lanes === b.lanes);
          const active = scenario === b.lanes;
          const delta = sc?.wait_delta_min;
          return (
            <button
              key={b.lanes}
              onClick={() => setScenario(active ? null : b.lanes)}
              style={{
                ...s.scenBtn,
                background: active ? '#1c2a3a' : '#161b22',
                border: `1px solid ${active ? '#58a6ff' : '#30363d'}`,
                color: active ? '#58a6ff' : '#e6edf3',
              }}
            >
              <span style={s.scenLabel}>{b.label}</span>
              {delta != null && (
                <span style={{ ...s.scenDelta, color: delta < 0 ? '#3fb950' : '#f85149' }}>
                  {delta > 0 ? '+' : ''}{Math.round(delta)} min
                </span>
              )}
            </button>
          );
        })}
      </div>

      {activeScenario && (
        <div style={s.scenDetail}>
          <span style={s.scenDetailText}>
            Avec {activeScenario.open_lanes} files : attente estimée{' '}
            <strong style={{ color: '#3fb950' }}>
              {Math.round(activeScenario.estimated_wait_min ?? 0)} min
            </strong>
          </span>
        </div>
      )}
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
  sub: { fontSize: 11, color: '#484f58' },
  recCard: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '14px 16px',
    border: '1px solid',
    borderRadius: 8,
    marginBottom: 10,
  },
  recArrow: { fontSize: 20, color: '#3fb950' },
  recText: { fontSize: 15, fontWeight: 600, color: '#3fb950' },
  waitRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginBottom: 4,
    color: '#8b949e',
    fontSize: 13,
  },
  waitIcon: { fontSize: 13 },
  waitText: { color: '#8b949e' },
  chartWrap: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '12px 8px 8px',
    marginBottom: 4,
  },
  legendRow: { display: 'flex', gap: 16, marginTop: 6, paddingLeft: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#8b949e' },
  legendLine: {
    display: 'inline-block',
    width: 18,
    height: 0,
    border: '1px solid #58a6ff',
  },
  scenarioRow: { display: 'flex', gap: 8 },
  scenBtn: {
    flex: 1,
    padding: '10px 6px',
    borderRadius: 8,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 4,
    fontWeight: 600,
    fontSize: 13,
    transition: 'border-color 0.15s',
  },
  scenLabel: {},
  scenDelta: { fontSize: 11, fontWeight: 700 },
  scenDetail: {
    marginTop: 10,
    background: '#1c2128',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '10px 14px',
  },
  scenDetailText: { fontSize: 13, color: '#e6edf3' },
  hint: { color: '#8b949e', fontSize: 13, padding: '10px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '10px 0' },
};
