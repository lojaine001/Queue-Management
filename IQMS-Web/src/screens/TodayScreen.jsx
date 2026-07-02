import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Line, ComposedChart,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';

function StatBox({ value, label, sub, valueColor }) {
  return (
    <div style={s.statBox}>
      <div style={{ ...s.statValue, color: valueColor ?? '#e6edf3' }}>{value ?? '—'}</div>
      <div style={s.statLabel}>{label}</div>
      {sub && <div style={s.statSub}>{sub}</div>}
    </div>
  );
}

function SummaryRow({ label, value, valueColor }) {
  return (
    <div style={s.summaryRow}>
      <span style={s.summaryLabel}>{label}</span>
      <span style={{ ...s.summaryValue, color: valueColor ?? '#3fb950' }}>{value ?? '—'}</span>
    </div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700, fontSize: 12 }}>
          {p.value} entrées
        </div>
      ))}
    </div>
  );
}

export default function TodayScreen() {
  const { data, loading, error } = useApi([`${API_URL}/day-recap`]);
  const [recap] = data;

  const totalCustomers = recap?.total_entries ?? recap?.total_customers;
  const peakHour = recap?.peak_hour;
  const avgWait = recap?.avg_wait_min;
  const maxWait = recap?.max_wait_min;
  const avgLanes = recap?.avg_open_lanes;
  const paniers = recap?.equipment?.store_basket ?? recap?.equipment?.basket;
  const chariots = recap?.equipment?.trolley;

  const hourlyData = (recap?.hourly ?? []).map(h => ({
    hour: h.hour != null ? `${String(h.hour).padStart(2, '0')}h` : h.label,
    today: h.count ?? h.today ?? 0,
    yesterday: h.yesterday ?? null,
  }));

  return (
    <div style={s.page}>
      {loading && <div style={s.hint}>Chargement…</div>}
      {error && <div style={s.errorHint}>{error}</div>}

      {/* Top stat grid */}
      <div style={s.statGrid}>
        <StatBox
          value={totalCustomers != null ? String(totalCustomers) : null}
          label="CLIENTS TOTAL"
          sub={recap?.vs_yesterday != null ? `${recap.vs_yesterday > 0 ? '▲' : '▼'} ${Math.abs(recap.vs_yesterday)}% vs hier` : undefined}
          valueColor="#e6edf3"
        />
        <StatBox
          value={peakHour}
          label="HEURE DE POINTE"
          sub={recap?.peak_hour_count != null ? `${recap.peak_hour_count} clients` : undefined}
          valueColor="#58a6ff"
        />
        <div style={s.equipBox}>
          <div style={s.equipTitle}>ÉQUIPEMENT</div>
          {paniers != null && (
            <div style={s.equipRow}>
              <span style={s.equipIcon}>🧺</span>
              <div>
                <div style={s.equipCount}>{paniers}</div>
                <div style={s.equipLabel}>Paniers</div>
              </div>
            </div>
          )}
          {chariots != null && (
            <div style={s.equipRow}>
              <span style={s.equipIcon}>🛒</span>
              <div>
                <div style={s.equipCount}>{chariots}</div>
                <div style={s.equipLabel}>Chariots</div>
              </div>
            </div>
          )}
          {paniers == null && chariots == null && (
            <div style={s.hint}>Aucune donnée</div>
          )}
        </div>
      </div>

      {/* Hourly bar chart */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>ENTRÉES PAR HEURE</span>
      </div>
      <div style={s.chartWrap}>
        {hourlyData.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={hourlyData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis dataKey="hour" tick={{ fill: '#8b949e', fontSize: 9 }} tickLine={false} axisLine={false} interval={1} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="today" fill="#3fb950" radius={[2, 2, 0, 0]} name="Aujourd'hui" />
              {hourlyData[0]?.yesterday != null && (
                <Line type="monotone" dataKey="yesterday" stroke="#484f58" strokeDasharray="4 3" dot={false} name="Hier" />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ ...s.hint, textAlign: 'center', padding: 40 }}>Aucune donnée horaire disponible</div>
        )}
        <div style={s.legendRow}>
          <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#3fb950' }} /> Aujourd'hui</span>
          {hourlyData[0]?.yesterday != null && (
            <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#484f58' }} /> Hier</span>
          )}
        </div>
      </div>

      {/* Daily summary */}
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>RÉSUMÉ JOURNALIER</span>
      </div>
      <div style={s.summaryCard}>
        <SummaryRow
          label="Temps d'attente moyen"
          value={avgWait != null ? `${Math.round(avgWait)} min` : null}
          valueColor="#3fb950"
        />
        <SummaryRow
          label="Temps d'attente max"
          value={maxWait != null ? `${Math.round(maxWait)} min` : null}
          valueColor={maxWait > 15 ? '#f85149' : '#d29922'}
        />
        <SummaryRow
          label="Files ouvertes (moyenne)"
          value={avgLanes != null ? avgLanes.toFixed(1) : null}
          valueColor="#58a6ff"
        />
      </div>
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
  statGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: 8,
    marginTop: 8,
  },
  statBox: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '12px 10px',
  },
  statValue: { fontSize: 22, fontWeight: 700, lineHeight: 1.1 },
  statLabel: { fontSize: 9, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginTop: 4, textTransform: 'uppercase' },
  statSub: { fontSize: 11, color: '#3fb950', marginTop: 4 },
  equipBox: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '10px 10px',
  },
  equipTitle: { fontSize: 9, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginBottom: 8, textTransform: 'uppercase' },
  equipRow: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 },
  equipIcon: { fontSize: 16 },
  equipCount: { fontSize: 15, fontWeight: 700, color: '#e6edf3', lineHeight: 1 },
  equipLabel: { fontSize: 10, color: '#8b949e' },
  chartWrap: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    padding: '12px 8px 8px',
  },
  legendRow: { display: 'flex', gap: 16, marginTop: 6, paddingLeft: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#8b949e' },
  legendSq: { display: 'inline-block', width: 10, height: 10, borderRadius: 2 },
  summaryCard: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    overflow: 'hidden',
  },
  summaryRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 14px',
    borderBottom: '1px solid #30363d',
  },
  summaryLabel: { fontSize: 13, color: '#e6edf3' },
  summaryValue: { fontSize: 13, fontWeight: 700 },
  hint: { color: '#8b949e', fontSize: 13 },
  errorHint: { color: '#f85149', fontSize: 13 },
};
