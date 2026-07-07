import {
  ComposedChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';

function StatCard({ value, label, sub, valueColor, children }) {
  return (
    <div style={s.statCard}>
      <div style={s.statLabel}>{label}</div>
      {value != null
        ? <div style={{ ...s.statValue, color: valueColor ?? '#e6edf3' }}>{value}</div>
        : <div style={s.statDash}>—</div>}
      {sub && <div style={s.statSub}>{sub}</div>}
      {children}
    </div>
  );
}

function SummaryRow({ label, value, valueColor }) {
  return (
    <div style={s.summaryRow}>
      <span style={s.summaryLabel}>{label}</span>
      <span style={{ fontWeight: 700, color: valueColor ?? '#3fb950' }}>{value ?? '—'}</span>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700, fontSize: 12 }}>
          {p.value}
        </div>
      ))}
    </div>
  );
}

export default function TodayScreen() {
  const { t } = useLang();
  const { data, loading, error } = useApi([`${API_URL}/day-recap`]);
  const [recap] = data;

  const totalCustomers = recap?.total_customers;
  const peakHour = recap?.peak_hour;
  const peakCount = recap?.peak_count;
  const avgWait = recap?.avg_wait_min;
  const lanesToday = recap?.lanes_today;
  const busiestLane = recap?.busiest_lane;
  const alertMinutes = recap?.alert_minutes;
  const equipment = recap?.equipment ?? [];
  const paniers = equipment.find(e => e.type === 'store_basket')?.count;
  const chariots = equipment.find(e => e.type === 'trolley')?.count;

  const hourlyData = (recap?.entries_by_hour ?? []).map(h => ({
    hour: h.hour,
    count: h.count ?? 0,
    isPeak: !!h.is_peak,
  }));

  return (
    <div className="screen-page">
      {loading && <div style={s.hint}>{t.loading}</div>}
      {error && <div style={s.errorHint}>{error}</div>}

      {/* Stat cards */}
      <div className="stat-grid">
        <StatCard
          label={t.totalClients}
          value={totalCustomers != null ? totalCustomers.toLocaleString() : null}
          valueColor="#e6edf3"
        />
        <StatCard
          label={t.peakHour}
          value={peakHour}
          valueColor="#58a6ff"
          sub={peakCount != null ? `${peakCount} ${t.clients}` : undefined}
        />
        <div style={{ ...s.statCard, gridColumn: 1 }}>
          <div style={s.statLabel}>{t.equipment}</div>
          {paniers != null && (
            <div style={s.equipRow}>
              <span style={s.equipIcon}>🧺</span>
              <div>
                <div style={s.equipCount}>{paniers}</div>
                <div style={s.equipName}>{t.baskets}</div>
              </div>
            </div>
          )}
          {chariots != null && (
            <div style={s.equipRow}>
              <span style={s.equipIcon}>🛒</span>
              <div>
                <div style={s.equipCount}>{chariots}</div>
                <div style={s.equipName}>{t.carts}</div>
              </div>
            </div>
          )}
          {paniers == null && chariots == null && (
            <div style={s.statDash}>{t.noData}</div>
          )}
        </div>
      </div>

      {/* Chart + summary — side by side on desktop */}
      <div className="today-bottom">
        {/* Hourly chart */}
        <div>
          <div className="section-header">
            <span className="section-title">{t.entriesByHour}</span>
          </div>
          <div style={s.chartCard}>
            {hourlyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={190}>
                <ComposedChart data={hourlyData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                  <XAxis dataKey="hour" tick={{ fill: '#8b949e', fontSize: 9 }} tickLine={false} axisLine={false} interval={1} />
                  <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="count" radius={[3, 3, 0, 0]} name={t.entriesByHour}>
                    {hourlyData.map((h, i) => (
                      <Cell key={i} fill={h.isPeak ? '#f85149' : '#3fb950'} />
                    ))}
                  </Bar>
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: 13 }}>
                {t.noHourlyData}
              </div>
            )}
            <div style={s.legendRow}>
              <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#3fb950' }} /> {t.today}</span>
              {hourlyData.some(h => h.isPeak) && (
                <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#f85149' }} /> {t.peakHour}</span>
              )}
            </div>
          </div>
        </div>

        {/* Daily summary */}
        <div>
          <div className="section-header">
            <span className="section-title">{t.dailySummary}</span>
          </div>
          <div style={s.summaryCard}>
            <SummaryRow
              label={t.avgWaitTime}
              value={avgWait != null ? `${Math.round(avgWait)} min` : null}
              valueColor="#3fb950"
            />
            <SummaryRow
              label={t.lanesUsed}
              value={lanesToday != null
                ? `${lanesToday}${busiestLane ? ` (${t.busiestLane(busiestLane)})` : ''}`
                : null}
              valueColor="#58a6ff"
            />
            <SummaryRow
              label={t.alertTime}
              value={alertMinutes != null ? `${alertMinutes} min` : null}
              valueColor={alertMinutes > 0 ? '#f85149' : '#3fb950'}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

const s = {
  statCard: {
    background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: '14px 12px',
  },
  statLabel: { fontSize: 9, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginBottom: 6, textTransform: 'uppercase' },
  statValue: { fontSize: 28, fontWeight: 700, lineHeight: 1.1 },
  statDash: { fontSize: 20, color: '#484f58', marginTop: 4 },
  statSub: { fontSize: 11, color: '#3fb950', marginTop: 5 },
  equipRow: { display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 },
  equipIcon: { fontSize: 18 },
  equipCount: { fontSize: 18, fontWeight: 700, color: '#e6edf3', lineHeight: 1 },
  equipName: { fontSize: 10, color: '#8b949e', marginTop: 1 },
  chartCard: { background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: '14px 10px 8px' },
  legendRow: { display: 'flex', gap: 16, marginTop: 6, paddingLeft: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#8b949e' },
  legendSq: { display: 'inline-block', width: 10, height: 10, borderRadius: 2 },
  summaryCard: {
    background: '#161b22', border: '1px solid #30363d', borderRadius: 10, overflow: 'hidden',
  },
  summaryRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '13px 16px', borderBottom: '1px solid #30363d', fontSize: 13,
  },
  summaryLabel: { color: '#e6edf3' },
  hint: { color: '#8b949e', fontSize: 13, padding: '8px 0' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
