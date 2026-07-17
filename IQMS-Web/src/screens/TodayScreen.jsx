import {
  ComposedChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

function KpiCell({ label, value, sub, color }) {
  return (
    <div className="kpi-cell">
      <div className="micro-label">{label}</div>
      <div className="mono" style={{ fontSize: 24, marginTop: 6, color: color ?? 'var(--text)' }}>
        {value ?? '—'}
      </div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div style={s.summaryRow}>
      <span style={{ fontSize: 13, color: 'var(--text-2)' }}>{label}</span>
      <span className="mono" style={{ fontSize: 13, color: 'var(--text)' }}>{value ?? '—'}</span>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={s.tooltip}>
      <div style={{ fontSize: 11, color: 'var(--text-3)' }}>{label}</div>
      <div className="mono" style={{ color: 'var(--text)' }}>{payload[0]?.value}</div>
    </div>
  );
}

export default function TodayScreen() {
  const { t } = useLang();
  const { data, loading, error, lastUpdated } = useApi([`${API_URL}/day-recap`]);
  const [recap] = data;

  const totalCustomers = recap?.total_customers;
  const vsLastWeekPct = recap?.vs_last_week_pct;
  const peakHour = recap?.peak_hour;
  const peakCount = recap?.peak_count;
  const peakPctOfTotal = recap?.peak_pct_of_total;
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
      <div className="section-header">
        <span className="section-title">{t.todayTitle}</span>
        <UpdatedAgo lastUpdated={lastUpdated} />
      </div>

      {error && <div style={s.errorText}>{error}</div>}

      {loading ? (
        <>
          <div className="kpi-strip">{[0, 1, 2, 3].map(i => <Skeleton key={i} height={70} />)}</div>
          <div className="today-bottom" style={{ marginTop: 24 }}>
            <Skeleton height={220} />
            <Skeleton height={150} />
          </div>
        </>
      ) : (
        <>
          <div className="kpi-strip">
            <KpiCell
              label={t.totalClients}
              value={totalCustomers != null ? totalCustomers.toLocaleString() : null}
              sub={vsLastWeekPct != null ? t.vsLastWeek(vsLastWeekPct) : undefined}
            />
            <KpiCell
              label={t.peakHour}
              value={peakHour}
              color="var(--cyan)"
              sub={peakCount != null
                ? `${peakCount} ${t.clients}${peakPctOfTotal != null ? ` · ${t.pctOfTotal(peakPctOfTotal)}` : ''}`
                : undefined}
            />
            <KpiCell
              label={t.avgWaitTime}
              value={avgWait != null ? `${Math.round(avgWait)} min` : null}
              color="var(--cyan)"
            />
            <KpiCell
              label={t.lanesUsed}
              value={lanesToday}
              sub={busiestLane ? t.busiestLane(busiestLane) : undefined}
            />
          </div>

          <div className="today-bottom">
            {/* Hourly chart */}
            <div>
              <div className="section-header">
                <span className="section-title">{t.entriesByHour}</span>
              </div>
              {hourlyData.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <ComposedChart data={hourlyData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid stroke="var(--hairline)" strokeWidth={0.5} horizontal vertical={false} />
                    <XAxis
                      dataKey="hour"
                      tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'var(--font-num)' }}
                      tickLine={false} axisLine={false} interval={1}
                    />
                    <YAxis
                      domain={[0, 'dataMax']}
                      allowDecimals={false}
                      tickCount={6}
                      tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'var(--font-num)' }}
                      tickLine={false} axisLine={false}
                    />
                    <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--raised)' }} />
                    <Bar dataKey="count" radius={0} maxBarSize={18}>
                      {hourlyData.map((h, i) => (
                        <Cell key={i} fill={h.isPeak ? 'var(--cyan)' : 'var(--text)'} fillOpacity={h.isPeak ? 1 : 0.85} />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-3)', fontSize: 13 }}>
                  {t.noHourlyData}
                </div>
              )}
            </div>

            {/* Ruled summary */}
            <div>
              <div className="section-header">
                <span className="section-title">{t.dailySummary}</span>
              </div>
              <div>
                <SummaryRow label={t.baskets} value={paniers} />
                <SummaryRow label={t.carts} value={chariots} />
                <SummaryRow label={t.alertTime} value={alertMinutes != null ? `${alertMinutes} min` : null} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

const s = {
  summaryRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 0', borderBottom: '1px solid var(--hairline)',
  },
  tooltip: {
    background: 'var(--surface)', border: '0.5px solid var(--hairline)', borderRadius: 4, padding: '6px 10px',
  },
  errorText: { color: 'var(--red)', fontSize: 13, padding: '8px 0' },
};
