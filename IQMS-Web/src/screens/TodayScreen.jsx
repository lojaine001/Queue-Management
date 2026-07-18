import { useState } from 'react';
import {
  ComposedChart, Bar, Cell, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function StatCard({ value, label, sub, subColor, valueColor }) {
  return (
    <div style={s.statCard}>
      <div style={s.statLabel}>{label}</div>
      {value != null
        ? <div style={{ ...s.statValue, color: valueColor ?? '#e6edf3' }}>{value}</div>
        : <div style={s.statDash}>—</div>}
      {sub && <div style={{ ...s.statSub, color: subColor ?? s.statSub.color }}>{sub}</div>}
    </div>
  );
}

function EntriesTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700, fontSize: 12 }}>{p.value}</div>
      ))}
    </div>
  );
}

function WaitTooltip({ active, payload, label, t }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 700, fontSize: 12 }}>
          {p.dataKey === 'wait' ? t.waitLegend : t.arrivalsLegend}: {p.value?.toFixed(1)}
        </div>
      ))}
    </div>
  );
}

export default function TodayScreen() {
  const { t } = useLang();
  const [selectedDate, setSelectedDate] = useState(() => todayStr());

  const { data, loading, error, lastUpdated } = useApi([`${API_URL}/day-recap?date=${selectedDate}`]);
  const [recap] = data;

  const { data: waitData, loading: waitLoading } = useApi([`${API_URL}/forecast-chart`]);
  const [dayWait] = waitData;

  const totalCustomers = recap?.total_customers;
  const vsYesterdayPct = recap?.vs_yesterday_pct;
  const avgAge = recap?.avg_age;
  const genderRows = recap?.demographics_gender ?? [];
  const ageRows = recap?.demographics_age ?? [];
  const femme = genderRows.find(g => g.key === 'female');
  const homme = genderRows.find(g => g.key === 'male');
  const genderTotal = (femme?.count ?? 0) + (homme?.count ?? 0);

  const hourlyData = (recap?.entries_by_hour ?? []).map(h => ({
    hour: h.hour,
    count: h.count ?? 0,
    isPeak: !!h.is_peak,
  }));

  const waitPoints = (dayWait?.slots ?? []).map(sl => ({ t: sl.time, wait: sl.wait_min, arrivals: sl.arrivals }));

  return (
    <div className="screen-page">
      <div className="section-header" style={{ alignItems: 'center' }}>
        <span style={s.pageTitle}>{t.statsTitle}</span>
        <div style={s.headerRight}>
          <UpdatedAgo lastUpdated={lastUpdated} />
          <div style={s.datePickerWrap}>
            <span style={s.calIcon}>📅</span>
            <input
              type="date"
              value={selectedDate}
              max={todayStr()}
              onChange={e => setSelectedDate(e.target.value)}
              style={s.dateInput}
            />
          </div>
        </div>
      </div>

      {error && <div style={s.errorHint}>{error}</div>}

      {/* Stat row */}
      {loading ? (
        <div className="stat-grid-4">
          {[0, 1, 2, 3].map(i => <Skeleton key={i} height={90} radius={16} />)}
        </div>
      ) : (
        <div className="stat-grid-4">
          <StatCard
            label={t.totalClients}
            value={totalCustomers != null ? totalCustomers.toLocaleString() : null}
            valueColor="#e6edf3"
            sub={vsYesterdayPct != null ? t.vsYesterday(vsYesterdayPct) : undefined}
            subColor={vsYesterdayPct != null ? (vsYesterdayPct >= 0 ? '#3fb950' : '#f85149') : undefined}
          />
          <StatCard
            label={t.femmeLabel}
            value={femme ? `${femme.percent}%` : null}
            valueColor="#ec4899"
            sub={femme ? t.femmeCount(femme.count) : undefined}
          />
          <StatCard
            label={t.hommeLabel}
            value={homme ? `${homme.percent}%` : null}
            valueColor="#3b82f6"
            sub={homme ? t.hommeCount(homme.count) : undefined}
          />
          <StatCard
            label={t.ageMoyenLabel}
            value={avgAge != null ? t.ageValue(Math.round(avgAge)) : null}
            valueColor="#e6edf3"
          />
        </div>
      )}

      {/* Entrées par heure — unchanged */}
      <div className="section-header">
        <span className="section-title">{t.entriesByHour}</span>
      </div>
      <div style={s.chartCard}>
        {hourlyData.length > 0 ? (
          <ResponsiveContainer width="100%" height={190}>
            <ComposedChart data={hourlyData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis dataKey="hour" tick={{ fill: '#8b949e', fontSize: 9 }} tickLine={false} axisLine={false} interval={1} />
              <YAxis
                domain={[0, 'dataMax']} allowDecimals={false} tickCount={6}
                tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false}
              />
              <Tooltip content={<EntriesTooltip />} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]} name={t.entriesByHour}>
                {hourlyData.map((h, i) => <Cell key={i} fill={h.isPeak ? '#f85149' : '#3fb950'} />)}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div style={s.emptyChart}>{t.noHourlyData}</div>
        )}
        <div style={s.legendRow}>
          <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#3fb950' }} /> {t.today}</span>
          {hourlyData.some(h => h.isPeak) && (
            <span style={s.legendItem}><span style={{ ...s.legendSq, background: '#f85149' }} /> {t.peakHour}</span>
          )}
        </div>
      </div>

      {/* Temps d'attente — same live forecast chart as Prévision */}
      <div className="section-header">
        <span className="section-title">{t.waitChartTitle}</span>
        <span style={s.sub}>{t.next60min}</span>
      </div>
      <div style={s.chartCard}>
        {waitLoading ? (
          <Skeleton height={220} radius={16} />
        ) : waitPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={waitPoints} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <defs>
                <linearGradient id="statsWaitFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#db6d28" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#db6d28" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis dataKey="t" tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} minTickGap={30} />
              <YAxis domain={[0, 32]} ticks={[0, 8, 16, 24, 32]} tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} unit=" min" />
              <Tooltip content={<WaitTooltip t={t} />} />
              <Area type="natural" dataKey="wait" stroke="#db6d28" strokeWidth={2.5} fill="url(#statsWaitFill)" dot={false} />
              <Line type="natural" dataKey="arrivals" stroke="#58a6ff" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div style={s.emptyChart}>{t.noHourlyData}</div>
        )}
        <div style={s.legendRow}>
          <span style={s.legendItem}><span style={{ ...s.legendDot, background: '#db6d28' }} /> {t.waitLegend}</span>
          <span style={s.legendItem}><span style={{ ...s.legendDot, background: '#58a6ff' }} /> {t.arrivalsLegend}</span>
        </div>
      </div>

      {/* Customer demographics */}
      <div className="section-header">
        <span className="section-title">{t.demographicsTitle}</span>
        <span style={s.sub}>{t.demographicsSubtitle}</span>
      </div>

      {genderRows.length === 0 && ageRows.length === 0 ? (
        <div style={s.chartCard}>
          <div style={s.emptyChart}>{t.noDemographicsData}</div>
        </div>
      ) : (
        <div className="demo-grid">
          <div style={s.chartCard}>
            <div style={s.demoLabel}>{t.genderSplitLabel}</div>
            <div style={s.donutWrap}>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={[femme, homme].filter(Boolean)}
                    dataKey="count" nameKey="label"
                    innerRadius={60} outerRadius={90} paddingAngle={2} strokeWidth={0}
                  >
                    {[femme, homme].filter(Boolean).map((g, i) => <Cell key={i} fill={g.color} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div style={s.donutCenter}>
                <div style={s.donutCenterValue}>{genderTotal.toLocaleString()}</div>
                <div style={s.donutCenterLabel}>{t.totalLabel}</div>
              </div>
            </div>
            <div style={s.legendRow}>
              {femme && <span style={s.legendItem}><span style={{ ...s.legendDot, background: femme.color }} /> {t.femmeLabel} {femme.percent}%</span>}
              {homme && <span style={s.legendItem}><span style={{ ...s.legendDot, background: homme.color }} /> {t.hommeLabel} {homme.percent}%</span>}
            </div>
          </div>

          <div style={s.chartCard}>
            <div style={s.demoLabel}>{t.ageGroupLabel}</div>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={ageRows} margin={{ top: 24, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                <XAxis dataKey="group" tick={{ fill: '#8b949e', fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip content={<EntriesTooltip />} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {ageRows.map((a, i) => <Cell key={i} fill={a.color} />)}
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

const s = {
  pageTitle: { fontSize: 20, fontWeight: 500, color: '#e6edf3' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 16 },
  datePickerWrap: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    borderRadius: 10, padding: '6px 12px',
  },
  calIcon: { fontSize: 13 },
  dateInput: {
    background: 'transparent', border: 'none', color: '#e6edf3',
    fontSize: 13, fontFamily: 'inherit', colorScheme: 'dark',
  },

  statCard: {
    background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16, padding: '18px 16px',
  },
  statLabel: { fontSize: 9, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginBottom: 8, textTransform: 'uppercase' },
  statValue: { fontSize: 30, fontWeight: 700, lineHeight: 1.1, fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' },
  statDash: { fontSize: 20, color: '#484f58', marginTop: 4 },
  statSub: { fontSize: 11, color: '#3fb950', marginTop: 6 },

  sub: { fontSize: 11, color: '#484f58' },
  chartCard: { background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16, padding: '18px 16px 12px' },
  emptyChart: { textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: 13 },
  legendRow: { display: 'flex', gap: 16, marginTop: 8, paddingLeft: 12, flexWrap: 'wrap' },
  legendItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#8b949e' },
  legendSq: { display: 'inline-block', width: 10, height: 10, borderRadius: 2 },
  legendDot: { display: 'inline-block', width: 8, height: 8, borderRadius: '50%' },

  demoLabel: { fontSize: 13, color: '#8b949e', marginBottom: 8 },
  donutWrap: { position: 'relative' },
  donutCenter: {
    position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
    textAlign: 'center', pointerEvents: 'none',
  },
  donutCenterValue: { fontSize: 26, fontWeight: 700, color: '#e6edf3', fontFamily: 'var(--font-mono)' },
  donutCenterLabel: { fontSize: 11, color: '#8b949e', marginTop: 2 },

  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
