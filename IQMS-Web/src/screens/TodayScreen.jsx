import { useState } from 'react';
import {
  ComposedChart, Bar, Cell, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, LineChart,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';
import UpdatedAgo from '../components/UpdatedAgo';
import Skeleton from '../components/Skeleton';

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function GenderTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: '#1c2128', border: '1px solid #30363d', borderRadius: 6, padding: '6px 10px' }}>
      <span style={{ color: d.color, fontWeight: 700, fontSize: 12 }}>{d.label} - {d.percent}%</span>
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
  const isToday = selectedDate === todayStr();

  const { data, loading, error, lastUpdated } = useApi([`${API_URL}/day-recap?date=${selectedDate}`]);
  const [recap] = data;

  const waitUrl = isToday ? `${API_URL}/forecast-chart` : `${API_URL}/day-wait-chart?date=${selectedDate}`;
  const { data: waitData, loading: waitLoading } = useApi([waitUrl]);
  const [dayWait] = waitData;

  const totalCustomers = recap?.total_customers;
  const vsYesterdayPct = recap?.vs_yesterday_pct;
  const trend7d = recap?.trend_7d ?? [];
  const genderRows = recap?.demographics_gender ?? [];
  const ageRows = recap?.demographics_age ?? [];
  const femme = genderRows.find(g => g.key === 'female');
  const homme = genderRows.find(g => g.key === 'male');

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

      {/* Clients Total banner */}
      {loading ? (
        <Skeleton height={96} radius={16} />
      ) : (
        <div style={s.banner}>
          <div style={s.bannerLeft}>
            <div style={s.bannerIcon}>👥</div>
            <div>
              <div style={s.statLabel}>{t.totalClients}</div>
              {totalCustomers != null ? (
                <div style={s.bannerValueRow}>
                  <span style={s.bannerValue}>{totalCustomers.toLocaleString()}</span>
                  {vsYesterdayPct != null && (
                    <span style={{ ...s.bannerDelta, color: vsYesterdayPct >= 0 ? '#3fb950' : '#f85149' }}>
                      {t.vsYesterday(vsYesterdayPct)}
                    </span>
                  )}
                </div>
              ) : <div style={s.statDash}>—</div>}
            </div>
          </div>
          {trend7d.length > 0 && (
            <div style={s.bannerRight}>
              <span style={s.sub}>{t.last7days}</span>
              <ResponsiveContainer width={140} height={40}>
                <LineChart data={trend7d}>
                  <Line type="monotone" dataKey="count" stroke="#3fb950" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
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

      {/* Temps d'attente — live 60-min forecast for today, full-day history for past dates */}
      <div className="section-header">
        <span className="section-title">{t.waitChartTitle}</span>
        <span style={s.sub}>{isToday ? t.next60min : t.dayWaitHistory}</span>
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
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Tooltip content={<GenderTooltip />} />
                <Pie
                  data={[femme, homme].filter(Boolean)}
                  dataKey="count" nameKey="label"
                  innerRadius={60} outerRadius={90} paddingAngle={2} strokeWidth={0}
                >
                  {[femme, homme].filter(Boolean).map((g, i) => <Cell key={i} fill={g.color} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div style={s.legendRow}>
              {femme && <span style={s.legendItem}><span style={{ ...s.legendDot, background: femme.color }} /> {t.femmeLabel}</span>}
              {homme && <span style={s.legendItem}><span style={{ ...s.legendDot, background: homme.color }} /> {t.hommeLabel}</span>}
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

  banner: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16,
    padding: '20px 24px', gap: 16,
  },
  bannerLeft: { display: 'flex', alignItems: 'center', gap: 16 },
  bannerIcon: {
    width: 48, height: 48, borderRadius: 12, background: 'rgba(63,185,80,0.12)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, flexShrink: 0,
  },
  bannerValueRow: { display: 'flex', alignItems: 'baseline', gap: 10, marginTop: 4 },
  bannerValue: { fontSize: 32, fontWeight: 700, fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums', color: '#e6edf3' },
  bannerDelta: { fontSize: 12, fontWeight: 600 },
  bannerRight: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 },

  statLabel: { fontSize: 9, fontWeight: 700, color: '#8b949e', letterSpacing: 0.8, marginBottom: 8, textTransform: 'uppercase' },
  statDash: { fontSize: 20, color: '#484f58', marginTop: 4 },

  sub: { fontSize: 11, color: '#484f58' },
  chartCard: { background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16, padding: '18px 16px 12px' },
  emptyChart: { textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: 13 },
  legendRow: { display: 'flex', gap: 16, marginTop: 8, paddingLeft: 12, flexWrap: 'wrap' },
  legendItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#8b949e' },
  legendSq: { display: 'inline-block', width: 10, height: 10, borderRadius: 2 },
  legendDot: { display: 'inline-block', width: 8, height: 8, borderRadius: '50%' },

  demoLabel: { fontSize: 13, color: '#8b949e', marginBottom: 8 },

  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },
};
