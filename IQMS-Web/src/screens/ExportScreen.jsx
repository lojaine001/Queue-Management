import { useEffect, useState } from 'react';
import { API_URL } from '../config';
import { useLang } from '../context/LanguageContext';

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

const KPI_KEYS = ['customers', 'avg_wait', 'max_wait', 'avg_waiting', 'peak_waiting',
                   'lanes_used', 'alert_minutes', 'trolley', 'store_basket', 'peak_hour'];

// Matches api.py's KPI_LABELS exactly (including the ASCII-safe "utilisees"
// spelling, chosen there to avoid an encoding risk when the backend file
// was deployed -- kept consistent here rather than diverging).
const KPI_LABELS = {
  fr: {
    customers: 'Clients total', avg_wait: "Temps d'attente moyen (min)",
    max_wait: "Temps d'attente max (min)", avg_waiting: 'Personnes en attente (moy.)',
    peak_waiting: 'Pic de personnes en attente', lanes_used: 'Files utilisees',
    alert_minutes: 'Temps en alerte (min)', trolley: 'Chariots', store_basket: 'Paniers',
    peak_hour: 'Heure de pointe',
  },
  en: {
    customers: 'Total customers', avg_wait: 'Avg checkout wait (min)',
    max_wait: 'Max checkout wait (min)', avg_waiting: 'Avg people waiting',
    peak_waiting: 'Peak people waiting', lanes_used: 'Lanes used',
    alert_minutes: 'Time in alert (min)', trolley: 'Trolleys', store_basket: 'Baskets',
    peak_hour: 'Peak hour',
  },
};

function fmtPeriodLabel(period, dateStr, lang) {
  const d = new Date(dateStr + 'T00:00:00');
  const locale = lang === 'fr' ? 'fr-FR' : 'en-GB';
  if (period === 'day') {
    return d.toLocaleDateString(locale, { day: '2-digit', month: '2-digit', year: 'numeric' });
  }
  if (period === 'week') {
    const dow = (d.getDay() + 6) % 7; // 0 = Monday
    const monday = new Date(d);
    monday.setDate(d.getDate() - dow);
    const sunday = new Date(monday);
    sunday.setDate(monday.getDate() + 6);
    const fmt = x => x.toLocaleDateString(locale, { day: '2-digit', month: '2-digit' });
    return lang === 'fr' ? `Semaine du ${fmt(monday)} au ${fmt(sunday)}` : `Week of ${fmt(monday)} to ${fmt(sunday)}`;
  }
  return d.toLocaleDateString(locale, { month: 'long', year: 'numeric' });
}

function fmtCell(v) {
  if (v == null) return '—';
  return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : v;
}

export default function ExportScreen() {
  const { t, lang } = useLang();
  const [period, setPeriod] = useState('day');
  const [date, setDate] = useState(() => todayStr());
  const [selectedKpis, setSelectedKpis] = useState(() => new Set(KPI_KEYS));
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  const allSelected = selectedKpis.size === KPI_KEYS.length;
  const noneSelected = selectedKpis.size === 0;
  const kpiParam = KPI_KEYS.filter(k => selectedKpis.has(k)).join(',');
  const labels = KPI_LABELS[lang] || KPI_LABELS.fr;

  const toggleKpi = key => {
    setSelectedKpis(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };
  const toggleAll = () => setSelectedKpis(allSelected ? new Set() : new Set(KPI_KEYS));

  useEffect(() => {
    if (noneSelected) {
      setPreview(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_URL}/export/preview?period=${period}&date=${date}&kpis=${kpiParam}&lang=${lang}`)
      .then(r => { if (!r.ok) throw new Error('bad response'); return r.json(); })
      .then(data => { if (!cancelled) setPreview(data); })
      .catch(() => { if (!cancelled) setError(t.exportError); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [period, date, lang, kpiParam, noneSelected]);

  const handleExport = async () => {
    if (noneSelected) return;
    setExporting(true);
    try {
      const resp = await fetch(`${API_URL}/export?period=${period}&date=${date}&kpis=${kpiParam}&format=csv&lang=${lang}`);
      if (!resp.ok) throw new Error('export failed');
      const blob = await resp.blob();
      const disposition = resp.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="([^"]+)"/);
      const filename = match ? match[1] : `IQMS_${period}_${date}.csv`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError(t.exportError);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="screen-page">
      <div className="section-header" style={{ alignItems: 'center' }}>
        <span style={s.pageTitle}>{t.tabs.export}</span>
      </div>

      {error && <div style={s.errorHint}>{error}</div>}

      {/* Period */}
      <div className="section-header">
        <span className="section-title">{t.periodLabel}</span>
      </div>
      <div style={s.segmentRow}>
        {['day', 'week', 'month'].map(p => (
          <button
            key={p}
            onClick={() => setPeriod(p)}
            style={{ ...s.segmentBtn, ...(period === p ? s.segmentBtnActive : {}) }}
          >
            {t.periods[p]}
          </button>
        ))}
      </div>

      {/* Date */}
      <div style={s.dateRow}>
        <input
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
          style={s.dateInput}
        />
        <span style={s.periodPreview}>{fmtPeriodLabel(period, date, lang)}</span>
      </div>

      {/* KPIs */}
      <div className="section-header">
        <span className="section-title">{t.kpisLabel}</span>
        <button onClick={toggleAll} style={s.selectAllBtn}>
          {allSelected ? t.deselectAll : t.selectAll}
        </button>
      </div>
      <div style={s.kpiGrid}>
        {KPI_KEYS.map(k => (
          <label key={k} style={s.kpiItem}>
            <input
              type="checkbox"
              checked={selectedKpis.has(k)}
              onChange={() => toggleKpi(k)}
              style={s.checkbox}
            />
            <span>{labels[k]}</span>
          </label>
        ))}
      </div>

      {/* Preview */}
      <div className="section-header">
        <span className="section-title">{t.previewLabel}</span>
        {preview && <span style={s.sub}>{preview.row_count} {period === 'day' ? t.hoursSuffix : t.daysSuffix}</span>}
      </div>
      <div style={s.chartCard}>
        {noneSelected ? (
          <div style={s.emptyChart}>{t.selectAtLeastOne}</div>
        ) : loading ? (
          <div style={s.emptyChart}>{t.loading}</div>
        ) : preview ? (
          <div style={s.tableWrap}>
            <table style={s.table}>
              <thead>
                <tr>
                  <th style={s.th}>{period === 'day' ? t.hourCol : t.dateCol}</th>
                  {KPI_KEYS.filter(k => selectedKpis.has(k)).map(k => (
                    <th key={k} style={s.th}>{labels[k]}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map(r => (
                  <tr key={r.label}>
                    <td style={s.td}>{r.label}</td>
                    {KPI_KEYS.filter(k => selectedKpis.has(k)).map(k => (
                      <td key={k} style={s.td}>{fmtCell(r[k])}</td>
                    ))}
                  </tr>
                ))}
                <tr>
                  <td style={s.tdTotal}>{preview.total.label}</td>
                  {KPI_KEYS.filter(k => selectedKpis.has(k)).map(k => (
                    <td key={k} style={s.tdTotal}>{fmtCell(preview.total[k])}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      {/* Export button */}
      <button
        onClick={handleExport}
        disabled={noneSelected || exporting}
        style={{ ...s.exportBtn, ...(noneSelected || exporting ? s.exportBtnDisabled : {}) }}
      >
        {exporting ? t.exporting : t.exportButton}
      </button>
    </div>
  );
}

const s = {
  pageTitle: { fontSize: 20, fontWeight: 500, color: '#e6edf3' },
  sub: { fontSize: 11, color: '#484f58' },
  errorHint: { color: '#f85149', fontSize: 13, padding: '8px 0' },

  segmentRow: {
    display: 'flex', gap: 8, marginBottom: 16,
  },
  segmentBtn: {
    flex: 1, padding: '10px 0', borderRadius: 10,
    background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    color: '#8b949e', fontSize: 13, fontWeight: 600,
  },
  segmentBtnActive: {
    background: 'rgba(88, 166, 255, 0.12)', border: '1px solid #58a6ff', color: '#58a6ff',
  },

  dateRow: {
    display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20,
    background: 'var(--card-bg)', border: '1px solid var(--card-border)',
    borderRadius: 12, padding: '10px 16px',
  },
  dateInput: {
    background: 'transparent', border: 'none', color: '#e6edf3',
    fontSize: 13, fontFamily: 'inherit', colorScheme: 'dark',
  },
  periodPreview: { fontSize: 12, color: '#8b949e' },

  selectAllBtn: {
    background: 'transparent', border: '1px solid #30363d', borderRadius: 999,
    padding: '4px 12px', color: '#58a6ff', fontSize: 11, fontWeight: 600,
  },
  kpiGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8,
    marginBottom: 8,
  },
  kpiItem: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 10,
    padding: '10px 12px', fontSize: 13, color: '#e6edf3',
  },
  checkbox: { accentColor: '#58a6ff', width: 15, height: 15 },

  chartCard: { background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: 16, padding: '12px', marginBottom: 16 },
  emptyChart: { textAlign: 'center', padding: '40px 0', color: '#8b949e', fontSize: 13 },
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
  th: {
    textAlign: 'left', padding: '8px 10px', color: '#8b949e', fontWeight: 600,
    borderBottom: '1px solid #30363d', whiteSpace: 'nowrap',
  },
  td: {
    padding: '7px 10px', color: '#c9d1d9', borderBottom: '1px solid #21262d', whiteSpace: 'nowrap',
  },
  tdTotal: {
    padding: '7px 10px', color: '#e6edf3', fontWeight: 700, whiteSpace: 'nowrap', background: 'rgba(63,185,80,0.08)',
  },

  exportBtn: {
    width: '100%', padding: '14px 0', borderRadius: 12, border: 'none',
    background: '#3fb950', color: '#0d1117', fontSize: 14, fontWeight: 700,
  },
  exportBtnDisabled: {
    background: '#21262d', color: '#484f58',
  },
};
