const TABS = [
  { id: 'live',     label: 'En direct' },
  { id: 'forecast', label: 'Prévision' },
  { id: 'today',    label: "Aujourd'hui" },
  { id: 'alerts',   label: 'Alertes' },
];

export default function TabBar({ active, onChange }) {
  return (
    <div style={s.bar}>
      {TABS.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)} style={s.btn}>
          <span style={{ ...s.label, color: active === t.id ? '#3fb950' : '#8b949e' }}>
            {t.label}
          </span>
          {active === t.id && <div style={s.indicator} />}
        </button>
      ))}
    </div>
  );
}

const s = {
  bar: {
    display: 'flex',
    borderBottom: '1px solid #30363d',
    background: '#0d1117',
    position: 'sticky',
    top: 52,
    zIndex: 10,
  },
  btn: {
    flex: 1,
    background: 'none',
    border: 'none',
    padding: '12px 4px 0',
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 10,
  },
  label: {
    fontSize: 13,
    fontWeight: 500,
    letterSpacing: 0.2,
  },
  indicator: {
    height: 2,
    width: '60%',
    background: '#3fb950',
    borderRadius: 1,
  },
};
