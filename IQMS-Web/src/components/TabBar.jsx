import { useLang } from '../context/LanguageContext';

const TABS = ['live', 'forecast', 'today', 'alerts'];

export default function TabBar({ active, onChange }) {
  const { t } = useLang();
  return (
    <div className="mobile-tabbar" style={s.bar}>
      {TABS.map(id => (
        <button key={id} onClick={() => onChange(id)} style={s.btn}>
          <span style={{ ...s.label, color: active === id ? '#3fb950' : '#8b949e' }}>
            {t.tabs[id]}
          </span>
          {active === id && <div style={s.indicator} />}
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
  label: { fontSize: 13, fontWeight: 500 },
  indicator: { height: 2, width: '60%', background: '#3fb950', borderRadius: 1 },
};
