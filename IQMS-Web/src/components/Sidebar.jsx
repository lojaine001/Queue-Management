import { useLang } from '../context/LanguageContext';

const TABS = ['live', 'forecast', 'today', 'alerts'];
const ICONS = { live: '◉', forecast: '◈', today: '◧', alerts: '◬' };

export default function Sidebar({ active, onChange }) {
  const { t, lang, setLang } = useLang();
  return (
    <aside className="app-sidebar">
      {/* Logo */}
      <div style={s.logoWrap}>
        <span style={s.logo}>IQMS</span>
      </div>

      {/* Nav */}
      <nav style={s.nav}>
        {TABS.map(id => (
          <button
            key={id}
            onClick={() => onChange(id)}
            style={{
              ...s.navBtn,
              background: active === id ? '#1c2128' : 'none',
              color: active === id ? '#3fb950' : '#8b949e',
              borderLeft: active === id ? '2px solid #3fb950' : '2px solid transparent',
            }}
          >
            <span style={s.navIcon}>{ICONS[id]}</span>
            {t.tabs[id]}
          </button>
        ))}
      </nav>

      {/* Language toggle at bottom */}
      <div style={s.langWrap}>
        <button
          onClick={() => setLang(lang === 'fr' ? 'en' : 'fr')}
          style={s.langBtn}
        >
          <span style={{ color: lang === 'fr' ? '#3fb950' : '#484f58' }}>FR</span>
          <span style={s.langSep}>/</span>
          <span style={{ color: lang === 'en' ? '#3fb950' : '#484f58' }}>EN</span>
        </button>
      </div>
    </aside>
  );
}

const s = {
  logoWrap: {
    padding: '0 20px 24px',
    borderBottom: '1px solid #30363d',
  },
  logo: {
    fontSize: 22,
    fontWeight: 800,
    color: '#e6edf3',
    letterSpacing: 0.5,
  },
  nav: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    paddingTop: 16,
  },
  navBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '11px 20px',
    border: 'none',
    fontSize: 14,
    fontWeight: 500,
    textAlign: 'left',
    transition: 'background 0.1s',
    width: '100%',
  },
  navIcon: { fontSize: 16, width: 18 },
  langWrap: {
    borderTop: '1px solid #30363d',
    padding: '16px 20px',
  },
  langBtn: {
    background: '#1c2128',
    border: '1px solid #30363d',
    borderRadius: 6,
    padding: '6px 14px',
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 1,
    display: 'flex',
    gap: 4,
    alignItems: 'center',
  },
  langSep: { color: '#30363d' },
};
