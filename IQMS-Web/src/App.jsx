import { useState } from 'react';
import { LanguageProvider, useLang } from './context/LanguageContext';
import { ToastProvider } from './context/ToastContext';
import Sidebar from './components/Sidebar';
import TabBar from './components/TabBar';
import LiveScreen from './screens/LiveScreen';
import TodayScreen from './screens/TodayScreen';

function Shell() {
  const [tab, setTab] = useState('live');
  const { t, lang, setLang } = useLang();

  const screen = {
    live:  <LiveScreen />,
    today: <TodayScreen />,
  }[tab];

  return (
    <div className="app-shell">
      {/* Desktop sidebar */}
      <Sidebar active={tab} onChange={setTab} />

      {/* Mobile header + tab bar: one sticky unit, no hardcoded offsets between them */}
      <div className="mobile-nav">
        <header className="mobile-header">
          <button style={s.iconBtn}>☰</button>
          <span style={s.logo}>IQMS</span>
          <button
            style={s.langToggle}
            onClick={() => setLang(lang === 'fr' ? 'en' : 'fr')}
          >
            {lang === 'fr' ? 'EN' : 'FR'}
          </button>
        </header>
        <TabBar active={tab} onChange={setTab} />
      </div>

      <div className="app-main">
        {screen}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <LanguageProvider>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </LanguageProvider>
  );
}

const s = {
  logo: {
    fontSize: 19,
    fontWeight: 800,
    color: '#e6edf3',
    letterSpacing: 0.5,
  },
  iconBtn: { background: 'none', border: 'none', color: '#8b949e', fontSize: 18, padding: 4 },
  langToggle: {
    background: '#1c2128',
    border: '1px solid #30363d',
    borderRadius: 6,
    padding: '4px 10px',
    color: '#3fb950',
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 1,
  },
};
