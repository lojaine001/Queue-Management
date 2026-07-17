import { useEffect, useState } from 'react';
import { LanguageProvider, useLang } from './context/LanguageContext';
import { ToastProvider } from './context/ToastContext';
import LiveScreen from './screens/LiveScreen';
import ForecastScreen from './screens/ForecastScreen';
import TodayScreen from './screens/TodayScreen';
import AlertsScreen from './screens/AlertsScreen';

const TABS = ['live', 'forecast', 'today', 'alerts'];

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function Shell() {
  const [tab, setTab] = useState('live');
  const { t, lang, setLang } = useLang();
  const now = useClock();

  const screen = {
    live:     <LiveScreen />,
    forecast: <ForecastScreen />,
    today:    <TodayScreen />,
    alerts:   <AlertsScreen />,
  }[tab];

  const clock = now.toLocaleTimeString(lang === 'fr' ? 'fr-FR' : 'en-GB', { hour12: false });

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <div style={s.brandRow}>
            <span style={s.logo}>IQMS</span>
            <span style={s.storeName}>{t.storeName}</span>
          </div>
          <div style={s.headerRight}>
            <div style={s.liveIndicator}>
              <span style={s.liveDot} />
              <span style={s.liveLabel}>{t.liveLabel}</span>
            </div>
            <span className="mono" style={s.clock}>{clock}</span>
            <button style={s.langToggle} onClick={() => setLang(lang === 'fr' ? 'en' : 'fr')}>
              {lang === 'fr' ? 'EN' : 'FR'}
            </button>
          </div>
        </div>
      </header>

      <nav className="app-tabs">
        <div className="app-tabs-inner">
          {TABS.map(id => (
            <button
              key={id}
              onClick={() => setTab(id)}
              style={{
                ...s.tabBtn,
                color: tab === id ? 'var(--text)' : 'var(--text-3)',
                borderBottomColor: tab === id ? 'var(--green)' : 'transparent',
              }}
            >
              {t.tabs[id]}
            </button>
          ))}
        </div>
      </nav>

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
  brandRow: { display: 'flex', alignItems: 'baseline', gap: 10 },
  logo: { fontSize: 16, fontWeight: 500, color: 'var(--text)', letterSpacing: 0.3 },
  storeName: { fontSize: 12, color: 'var(--text-3)' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 20 },
  liveIndicator: { display: 'flex', alignItems: 'center', gap: 6 },
  liveDot: {
    width: 7, height: 7, borderRadius: '50%',
    background: 'var(--green)',
    animation: 'pulse 2s ease-in-out infinite',
  },
  liveLabel: {
    fontSize: 11, fontWeight: 500, letterSpacing: '0.1em',
    color: 'var(--green)', textTransform: 'uppercase',
  },
  clock: { fontSize: 13, color: 'var(--text-2)' },
  tabBtn: {
    padding: '10px 0',
    fontSize: 12, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase',
    borderBottom: '2px solid transparent',
    transition: 'color 150ms ease, border-color 150ms ease',
  },
  langToggle: {
    fontSize: 12, fontWeight: 500, color: 'var(--text-2)', letterSpacing: '0.05em',
  },
};
