import { useState } from 'react';
import TabBar from './components/TabBar';
import LiveScreen from './screens/LiveScreen';
import ForecastScreen from './screens/ForecastScreen';
import TodayScreen from './screens/TodayScreen';
import AlertsScreen from './screens/AlertsScreen';

export default function App() {
  const [tab, setTab] = useState('live');

  return (
    <div style={s.root}>
      {/* Top header */}
      <header style={s.header}>
        <button style={s.menuBtn}>☰</button>
        <span style={s.logo}>IQMS</span>
        <button style={s.bellBtn}>🔔</button>
      </header>

      {/* Tab bar */}
      <TabBar active={tab} onChange={setTab} />

      {/* Screen content */}
      <main style={s.main}>
        {tab === 'live'     && <LiveScreen />}
        {tab === 'forecast' && <ForecastScreen />}
        {tab === 'today'    && <TodayScreen />}
        {tab === 'alerts'   && <AlertsScreen />}
      </main>
    </div>
  );
}

const s = {
  root: {
    maxWidth: 480,
    margin: '0 auto',
    minHeight: '100vh',
    background: '#0d1117',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 16px',
    background: '#0d1117',
    borderBottom: '1px solid #30363d',
    position: 'sticky',
    top: 0,
    zIndex: 20,
  },
  logo: {
    fontSize: 18,
    fontWeight: 700,
    color: '#e6edf3',
    letterSpacing: 2,
  },
  menuBtn: {
    background: 'none',
    border: 'none',
    color: '#8b949e',
    fontSize: 18,
    padding: 4,
  },
  bellBtn: {
    background: 'none',
    border: 'none',
    fontSize: 18,
    padding: 4,
  },
  main: {
    flex: 1,
    overflowY: 'auto',
  },
};
