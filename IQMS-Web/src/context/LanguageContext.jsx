import { createContext, useContext, useState } from 'react';

const T = {
  fr: {
    tabs: { live: 'En direct', forecast: 'Prévision', today: "Aujourd'hui", alerts: 'Alertes' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    inQueue: 'EN FILE', avgWait: 'ATTENTE MOY.',
    openLanes: n => `${n} file${n !== 1 ? 's' : ''} ouverte${n !== 1 ? 's' : ''}`,
    liveCameras: 'LIVE CAMERAS',
    loading: 'Chargement…', serverError: 'Impossible de joindre le serveur.',
    forecast15min: 'PRÉVISION 15 MIN', recommendation: 'Recommandation',
    openMoreLanes: n => `Ouvrir ${n} file${n !== 1 ? 's' : ''} de plus`,
    optimalConditions: 'Files optimales — aucune action requise',
    recOptimal: 'Conditions de file optimales',
    recOpenMore: (n, saved) => `Ouvrir ${n} file${n > 1 ? 's' : ''} de plus — économise ~${saved} min`,
    recHighDemand: 'Forte demande — toutes les files disponibles recommandées',
    recLight: n => `File légère — ${n} file${n > 1 ? 's' : ''} suffit pour maintenir sous 5 min`,
    estimatedWait: "Temps d'attente estimé",
    forecastChart: "PRÉVISION TEMPS D'ATTENTE", next60min: 'Prochaines 60 minutes',
    estimation: 'Estimation', confidenceInterval: 'Intervalle de confiance',
    waitLegend: 'Attente', arrivalsLegend: 'Arrivées',
    laneScenarios: 'FILES',
    simulateScenarios: "Toucher pour ouvrir/fermer une file",
    laneLabel: n => `File ${n}`, openBadge: 'OUVERTE', closedBadge: 'FERMÉE',
    withLanes: (n, w) => `Avec ${n} file${n !== 1 ? 's' : ''} : attente estimée ${w} min`,
    totalClients: 'CLIENTS TOTAL', peakHour: 'HEURE DE POINTE', equipment: 'ÉQUIPEMENT',
    baskets: 'Paniers', carts: 'Chariots', noData: 'Aucune donnée',
    entriesByHour: 'ENTRÉES PAR HEURE',
    today: "Aujourd'hui", yesterday: 'Hier',
    noHourlyData: 'Aucune donnée horaire disponible',
    dailySummary: 'RÉSUMÉ JOURNALIER',
    avgWaitTime: "Temps d'attente moyen",
    lanesUsed: 'Files utilisées', busiestLane: id => `la + chargée : ${id}`,
    alertTime: 'Temps en alerte',
    alerts: 'ALERTES', active: n => `${n} active${n !== 1 ? 's' : ''}`,
    critical: 'CRITIQUE', urgent: 'URGENT', warning: 'ATTENTION',
    noAlerts: 'Aucune alerte active', allGood: 'Tout est sous contrôle',
    vsYesterday: pct => `${pct > 0 ? '▲' : '▼'} ${Math.abs(pct)}% vs hier`,
  },
  en: {
    tabs: { live: 'Live', forecast: 'Forecast', today: 'Today', alerts: 'Alerts' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    inQueue: 'IN QUEUE', avgWait: 'AVG WAIT',
    openLanes: n => `${n} open lane${n !== 1 ? 's' : ''}`,
    liveCameras: 'LIVE CAMERAS',
    loading: 'Loading…', serverError: 'Cannot reach server.',
    forecast15min: '15-MIN FORECAST', recommendation: 'Recommendation',
    openMoreLanes: n => `Open ${n} more lane${n !== 1 ? 's' : ''}`,
    optimalConditions: 'Queue conditions are optimal',
    recOptimal: 'Queue conditions are optimal',
    recOpenMore: (n, saved) => `Open ${n} more lane${n > 1 ? 's' : ''} — saves ~${saved} min wait`,
    recHighDemand: 'High demand — all available lanes recommended',
    recLight: n => `Queue is light — ${n} lane${n > 1 ? 's' : ''} would still keep wait under 5 min`,
    estimatedWait: 'Estimated wait time',
    forecastChart: 'WAIT TIME FORECAST', next60min: 'Next 60 minutes',
    estimation: 'Estimation', confidenceInterval: 'Confidence interval',
    waitLegend: 'Wait', arrivalsLegend: 'Arrivals',
    laneScenarios: 'LANES',
    simulateScenarios: 'Tap to open/close a lane',
    laneLabel: n => `Lane ${n}`, openBadge: 'OPEN', closedBadge: 'CLOSED',
    withLanes: (n, w) => `With ${n} lane${n !== 1 ? 's' : ''}: estimated wait ${w} min`,
    totalClients: 'TOTAL CLIENTS', peakHour: 'PEAK HOUR', equipment: 'EQUIPMENT',
    baskets: 'Baskets', carts: 'Carts', noData: 'No data',
    entriesByHour: 'ENTRIES BY HOUR',
    today: 'Today', yesterday: 'Yesterday',
    noHourlyData: 'No hourly data available',
    dailySummary: 'DAILY SUMMARY',
    avgWaitTime: 'Average wait time',
    lanesUsed: 'Lanes used', busiestLane: id => `busiest: ${id}`,
    alertTime: 'Alert time',
    alerts: 'ALERTS', active: n => `${n} active`,
    critical: 'CRITICAL', urgent: 'URGENT', warning: 'WARNING',
    noAlerts: 'No active alerts', allGood: 'Everything is under control',
    vsYesterday: pct => `${pct > 0 ? '▲' : '▼'} ${Math.abs(pct)}% vs yesterday`,
  },
};

const Ctx = createContext(null);

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState('fr');
  return (
    <Ctx.Provider value={{ t: T[lang], lang, setLang }}>
      {children}
    </Ctx.Provider>
  );
}

export const useLang = () => useContext(Ctx);
