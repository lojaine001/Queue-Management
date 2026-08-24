import { createContext, useContext, useState } from 'react';

const T = {
  fr: {
    tabs: { live: 'En direct', today: 'Statistique' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    avgWait: 'ATTENTE',
    alertsLabel: 'Alertes', seuilLabel: 'Seuil',
    horizonLabel: 'Horizon', horizonLive: 'Actuel',
    horizonForecastAt: min => `Attente prévue à +${min} min`,
    horizonMaxHint: max => `prévision disponible jusqu'à +${max} min`,
    horizonCustomPlaceholder: 'min',
    selectCameraHint: 'Sélectionnez une caméra pour voir son flux',
    alertPopupMessage: (current, threshold) => `Alerte : attente moyenne à ${current} min (seuil : ${threshold} min)`,
    liveCameras: 'LIVE CAMERAS', cameraPending: 'Flux caméra en attente',
    loading: 'Chargement…', serverError: 'Impossible de joindre le serveur.',
    justNow: "À l'instant", secondsAgo: s => `il y a ${s}s`,
    next60min: 'Prochaines 60 minutes',
    waitLegend: 'Attente',
    totalClients: 'CLIENTS TOTAL', peakHour: 'HEURE DE POINTE',
    entriesByHour: 'ENTRÉES PAR HEURE',
    today: "Aujourd'hui", yesterday: 'Hier',
    noHourlyData: 'Aucune donnée horaire disponible',
    vsYesterday: pct => `${pct > 0 ? '▲' : '▼'} ${Math.abs(pct)}% vs hier`,
    statsTitle: 'Statistique',
    femmeLabel: 'FEMME', hommeLabel: 'HOMME',
    waitChartTitle: "TEMPS D'ATTENTE", dayWaitHistory: 'Historique de la journée',
    demographicsTitle: 'DÉMOGRAPHIE CLIENTS', demographicsSubtitle: 'Profil genre et âge des visiteurs',
    genderSplitLabel: 'Répartition par genre', ageGroupLabel: 'Répartition par âge',
    noDemographicsData: 'Aucune donnée démographique pour cette date',
    last7days: '7 derniers jours',
  },
  en: {
    tabs: { live: 'Live', today: 'Statistics' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    avgWait: 'WAIT',
    alertsLabel: 'Alerts', seuilLabel: 'Threshold',
    horizonLabel: 'Horizon', horizonLive: 'Current',
    horizonForecastAt: min => `Predicted wait at +${min} min`,
    horizonMaxHint: max => `forecast available up to +${max} min`,
    horizonCustomPlaceholder: 'min',
    selectCameraHint: 'Select a camera to view its feed',
    alertPopupMessage: (current, threshold) => `Alert: average wait at ${current} min (threshold: ${threshold} min)`,
    liveCameras: 'LIVE CAMERAS', cameraPending: 'Camera feed pending',
    loading: 'Loading…', serverError: 'Cannot reach server.',
    justNow: 'Just now', secondsAgo: s => `${s}s ago`,
    next60min: 'Next 60 minutes',
    waitLegend: 'Wait',
    totalClients: 'TOTAL CLIENTS', peakHour: 'PEAK HOUR',
    entriesByHour: 'ENTRIES BY HOUR',
    today: 'Today', yesterday: 'Yesterday',
    noHourlyData: 'No hourly data available',
    vsYesterday: pct => `${pct > 0 ? '▲' : '▼'} ${Math.abs(pct)}% vs yesterday`,
    statsTitle: 'Statistics',
    femmeLabel: 'FEMALE', hommeLabel: 'MALE',
    waitChartTitle: 'WAIT TIME', dayWaitHistory: 'Full-day history',
    demographicsTitle: 'CUSTOMER DEMOGRAPHICS', demographicsSubtitle: 'Gender and age profile of visitors',
    genderSplitLabel: 'Gender split', ageGroupLabel: 'Age group distribution',
    noDemographicsData: 'No demographic data for this date',
    last7days: 'Last 7 days',
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
