import { createContext, useContext, useState } from 'react';

const T = {
  fr: {
    tabs: { live: 'En direct', today: 'Statistique', export: 'Export' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    avgWait: 'ATTENTE',
    alertsLabel: 'Alertes', seuilLabel: 'Seuil',
    enableNotifications: 'Activer les notifications',
    horizonLabel: 'Horizon', horizonLive: 'Actuel',
    horizonForecastAt: min => `Attente prévue à +${min} min`,
    horizonSourceNote: 'Basée sur les caisses détectées automatiquement par la caméra — peut différer du tableau de bord si un nombre de caisses est forcé manuellement là-bas.',
    forecastUpdatedAt: hhmm => `Mis à jour ${hhmm}`, forecastStale: 'Prévision obsolète',
    avgCheckoutWait: "Temps d'attente moyen (caisse)", maxCheckoutWait: "Temps d'attente max",
    imageDelayed: "Image en retard", checkoutCameraUnavailable: "Caméra caisse indisponible",
    avgPeopleWaiting: "Nb moyen de personnes en attente en caisse", peakPeopleWaiting: "Pic de personnes en attente",
    peopleUnit: "personnes", noCheckoutQueueData: "Aucune donnée de file en caisse pour cette journée",
    periodLabel: "Période", periods: { day: "Jour", week: "Semaine", month: "Mois" },
    kpisLabel: "KPIs", selectAll: "Tout sélectionner", deselectAll: "Tout désélectionner",
    previewLabel: "Aperçu", hoursSuffix: "heures", daysSuffix: "jours",
    selectAtLeastOne: "Sélectionnez au moins un KPI",
    hourCol: "Heure", dateCol: "Date",
    exportButton: "Exporter", exporting: "Export en cours...", exportError: "Erreur lors du chargement",
    selectCameraHint: 'Sélectionnez une caméra pour voir son flux',
    // Wording adapts to whichever horizon is selected — same value the
    // gauge itself is showing, so the alert and the display never disagree.
    alertPopupMessage: (value, threshold, horizonMin) => horizonMin === 0
      ? `Alerte : attente actuelle à ${value} min (seuil : ${threshold} min)`
      : `Prévision : attente de ${value} min dans ${horizonMin} min (seuil : ${threshold} min)`,
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
    waitChartTitle: "TEMPS D'ATTENTE", dayWaitHistory: 'Historique de la journée',
    last7days: '7 derniers jours',
  },
  en: {
    tabs: { live: 'Live', today: 'Statistics', export: 'Export' },
    live: 'LIVE', open: 'OPEN', busy: 'BUSY', closed: 'CLOSED', clients: 'clients',
    liveQueueStatus: 'LIVE QUEUE STATUS',
    avgWait: 'WAIT',
    alertsLabel: 'Alerts', seuilLabel: 'Threshold',
    enableNotifications: 'Enable notifications',
    horizonLabel: 'Horizon', horizonLive: 'Current',
    horizonForecastAt: min => `Predicted wait at +${min} min`,
    horizonSourceNote: 'Based on lanes auto-detected by the camera — may differ from the dashboard if a lane count is manually overridden there.',
    forecastUpdatedAt: hhmm => `Updated ${hhmm}`, forecastStale: 'Forecast stale',
    avgCheckoutWait: "Average checkout wait", maxCheckoutWait: "Max checkout wait",
    imageDelayed: "Image delayed", checkoutCameraUnavailable: "Checkout camera unavailable",
    avgPeopleWaiting: "Average people waiting at checkout", peakPeopleWaiting: "Peak people waiting",
    peopleUnit: "people", noCheckoutQueueData: "No checkout queue data for this day",
    periodLabel: "Period", periods: { day: "Day", week: "Week", month: "Month" },
    kpisLabel: "KPIs", selectAll: "Select all", deselectAll: "Deselect all",
    previewLabel: "Preview", hoursSuffix: "hours", daysSuffix: "days",
    selectAtLeastOne: "Select at least one KPI",
    hourCol: "Hour", dateCol: "Date",
    exportButton: "Export", exporting: "Exporting...", exportError: "Error loading data",
    selectCameraHint: 'Select a camera to view its feed',
    // Wording adapts to whichever horizon is selected — same value the
    // gauge itself is showing, so the alert and the display never disagree.
    alertPopupMessage: (value, threshold, horizonMin) => horizonMin === 0
      ? `Alert: current wait at ${value} min (threshold: ${threshold} min)`
      : `Alert: wait predicted at ${value} min in ${horizonMin} min (threshold: ${threshold} min)`,
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
    waitChartTitle: 'WAIT TIME', dayWaitHistory: 'Full-day history',
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
