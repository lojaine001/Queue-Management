import { useEffect, useState, useCallback, useRef } from 'react';
import {
  StyleSheet, Text, View, ScrollView, TouchableOpacity,
  RefreshControl, StatusBar, ActivityIndicator, SafeAreaView, Dimensions, Image,
} from 'react-native';
import { VictoryChart, VictoryLine, VictoryAxis, VictoryArea, VictoryTheme, VictoryPie, VictoryBar } from 'victory-native';

// ── Config ────────────────────────────────────────────────────────────────────
const API_URL = 'https://wildfowl-agenda-curve.ngrok-free.dev';
const REFRESH_MS = 15000;
const H = { 'ngrok-skip-browser-warning': '1' };

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:       '#0d1117',
  surface:  '#161b22',
  surface2: '#1c2128',
  border:   '#30363d',
  text:     '#e6edf3',
  textSub:  '#8b949e',
  textDim:  '#484f58',
  accent:   '#f97316',
  green:    '#3fb950',
  yellow:   '#d29922',
  orange:   '#db6d28',
  red:      '#f85149',
  blue:     '#58a6ff',
  purple:   '#a371f7',
  cyan:     '#39d353',
};

const LANE_STATUS = {
  closed:    { color: C.textDim,  label: 'CLOSED', bg: '#21262d' },
  open:      { color: C.green,    label: 'OPEN',   bg: '#1a2e22' },
  busy:      { color: C.yellow,   label: 'BUSY',   bg: '#2d2a1a' },
  busy_high: { color: C.red,      label: 'BUSY',   bg: '#2d1a1a' },
};

const ALERT_STYLE = {
  red:    { bg: '#2d1a1a', border: C.red,    label: 'CRITICAL', color: C.red    },
  orange: { bg: '#2d2218', border: C.orange, label: 'URGENT',   color: C.orange },
  yellow: { bg: '#2d2a1a', border: C.yellow, label: 'WARNING',  color: C.yellow },
};

const SCENARIO_COLOR = { red: C.red, orange: C.orange, yellow: C.yellow, green: C.green };

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = (n) => (n == null ? '—' : String(n));
const fmtMin = (n) => {
  if (n == null) return '—';
  const m = Math.floor(n);
  const s = Math.round((n - m) * 60);
  return s > 0 ? `${m}:${String(s).padStart(2, '0')}` : `${m}:00`;
};

// ── Shared fetch hook ─────────────────────────────────────────────────────────
function useApi(urls, interval = REFRESH_MS) {
  const [data, setData]       = useState(urls.map(() => null));
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetch_ = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      setError(null);
      const results = await Promise.all(
        urls.map(u => fetch(u, { headers: H }).then(r => r.ok ? r.json() : null))
      );
      setData(results);
    } catch {
      setError('Cannot reach server.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetch_();
    const id = setInterval(() => fetch_(), interval);
    return () => clearInterval(id);
  }, [fetch_]);

  const refresh = () => fetch_(true);
  return { data, loading, error, refreshing, refresh };
}

// ── Translations ─────────────────────────────────────────────────────────────
const T = {
  en: {
    storeName: 'Queue Manager',
    tabs: { live: 'Live', forecast: 'Forecast', today: 'Today' },
    storeClosed: 'Store closed',
    opensIn: 'Opens in',
    lastSession: 'Data from last session',
    // Live
    liveQueueStatus: 'LIVE QUEUE STATUS', updatesEvery15s: 'updates every 15s',
    lane: 'LANE', waiting: 'WAITING', closed: 'CLOSED', fill: 'FILL', avg: 'avg',
    snapshot: 'SNAPSHOT', nowChip: 'now',
    inQueue: 'IN QUEUE', acrossLanes: n => `across ${n} open lane${n !== 1 ? 's' : ''}`,
    avgWait: 'AVG WAIT', min: 'min',
    liveCameras: 'LIVE CAMERAS', updatesEvery60s: 'updates every 60s',
    checkout: 'CHECKOUT', entrance: 'ENTRANCE',
    openBtn: 'OPEN', cannotOpen: 'Cannot Open', falseAlarm: 'False Alarm',
    // Forecast
    forecast15min: '15-MIN FORECAST', nowLabel: 'NOW',
    recOptimal: 'Queue conditions are optimal',
    recOpenMore: (n, saved) => `Open ${n} more lane${n > 1 ? 's' : ''} — saves ~${saved} min wait`,
    recHighDemand: 'High demand — all available lanes recommended',
    recLight: n => `Queue is light — ${n} lane${n > 1 ? 's' : ''} would still keep wait under 5 min`,
    forecastDetail: 'FORECAST DETAIL',
    nextSlot: 'NEXT SLOT', currentTimeWindow: 'Current time window',
    peakWait: 'PEAK WAIT', highestIn15min: 'Highest in 15 min',
    windowEnd: 'WINDOW END', endOf15min: 'End of 15-min view',
    statusSection: 'STATUS',
    trend: 'TREND', waitOver15min: 'Wait over 15 min',
    trendRising: '↑ Rising', trendStable: '→ Stable', trendEasing: '↓ Easing',
    inQueueLabel: 'IN QUEUE', peopleRightNow: 'People right now',
    dataAge: 'DATA AGE', lastDashboardUpdate: 'Last dashboard update',
    justNow: 'just now', minAgo: n => `${n} min ago`,
    outlook60min: '60-MIN OUTLOOK', outlook3h: '3-HOUR OUTLOOK',
    outlook12h: '12-HOUR OUTLOOK', outlook2d: '2-DAY HISTORY',
    waitLegend: 'Wait', arrivalsLegend: 'Arrivals', alertLegend: 'Alert',
    laneScenarios: 'LANE SCENARIOS', tapToSetLanes: 'tap to set open lanes',
    laneLabel: n => `${n} lane${n > 1 ? 's' : ''}`, openBadge: 'OPEN',
    // Today
    summary: 'SUMMARY', totalCustomers: 'TOTAL CUSTOMERS',
    avgCheckout: 'AVG CHECKOUT', busiestHour: 'BUSIEST HOUR',
    lanesUsed: 'LANES USED', busiestLane: id => `busiest: ${id}`,
    alertTime: 'ALERT TIME', equipmentMix: 'EQUIPMENT MIX',
    equipLabel: { trolley: 'Trolley', store_basket: 'Store basket' },
    customers: n => `${n} customers`, noEquipData: 'No equipment data yet today',
    demographics: 'CUSTOMER DEMOGRAPHICS', ageGroups: 'AGE GROUPS',
    entriesByHour: 'ENTRIES BY HOUR', trafficPattern: 'traffic pattern',
    gender: 'GENDER', female: 'Female', male: 'Male',
    visitors: 'visitors', noGenderData: 'No gender data yet today',
    analyzedOf: (n, total) => `based on ${n} of ${total} analyzed`,
  },
  fr: {
    storeName: 'Gestion File',
    tabs: { live: 'En direct', forecast: 'Prévision', today: "Aujourd'hui" },
    storeClosed: 'Magasin fermé',
    opensIn: 'Ouvre dans',
    lastSession: 'Données de la dernière session',
    // Live
    liveQueueStatus: 'ÉTAT FILE EN DIRECT', updatesEvery15s: 'mise à jour toutes les 15s',
    lane: 'FILE', waiting: 'EN ATTENTE', closed: 'FERMÉ', fill: 'REMPLISSAGE', avg: 'moy',
    snapshot: 'APERÇU', nowChip: 'maintenant',
    inQueue: 'EN FILE', acrossLanes: n => `sur ${n} file${n !== 1 ? 's' : ''} ouverte${n !== 1 ? 's' : ''}`,
    avgWait: 'ATTENTE MOY.', min: 'min',
    liveCameras: 'CAMÉRAS EN DIRECT', updatesEvery60s: 'mise à jour toutes les 60s',
    checkout: 'CAISSE', entrance: 'ENTRÉE',
    openBtn: 'OUVRIR', cannotOpen: "Impossible d'ouvrir", falseAlarm: 'Fausse alarme',
    // Forecast
    forecast15min: 'PRÉVISION 15 MIN', nowLabel: 'MAINTENANT',
    recOptimal: 'Conditions de file optimales',
    recOpenMore: (n, saved) => `Ouvrir ${n} file${n > 1 ? 's' : ''} de plus — économise ~${saved} min`,
    recHighDemand: 'Forte demande — toutes les files disponibles recommandées',
    recLight: n => `File légère — ${n} file${n > 1 ? 's' : ''} suffit pour maintenir sous 5 min`,
    forecastDetail: 'DÉTAIL PRÉVISION',
    nextSlot: 'PROCHAIN CRÉNEAU', currentTimeWindow: 'Fenêtre de temps actuelle',
    peakWait: 'ATTENTE MAX', highestIn15min: 'Plus haute en 15 min',
    windowEnd: 'FIN DE FENÊTRE', endOf15min: 'Fin de la vue 15 min',
    statusSection: 'STATUT',
    trend: 'TENDANCE', waitOver15min: 'Attente sur 15 min',
    trendRising: '↑ En hausse', trendStable: '→ Stable', trendEasing: '↓ En baisse',
    inQueueLabel: 'EN FILE', peopleRightNow: 'Personnes actuellement',
    dataAge: 'ÂGE DONNÉES', lastDashboardUpdate: 'Dernière mise à jour',
    justNow: "à l'instant", minAgo: n => `il y a ${n} min`,
    outlook60min: 'PRÉVISION 60 MIN', outlook3h: 'PRÉVISION 3H',
    outlook12h: 'PRÉVISION 12H', outlook2d: 'HISTORIQUE 2J',
    waitLegend: 'Attente', arrivalsLegend: 'Arrivées', alertLegend: 'Alerte',
    laneScenarios: 'SCÉNARIOS FILES', tapToSetLanes: 'toucher pour définir',
    laneLabel: n => `${n} file${n > 1 ? 's' : ''}`, openBadge: 'OUVERT',
    // Today
    summary: 'RÉSUMÉ', totalCustomers: 'CLIENTS TOTAL',
    avgCheckout: 'CAISSE MOY.', busiestHour: 'HEURE DE POINTE',
    lanesUsed: 'FILES UTILISÉES', busiestLane: id => `la + chargée : ${id}`,
    alertTime: 'TEMPS EN ALERTE', equipmentMix: 'ÉQUIPEMENT',
    equipLabel: { trolley: 'Chariot', store_basket: 'Panier' },
    customers: n => `${n} clients`, noEquipData: "Aucune donnée d'équipement aujourd'hui",
    demographics: 'DÉMOGRAPHIE CLIENTS', ageGroups: "TRANCHES D'ÂGE",
    entriesByHour: 'ENTRÉES PAR HEURE', trafficPattern: 'flux de trafic',
    gender: 'GENRE', female: 'Femme', male: 'Homme',
    visitors: 'visiteurs', noGenderData: "Pas de données de genre aujourd'hui",
    analyzedOf: (n, total) => `basé sur ${n} des ${total} analysés`,
  },
};

// ── Store hours check (phone local time) ─────────────────────────────────────
const SHOP_OPEN_MIN  = 8  * 60 + 30;  // 08:30
const SHOP_CLOSE_MIN = 20 * 60 + 30;  // 20:30

function isStoreOpen() {
  const now = new Date();
  const tot = now.getHours() * 60 + now.getMinutes();
  return tot >= SHOP_OPEN_MIN && tot < SHOP_CLOSE_MIN;
}

function closedBannerText(lang) {
  const tr = T[lang];
  const now = new Date();
  const tot = now.getHours() * 60 + now.getMinutes();
  if (tot < SHOP_OPEN_MIN) {
    const diff = SHOP_OPEN_MIN - tot;
    const h = Math.floor(diff / 60), m = diff % 60;
    return h > 0 ? `${tr.storeClosed} · ${tr.opensIn} ${h}h ${m.toString().padStart(2,'0')}m`
                 : `${tr.storeClosed} · ${tr.opensIn} ${m} min`;
  }
  return `${tr.storeClosed} · ${tr.lastSession}`;
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]   = useState('forecast');
  const [lang, setLang] = useState('en');
  const [time, setTime] = useState(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  const [storeOpen, setStoreOpen] = useState(isStoreOpen());

  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      setStoreOpen(isStoreOpen());
    }, 30000);
    return () => clearInterval(id);
  }, []);

  const tr = T[lang];

  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.surface} />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <View style={s.header}>
        <View style={s.headerLeft}>
          <View style={s.logoChip}>
            <Text style={s.logoText}>IQMS</Text>
          </View>
          <Text style={s.storeName}>{tr.storeName}</Text>
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <View style={s.langToggle}>
            {['en', 'fr'].map(l => (
              <TouchableOpacity key={l} onPress={() => setLang(l)} activeOpacity={0.75}
                style={[s.langBtn, lang === l && s.langBtnActive]}>
                <Text style={[s.langFlag, lang === l && s.langFlagActive]}>{l === 'en' ? 'EN' : 'FR'}</Text>
              </TouchableOpacity>
            ))}
          </View>
          <View style={s.liveChip}>
            <View style={s.liveDot} />
            <Text style={s.liveText}>{time}</Text>
          </View>
        </View>
      </View>

      {/* ── Tab Bar ────────────────────────────────────────────────────────── */}
      <View style={s.tabBar}>
        {['live', 'forecast', 'today'].map(t => (
          <TouchableOpacity
            key={t}
            style={[s.tabBtn, tab === t && s.tabBtnActive]}
            onPress={() => setTab(t)}
            activeOpacity={0.8}
          >
            <Text style={[s.tabText, tab === t && s.tabTextActive]}>
              {tr.tabs[t]}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* ── Closed banner ──────────────────────────────────────────────────── */}
      {!storeOpen && (
        <View style={s.closedBanner}>
          <Text style={s.closedIcon}>🔒</Text>
          <Text style={s.closedText}>{closedBannerText(lang)}</Text>
        </View>
      )}

      {/* ── Screens ────────────────────────────────────────────────────────── */}
      {tab === 'live'     && <LiveScreen lang={lang} />}
      {tab === 'forecast' && <ForecastScreen lang={lang} />}
      {tab === 'today'    && <TodayScreen lang={lang} />}
    </SafeAreaView>
  );
}

// ── Live Screen ───────────────────────────────────────────────────────────────
function LiveScreen({ lang }) {
  const tr = T[lang];
  const { data, loading, error, refreshing, refresh } = useApi([
    `${API_URL}/live-lanes`,
    `${API_URL}/alerts`,
  ]);
  const { data: snapData } = useApi([
    `${API_URL}/snapshot/checkout`,
    `${API_URL}/snapshot/entrance`,
  ], 60000);
  const camImgH = Math.round((Dimensions.get('window').width - 28 - 10) / 2 * 3 / 4);
  const [lanesData, alertData] = data;

  if (loading) return <Loader />;

  const lanes    = lanesData?.lanes    || [];
  const snapshot = lanesData?.snapshot || {};
  const alert    = alertData;
  const alertS   = alert?.level ? ALERT_STYLE[alert.level] : null;
  const allLanesOpen = lanes.length > 0 && lanes.every(l => ['open', 'busy', 'busy_high'].includes(l.status));

  const sendResponse = async (response) => {
    try {
      await fetch(`${API_URL}/alert-response`, {
        method: 'POST',
        headers: { ...H, 'Content-Type': 'application/json' },
        body: JSON.stringify({ response }),
      });
    } catch {}
  };

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.screenContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={C.accent} />}
      showsVerticalScrollIndicator={false}
    >
      {error && <ErrorBanner msg={error} />}

      {/* Alert Banner */}
      {alert?.level && alertS && (
        <View style={[s.alertBanner, { backgroundColor: alertS.bg, borderColor: alertS.border }]}>
          <View style={s.alertBannerTop}>
            <View style={[s.alertIconBox, { backgroundColor: alertS.border + '33' }]}>
              <Text style={{ fontSize: 14 }}>⚠</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[s.alertBannerTitle, { color: alertS.color }]}>{alert.message}</Text>
              {allLanesOpen && (
                <Text style={[s.alertAllOpenNote, { color: alertS.color }]}>
                  {lang === 'fr' ? 'Toutes les files sont déjà ouvertes' : 'All lanes already open — monitoring'}
                </Text>
              )}
            </View>
            {!allLanesOpen && (
              <TouchableOpacity
                style={[s.openBtn, { backgroundColor: alertS.border }]}
                onPress={() => sendResponse('opening_lane')}
                activeOpacity={0.8}
              >
                <Text style={s.openBtnText}>{tr.openBtn}</Text>
              </TouchableOpacity>
            )}
          </View>
          <View style={s.alertBtnRow}>
            {[
              { key: 'cannot_open', label: tr.cannotOpen },
              { key: 'false_alarm', label: tr.falseAlarm },
            ].map(b => (
              <TouchableOpacity key={b.key} style={s.alertSecondBtn} onPress={() => sendResponse(b.key)} activeOpacity={0.8}>
                <Text style={s.alertSecondBtnText}>{b.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* Section Label */}
      <View style={s.sectionRow}>
        <View style={[s.sectionDot, { backgroundColor: C.green }]} />
        <Text style={s.sectionLabel}>{tr.liveQueueStatus}</Text>
        <Text style={s.sectionRight}>{tr.updatesEvery15s}</Text>
      </View>

      {/* Lane Cards */}
      {lanes.map(lane => {
        const ls = LANE_STATUS[lane.status] || LANE_STATUS.closed;
        const fillPct = Math.min((lane.fill / lane.fill_max) * 100, 100);
        return (
          <View key={lane.lane_number} style={[s.laneCard, { backgroundColor: ls.bg, borderColor: ls.color + '44' }]}>
            <View style={s.laneCardTop}>
              <Text style={s.laneNumber}>{tr.lane} {String(lane.lane_number).padStart(2, '0')}</Text>
              <View style={[s.statusBadge, { backgroundColor: ls.color + '22', borderColor: ls.color }]}>
                <View style={[s.statusDot2, { backgroundColor: ls.color }]} />
                <Text style={[s.statusBadgeText, { color: ls.color }]}>{ls.label}</Text>
              </View>
              <Text style={[s.laneAvg, { color: ls.color }]}>
                {tr.avg} {lane.status === 'closed' ? '—' : fmtMin(lane.avg_wait_min)}
              </Text>
            </View>

            <View style={s.laneCardMid}>
              <Text style={[s.waitingNum, { color: ls.color }]}>
                {lane.status === 'closed' ? '—' : lane.waiting}
              </Text>
              <Text style={s.waitingLabel}>{tr.waiting}</Text>

              <View style={s.fillSection}>
                <View style={s.fillBarBg}>
                  <View style={[s.fillBarFg, { width: `${fillPct}%`, backgroundColor: ls.color }]} />
                </View>
                <Text style={s.fillLabel}>
                  {lane.status === 'closed' ? tr.closed : `${tr.fill}   ${lane.fill} / ${lane.fill_max}`}
                </Text>
              </View>
            </View>
          </View>
        );
      })}

      {/* Snapshot */}
      <View style={s.sectionRow}>
        <View style={[s.sectionDot, { backgroundColor: C.purple }]} />
        <Text style={s.sectionLabel}>{tr.snapshot}</Text>
        <Text style={s.sectionRight}>{tr.nowChip}</Text>
      </View>

      <View style={s.snapshotRow}>
        <View style={[s.snapshotCard, { borderColor: C.border }]}>
          <Text style={s.snapshotVal}>{fmt(snapshot.total_in_queue)}</Text>
          <Text style={s.snapshotLbl}>{tr.inQueue}</Text>
          <Text style={s.snapshotSub}>{tr.acrossLanes(snapshot.open_lanes || 0)}</Text>
        </View>
        <View style={[s.snapshotCard, { borderColor: C.border }]}>
          <Text style={[s.snapshotVal, { color: C.yellow }]}>{fmtMin(snapshot.avg_wait_min)}</Text>
          <Text style={s.snapshotLbl}>{tr.avgWait}</Text>
          <Text style={s.snapshotSub}>{tr.min}</Text>
        </View>
      </View>

      {/* Camera snapshots */}
      <View style={[s.sectionRow, { marginTop: 8 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.textDim }]} />
        <Text style={s.sectionLabel}>{tr.liveCameras}</Text>
        <Text style={s.sectionRight}>{tr.updatesEvery60s}</Text>
      </View>
      <View style={s.cameraRow}>
        {[
          { label: tr.checkout, img: snapData?.[0]?.image },
          { label: tr.entrance, img: snapData?.[1]?.image },
        ].map(cam => (
          <View key={cam.label} style={[s.cameraCard, { borderColor: C.border }]}>
            <Text style={s.cameraLabel}>{cam.label}</Text>
            {cam.img
              ? <Image source={{ uri: cam.img }} style={[s.cameraImage, { height: camImgH }]} resizeMode="contain" />
              : <View style={[s.cameraImage, { height: camImgH }]} />}
          </View>
        ))}
      </View>

      <View style={s.spacer} />
    </ScrollView>
  );
}

// ── Forecast Screen ───────────────────────────────────────────────────────────
function ForecastScreen({ lang }) {
  const tr = T[lang];
  const [chartHorizon, setChartHorizon] = useState('60min');
  const { data, loading, error, refreshing, refresh } = useApi([
    `${API_URL}/forecast`,
    `${API_URL}/forecast-chart`,
    `${API_URL}/forecast-chart-3h`,
    `${API_URL}/forecast-chart-12h`,
    `${API_URL}/forecast-chart-2d`,
  ]);
  const { data: snapData } = useApi([
    `${API_URL}/snapshot/checkout`,
    `${API_URL}/snapshot/entrance`,
  ], 60000);
  const [forecastData, chartData, chartData3h, chartData12h, chartData2d] = data;
  const camThumbH = Math.round((Dimensions.get('window').width - 28 - 10) / 2 * 3 / 4 * 0.55);

  if (loading) return <Loader />;

  const scenarios   = forecastData?.lane_scenarios || [];
  const waitNow     = forecastData?.wait_now_min;
  const wait5       = forecastData?.wait_5_min;
  const wait10      = forecastData?.wait_10_min;
  const wait15      = forecastData?.wait_15_min;
  const activeLanes = forecastData?.current_lanes || 1;
  const queueNow    = forecastData?.queue_now;
  const updatedAt   = forecastData?.updated_at;
  const slots        = chartData?.slots    || [];
  const slots3h      = chartData3h?.slots  || [];
  const slots12h     = chartData12h?.slots || [];
  const slots2d      = chartData2d?.slots  || [];
  const activeSlots  = chartHorizon === '3h' ? slots3h
                     : chartHorizon === '12h' ? slots12h
                     : chartHorizon === '2d'  ? slots2d
                     : slots;

  const waitColor = (w) => {
    if (w == null) return C.textSub;
    if (w > 10) return C.red;
    if (w > 7)  return C.orange;
    if (w > 4)  return C.yellow;
    return C.green;
  };

  // Trend: compare now vs +15min
  const trendDelta = (waitNow != null && wait15 != null) ? wait15 - waitNow : null;
  const trend = trendDelta == null ? null : trendDelta > 1 ? 'rising' : trendDelta < -1 ? 'easing' : 'stable';
  const trendLabel = { rising: tr.trendRising, stable: tr.trendStable, easing: tr.trendEasing };
  const trendColor = { rising: C.red, stable: C.yellow, easing: C.green };

  // Data freshness
  const dataAgeMin = updatedAt
    ? Math.round((Date.now() - new Date(updatedAt).getTime()) / 60000)
    : null;
  const dataAgeLabel = dataAgeMin == null ? '—'
    : dataAgeMin < 1 ? tr.justNow
    : tr.minAgo(dataAgeMin);
  const dataAgeColor = dataAgeMin == null ? C.textSub
    : dataAgeMin > 5 ? C.red
    : dataAgeMin > 2 ? C.yellow
    : C.green;

  // Action recommendation (client-side)
  const currentWait = waitNow ?? 0;
  const betterScenario = scenarios.find(sc => sc.lanes > activeLanes && sc.est_wait_min < currentWait - 1);
  const worseScenario  = scenarios.find(sc => sc.lanes < activeLanes && sc.est_wait_min <= 5);
  let recommendation = null;
  let recommendationColor = C.textSub;
  if (currentWait <= 5 && worseScenario) {
    recommendation = tr.recLight(worseScenario.lanes);
    recommendationColor = C.green;
  } else if (currentWait > 5 && betterScenario) {
    const saved = Math.round(currentWait - betterScenario.est_wait_min);
    recommendation = tr.recOpenMore(betterScenario.lanes - activeLanes, saved);
    recommendationColor = C.orange;
  } else if (currentWait > 10) {
    recommendation = tr.recHighDemand;
    recommendationColor = C.red;
  } else if (currentWait <= 5) {
    recommendation = tr.recOptimal;
    recommendationColor = C.green;
  }

  // Chart data
  const waitLine     = activeSlots.map((sl, i) => ({ x: i, y: sl.wait_min }));
  const arrivalsLine = activeSlots.map((sl, i) => ({ x: i, y: sl.arrivals }));
  const alertLine    = activeSlots.map((_, i) => ({ x: i, y: 5 }));
  const chartW       = Dimensions.get('window').width - 28;

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.screenContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={C.accent} />}
      showsVerticalScrollIndicator={false}
    >
      {error && <ErrorBanner msg={error} />}

      {/* Compact camera preview */}
      {(snapData?.[0]?.image || snapData?.[1]?.image) && (
        <View style={s.cameraRow}>
          {[
            { label: tr.checkout, img: snapData?.[0]?.image },
            { label: tr.entrance, img: snapData?.[1]?.image },
          ].map(cam => (
            <View key={cam.label} style={[s.cameraCard, { borderColor: C.border }]}>
              <Text style={s.cameraLabel}>{cam.label}</Text>
              {cam.img
                ? <Image source={{ uri: cam.img }} style={[s.cameraImage, { height: camThumbH }]} resizeMode="contain" />
                : <View style={[s.cameraImage, { height: camThumbH }]} />}
            </View>
          ))}
        </View>
      )}

      {/* Forecast header */}
      <View style={s.sectionRow}>
        <View style={[s.sectionDot, { backgroundColor: C.yellow }]} />
        <Text style={s.sectionLabel}>{tr.forecast15min}</Text>
        <Text style={s.sectionRight}></Text>
      </View>

      {/* Wait cards NOW / +5 / +10 / +15 */}
      <View style={s.waitCardsRow}>
        {[
          { label: tr.nowLabel, value: waitNow },
          { label: '+5 MIN',  value: wait5   },
          { label: '+10 MIN', value: wait10  },
          { label: '+15 MIN', value: wait15  },
        ].map(card => (
          <View key={card.label} style={[s.waitCard, { borderColor: C.border }]}>
            <Text style={s.waitCardLabel}>{card.label}</Text>
            <Text
              style={[s.waitCardValue, { color: waitColor(card.value) }]}
              adjustsFontSizeToFit
              numberOfLines={1}
            >
              {fmtMin(card.value)}
            </Text>
            <Text style={s.waitCardUnit}>{tr.min}</Text>
          </View>
        ))}
      </View>

      {/* Recommendation banner */}
      {recommendation && (
        <View style={[s.recommendationBanner, { borderColor: recommendationColor + '55', backgroundColor: recommendationColor + '11' }]}>
          <Text style={[s.recommendationText, { color: recommendationColor }]}>{recommendation}</Text>
        </View>
      )}

      {/* Forecast Detail */}
      <View style={[s.sectionRow, { marginTop: 10 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.blue }]} />
        <Text style={s.sectionLabel}>{tr.forecastDetail}</Text>
      </View>

      <View style={s.detailRow}>
        {[
          { label: tr.nextSlot,  value: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sub: tr.currentTimeWindow },
          { label: tr.peakWait,  value: (() => { const all = [waitNow, wait5, wait10, wait15].filter(v => v != null); return all.length ? fmtMin(Math.max(...all)) : '—'; })(), sub: tr.highestIn15min, color: waitColor(Math.max(...[waitNow, wait5, wait10, wait15].filter(v => v != null))) },
          { label: tr.windowEnd, value: new Date(Date.now() + 15 * 60000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sub: tr.endOf15min },
        ].map(item => (
          <View key={item.label} style={[s.detailCard, { borderColor: C.border }]}>
            <Text style={s.detailLabel}>{item.label}</Text>
            <Text style={[s.detailValue, item.color ? { color: item.color } : {}]} adjustsFontSizeToFit numberOfLines={1}>
              {item.value}
            </Text>
            <Text style={s.detailSub}>{item.sub}</Text>
          </View>
        ))}
      </View>

      {/* Status row: TREND / QUEUE NOW / DATA AGE */}
      <View style={[s.sectionRow, { marginTop: 2 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.blue }]} />
        <Text style={s.sectionLabel}>{tr.statusSection}</Text>
      </View>

      <View style={s.detailRow}>
        {[
          {
            label: tr.trend,
            value: trend ? trendLabel[trend] : '—',
            sub:   tr.waitOver15min,
            color: trend ? trendColor[trend] : C.textSub,
          },
          {
            label: tr.inQueueLabel,
            value: queueNow != null ? `${queueNow}` : '—',
            sub:   tr.peopleRightNow,
            color: C.text,
          },
          {
            label: tr.dataAge,
            value: dataAgeLabel,
            sub:   tr.lastDashboardUpdate,
            color: dataAgeColor,
          },
        ].map(item => (
          <View key={item.label} style={[s.detailCard, { borderColor: C.border }]}>
            <Text style={s.detailLabel}>{item.label}</Text>
            <Text style={[s.detailValue, { color: item.color }]} adjustsFontSizeToFit numberOfLines={1}>
              {item.value}
            </Text>
            <Text style={s.detailSub}>{item.sub}</Text>
          </View>
        ))}
      </View>

      {/* Chart with horizon toggle */}
      {(slots.length > 2 || slots3h.length > 2) && (
        <>
          <View style={[s.sectionRow, { marginTop: 10 }]}>
            <View style={[s.sectionDot, { backgroundColor: C.accent }]} />
            <Text style={s.sectionLabel}>
              {chartHorizon === '3h'  ? tr.outlook3h
               : chartHorizon === '12h' ? tr.outlook12h
               : chartHorizon === '2d'  ? tr.outlook2d
               : tr.outlook60min}
            </Text>
            <View style={s.horizonToggle}>
              {[
                { key: '60min', label: '60m' },
                { key: '3h',   label: '3h'  },
                { key: '12h',  label: '12h' },
                { key: '2d',   label: '2d'  },
              ].map(h => (
                <TouchableOpacity
                  key={h.key}
                  style={[s.horizonBtn, chartHorizon === h.key && s.horizonBtnActive]}
                  onPress={() => setChartHorizon(h.key)}
                  activeOpacity={0.8}
                >
                  <Text style={[s.horizonBtnText, chartHorizon === h.key && s.horizonBtnTextActive]}>
                    {h.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            <View style={[s.chartLegend, { marginLeft: 8 }]}>
              <View style={[s.legendDot, { backgroundColor: C.orange }]} />
              <Text style={s.legendTxt}>{tr.waitLegend}</Text>
              <View style={[s.legendDot, { backgroundColor: C.blue, marginLeft: 8 }]} />
              <Text style={s.legendTxt}>{tr.arrivalsLegend}</Text>
              <View style={[s.legendDot, { backgroundColor: C.red, marginLeft: 8 }]} />
              <Text style={s.legendTxt}>{tr.alertLegend}</Text>
            </View>
          </View>

          <View style={[s.chartCard, { borderColor: C.border }]}>
            <VictoryChart
              width={chartW}
              height={180}
              padding={{ top: 10, bottom: 30, left: 36, right: 16 }}
              theme={VictoryTheme.material}
            >
              <VictoryAxis
                tickCount={chartHorizon === '2d' ? 8 : chartHorizon === '12h' ? 6 : 5}
                tickFormat={(i) => activeSlots[Math.round(i)]?.time ?? ''}
                style={{
                  axis: { stroke: C.border },
                  tickLabels: { fill: C.textDim, fontSize: 9 },
                  grid: { stroke: 'transparent' },
                }}
              />
              <VictoryAxis
                dependentAxis
                style={{
                  axis: { stroke: C.border },
                  tickLabels: { fill: C.textDim, fontSize: 9 },
                  grid: { stroke: C.border, strokeDasharray: '4' },
                }}
              />
              <VictoryArea
                data={arrivalsLine}
                style={{ data: { fill: C.blue + '22', stroke: C.blue, strokeWidth: 1.5 } }}
                interpolation="monotoneX"
              />
              <VictoryLine
                data={alertLine}
                style={{ data: { stroke: C.red, strokeWidth: 1, strokeDasharray: '6' } }}
              />
              <VictoryLine
                data={waitLine}
                style={{ data: { stroke: C.orange, strokeWidth: 2 } }}
                interpolation="monotoneX"
              />
            </VictoryChart>
          </View>
        </>
      )}

      {/* Lane Scenarios — tap to toggle open/closed */}
      <View style={[s.sectionRow, { marginTop: 8 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.green }]} />
        <Text style={s.sectionLabel}>{tr.laneScenarios}</Text>
        <Text style={s.sectionRight}>{tr.tapToSetLanes}</Text>
      </View>

      <View style={[s.scenariosCard, { borderColor: C.border }]}>
        {(() => {
          const maxScenarioWait = Math.max(...scenarios.map(sc => sc.est_wait_min || 0), 1);
          return scenarios.map(sc => {
          const isActive = sc.lanes === activeLanes;
          const color = SCENARIO_COLOR[sc.color] || C.textSub;
          const barWidth = sc.est_wait_min > 0 ? Math.min((sc.est_wait_min / maxScenarioWait) * 100, 100) : 5;
          return (
            <TouchableOpacity
              key={sc.lanes}
              style={[s.scenarioRow, isActive && s.scenarioRowActive]}
              onPress={async () => {
                try {
                  await fetch(`${API_URL}/set-lanes`, {
                    method: 'POST',
                    headers: { ...H, 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lanes: sc.lanes }),
                  });
                  refresh();
                } catch {}
              }}
              activeOpacity={0.75}
            >
              <View style={[s.scenarioLaneBox, isActive && { backgroundColor: C.accent }]}>
                <Text style={[s.scenarioLaneText, isActive && { color: '#fff' }]}>
                  {tr.laneLabel(sc.lanes)}
                </Text>
              </View>
              <View style={s.scenarioBarBg}>
                <View style={[s.scenarioBarFg, { width: `${barWidth}%`, backgroundColor: color }]} />
              </View>
              <View style={s.scenarioRight}>
                <Text style={[s.scenarioWait, { color }]}>{fmtMin(sc.est_wait_min)}</Text>
                {isActive && <Text style={s.scenarioOpen}>{tr.openBadge}</Text>}
              </View>
            </TouchableOpacity>
          );
        });
        })()}
      </View>

      <View style={s.spacer} />
    </ScrollView>
  );
}

// ── Today Screen ──────────────────────────────────────────────────────────────
function TodayScreen({ lang }) {
  const tr = T[lang];
  const { data, loading, error, refreshing, refresh } = useApi([
    `${API_URL}/day-recap`,
  ]);
  const [recap] = data;

  if (loading) return <Loader />;

  const equipment           = recap?.equipment             || [];
  const demographicsGender  = (recap?.demographics_gender  || []).filter(g => g.key !== 'unknown');
  const demographicsAge     = recap?.demographics_age      || [];
  const entriesByHour       = recap?.entries_by_hour       || [];
  const chartW              = Dimensions.get('window').width - 28;

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.screenContent}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={C.accent} />}
      showsVerticalScrollIndicator={false}
    >
      {error && <ErrorBanner msg={error} />}

      {/* Summary */}
      <View style={s.sectionRow}>
        <View style={[s.sectionDot, { backgroundColor: C.blue }]} />
        <Text style={s.sectionLabel}>{tr.summary}</Text>
        {recap?.date && <Text style={s.sectionRight}>{recap.date}</Text>}
      </View>

      <View style={[s.summaryCard, { borderColor: C.border }]}>
        {[
          { label: tr.totalCustomers, value: recap?.total_customers?.toLocaleString(), color: C.cyan,   big: true  },
          { label: tr.avgCheckout,    value: fmtMin(recap?.avg_wait_min),              color: C.text,   big: false,
            suffix: tr.min },
          { label: tr.busiestHour,
            value: recap?.peak_hour ? `${recap.peak_hour}` : '—',
            extra: recap?.peak_hour_end ? `– ${recap.peak_hour_end} · ${recap.peak_count}` : '',
            color: C.text, big: false },
          { label: tr.lanesUsed,
            value: recap?.lanes_today != null ? `${recap.lanes_today}` : '—',
            extra: recap?.busiest_lane ? tr.busiestLane(recap.busiest_lane) : '',
            color: C.text, big: false },
          { label: tr.alertTime,
            value: recap?.alert_minutes != null ? `${recap.alert_minutes}` : '—',
            suffix: tr.min,
            color: recap?.alert_minutes > 0 ? C.red : C.text, big: false },
        ].map((item, i, arr) => (
          <View key={item.label}>
            <View style={s.summaryRow_}>
              <Text style={s.summaryLbl}>{item.label}</Text>
              <View style={s.summaryValRow}>
                <Text style={[s.summaryVal, item.big && s.summaryValBig, { color: item.color }]}>
                  {item.value ?? '—'}
                </Text>
                {item.suffix && <Text style={s.summaryUnit}> {item.suffix}</Text>}
                {item.extra  && <Text style={s.summaryExtra}> {item.extra}</Text>}
              </View>
            </View>
            {i < arr.length - 1 && <View style={[s.divider, { backgroundColor: C.border }]} />}
          </View>
        ))}
      </View>

      {/* Equipment Mix */}
      <View style={[s.sectionRow, { marginTop: 8 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.purple }]} />
        <Text style={s.sectionLabel}>{tr.equipmentMix}</Text>
      </View>

      <View style={[s.equipCard, { borderColor: C.border }]}>
        {equipment.filter(eq => eq.type !== 'personal_bag').map((eq, i, arr) => (
          <View key={eq.type}>
            <View style={s.equipRow}>
              <View style={[s.equipIcon, { backgroundColor: eq.color + '22' }]}>
                <Text style={{ fontSize: 16 }}>
                  {eq.type === 'trolley' ? '🛒' : '🧺'}
                </Text>
              </View>
              <View style={{ flex: 1 }}>
                <View style={s.equipLabelRow}>
                  <Text style={s.equipLabel}>{tr.equipLabel[eq.type] || eq.label}</Text>
                  <Text style={[s.equipPct, { color: eq.color }]}>{eq.percent}%</Text>
                </View>
                <View style={s.equipBarBg}>
                  <View style={[s.equipBarFg, { width: `${eq.percent}%`, backgroundColor: eq.color }]} />
                </View>
                <Text style={s.equipCount}>{tr.customers(eq.count.toLocaleString())}</Text>
              </View>
            </View>
            {i < arr.length - 1 && <View style={[s.divider, { backgroundColor: C.border }]} />}
          </View>
        ))}
        {equipment.length === 0 && (
          <Text style={[s.emptyText, { color: C.textDim }]}>{tr.noEquipData}</Text>
        )}
      </View>

      {/* Customer Demographics */}
      <View style={[s.sectionRow, { marginTop: 8 }]}>
        <View style={[s.sectionDot, { backgroundColor: C.accent }]} />
        <Text style={s.sectionLabel}>{tr.demographics}</Text>
      </View>

      <View style={[s.demoCard, { borderColor: C.border }]}>

        {/* Gender donut */}
        <Text style={s.demoSubTitle}>{tr.gender}</Text>
        {demographicsGender.length > 0 ? (() => {
          const analyzedTotal = demographicsGender.reduce((sum, g) => sum + g.count, 0);
          const genderLabel = { Female: tr.female, Male: tr.male };
          return (
            <>
              {recap?.total_customers > 0 && (
                <Text style={[s.demoSampleNote, { color: C.textDim }]}>
                  {tr.analyzedOf(analyzedTotal.toLocaleString(), recap.total_customers.toLocaleString())}
                </Text>
              )}
              <View style={s.genderRow}>
                <VictoryPie
                  data={demographicsGender.map(g => ({ x: g.label, y: g.count }))}
                  colorScale={demographicsGender.map(g => g.color)}
                  width={160} height={160}
                  innerRadius={50}
                  padding={10}
                  labels={() => null}
                />
                <View style={s.genderLegend}>
                  {demographicsGender.map(g => (
                    <View key={g.key} style={s.genderLegendRow}>
                      <View style={[s.genderDot, { backgroundColor: g.color }]} />
                      <View>
                        <Text style={[s.genderPct, { color: g.color }]}>{g.percent}%</Text>
                        <Text style={s.genderLbl}>{genderLabel[g.label] || g.label}</Text>
                        <Text style={s.genderCount}>{g.count.toLocaleString()} {tr.visitors}</Text>
                      </View>
                    </View>
                  ))}
                </View>
              </View>
            </>
          );
        })() : <Text style={[s.emptyText, { color: C.textDim }]}>{tr.noGenderData}</Text>}

        {demographicsAge.length > 0 && (
          <>
            <View style={[s.divider, { backgroundColor: C.border, marginVertical: 10 }]} />
            {/* Age Groups */}
            <Text style={s.demoSubTitle}>{tr.ageGroups}</Text>
            <View style={s.ageRow}>
              {demographicsAge.map(ag => (
                <View key={ag.group} style={s.ageCard}>
                  <Text style={[s.agePct, { color: ag.color }]}>{ag.percent}%</Text>
                  <View style={s.ageBarBg}>
                    <View style={[s.ageBarFg, { height: `${ag.percent}%`, backgroundColor: ag.color }]} />
                  </View>
                  <Text style={s.ageLabel}>{ag.group}</Text>
                  <Text style={s.ageCount}>{ag.count.toLocaleString()}</Text>
                </View>
              ))}
            </View>
          </>
        )}
      </View>

      {/* Entries by Hour */}
      {entriesByHour.length > 0 && (
        <>
          <View style={[s.sectionRow, { marginTop: 8 }]}>
            <View style={[s.sectionDot, { backgroundColor: C.blue }]} />
            <Text style={s.sectionLabel}>{tr.entriesByHour}</Text>
            <Text style={s.sectionRight}>{tr.trafficPattern}</Text>
          </View>
          <View style={[s.chartCard, { borderColor: C.border }]}>
            <VictoryChart
              width={chartW} height={200}
              padding={{ top: 20, bottom: 36, left: 40, right: 16 }}
              domainPadding={{ x: 20 }}
            >
              <VictoryAxis
                tickFormat={t => entriesByHour[Math.round(t)]?.hour ?? ''}
                tickCount={Math.min(entriesByHour.length, 6)}
                style={{
                  axis: { stroke: C.border },
                  tickLabels: { fill: C.textDim, fontSize: 9 },
                  grid: { stroke: 'transparent' },
                }}
              />
              <VictoryAxis
                dependentAxis
                style={{
                  axis: { stroke: C.border },
                  tickLabels: { fill: C.textDim, fontSize: 9 },
                  grid: { stroke: C.border, strokeDasharray: '4' },
                }}
              />
              <VictoryBar
                data={entriesByHour.map((h, i) => ({ x: i, y: h.count, peak: h.is_peak }))}
                style={{
                  data: {
                    fill: ({ datum }) => datum.peak ? C.red : C.blue,
                    borderRadius: 3,
                  },
                }}
                barWidth={Math.max(8, (chartW - 56) / entriesByHour.length - 4)}
              />
            </VictoryChart>
          </View>
        </>
      )}

      <View style={s.spacer} />
    </ScrollView>
  );
}

// ── Shared components ─────────────────────────────────────────────────────────
function Loader() {
  return (
    <View style={s.loader}>
      <ActivityIndicator size="large" color={C.accent} />
    </View>
  );
}

function ErrorBanner({ msg }) {
  return (
    <View style={s.errorBanner}>
      <Text style={s.errorText}>⚠ {msg}</Text>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root:           { flex: 1, backgroundColor: C.bg },
  loader:         { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: C.bg },
  screen:         { flex: 1, backgroundColor: C.bg },
  screenContent:  { padding: 14 },
  spacer:         { height: 32 },

  // Header
  header:         { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
                    backgroundColor: C.surface, paddingHorizontal: 16, paddingVertical: 12,
                    borderBottomWidth: 1, borderBottomColor: C.border },
  headerLeft:     { flexDirection: 'row', alignItems: 'center', gap: 10 },
  logoChip:       { backgroundColor: C.accent, paddingHorizontal: 9, paddingVertical: 5, borderRadius: 6 },
  logoText:       { color: '#fff', fontWeight: '800', fontSize: 14, letterSpacing: 1 },
  storeName:      { color: C.textSub, fontSize: 12, fontWeight: '500' },
  liveChip:       { flexDirection: 'row', alignItems: 'center', gap: 6,
                    backgroundColor: '#1a2e22', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20 },
  liveDot:        { width: 6, height: 6, borderRadius: 3, backgroundColor: C.green },
  liveText:       { color: C.green, fontSize: 12, fontWeight: '700' },
  langToggle:     { flexDirection: 'row', backgroundColor: C.surface2, borderRadius: 16,
                    borderWidth: 1, borderColor: C.border, overflow: 'hidden' },
  langBtn:        { paddingHorizontal: 8, paddingVertical: 4 },
  langBtnActive:  { backgroundColor: C.border },
  langFlag:       { fontSize: 11, fontWeight: '700', color: C.textSub, letterSpacing: 0.5 },
  langFlagActive: { color: C.text },

  // Tab bar
  tabBar:         { flexDirection: 'row', backgroundColor: C.surface,
                    borderBottomWidth: 1, borderBottomColor: C.border },
  tabBtn:         { flex: 1, paddingVertical: 12, alignItems: 'center' },
  tabBtnActive:   { borderBottomWidth: 2, borderBottomColor: C.accent },
  tabText:        { color: C.textSub, fontSize: 13, fontWeight: '500' },
  tabTextActive:  { color: C.text, fontWeight: '700' },

  // Section labels
  sectionRow:     { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10, marginTop: 4 },
  sectionDot:     { width: 7, height: 7, borderRadius: 4 },
  sectionLabel:   { color: C.textSub, fontSize: 11, fontWeight: '700', letterSpacing: 1, flex: 1 },
  sectionRight:   { color: C.textDim, fontSize: 11 },

  // Closed banner
  closedBanner:   { flexDirection: 'row', alignItems: 'center', gap: 8,
                    backgroundColor: '#1c2128', borderBottomWidth: 1,
                    borderBottomColor: C.border, paddingHorizontal: 16, paddingVertical: 8 },
  closedIcon:     { fontSize: 13 },
  closedText:     { color: C.textSub, fontSize: 12, fontWeight: '600' },

  // Error
  errorBanner:    { backgroundColor: '#2d1a1a', borderRadius: 8, padding: 12, marginBottom: 12,
                    borderWidth: 1, borderColor: C.red },
  errorText:      { color: C.red, fontSize: 13 },

  // Alert banner
  alertBanner:    { borderRadius: 12, padding: 14, marginBottom: 16, borderWidth: 1 },
  alertBannerTop: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  alertIconBox:   { width: 36, height: 36, borderRadius: 10, justifyContent: 'center', alignItems: 'center' },
  alertBannerTitle: { fontSize: 13, fontWeight: '600', color: C.text, flex: 1 },
  alertAllOpenNote: { fontSize: 11, marginTop: 3, opacity: 0.8 },
  openBtn:        { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 8 },
  openBtnText:    { color: '#fff', fontSize: 12, fontWeight: '800' },
  alertBtnRow:    { flexDirection: 'row', gap: 8, marginTop: 10 },
  alertSecondBtn: { flex: 1, paddingVertical: 8, borderRadius: 8, alignItems: 'center',
                    backgroundColor: C.surface2, borderWidth: 1, borderColor: C.border },
  alertSecondBtnText: { color: C.textSub, fontSize: 12, fontWeight: '600' },

  // Lane cards
  laneCard:       { borderRadius: 12, padding: 14, marginBottom: 10, borderWidth: 1 },
  laneCardTop:    { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10 },
  laneNumber:     { fontSize: 12, fontWeight: '700', color: C.textSub, letterSpacing: 0.5, flex: 0 },
  statusBadge:    { flexDirection: 'row', alignItems: 'center', gap: 5,
                    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 20, borderWidth: 1 },
  statusDot2:     { width: 5, height: 5, borderRadius: 3 },
  statusBadgeText:{ fontSize: 10, fontWeight: '700', letterSpacing: 0.5 },
  laneAvg:        { marginLeft: 'auto', fontSize: 13, fontWeight: '600' },
  laneCardMid:    { flexDirection: 'row', alignItems: 'center', gap: 14 },
  waitingNum:     { fontSize: 36, fontWeight: '800', lineHeight: 42, minWidth: 44 },
  waitingLabel:   { fontSize: 10, color: C.textDim, marginLeft: -10 },
  fillSection:    { flex: 1 },
  fillBarBg:      { height: 6, backgroundColor: C.border, borderRadius: 3, marginBottom: 5 },
  fillBarFg:      { height: 6, borderRadius: 3 },
  fillLabel:      { fontSize: 10, color: C.textDim },

  // Snapshot
  snapshotRow:    { flexDirection: 'row', gap: 10, marginBottom: 4 },
  snapshotCard:   { flex: 1, backgroundColor: C.surface, borderRadius: 12, padding: 16, borderWidth: 1 },
  snapshotVal:    { fontSize: 32, fontWeight: '800', color: C.text },
  snapshotLbl:    { fontSize: 10, color: C.textSub, fontWeight: '700', letterSpacing: 0.8, marginTop: 4 },
  snapshotSub:    { fontSize: 11, color: C.textDim, marginTop: 3 },

  // Forecast wait cards
  waitCardsRow:   { flexDirection: 'row', gap: 8, marginBottom: 16 },
  waitCard:       { flex: 1, backgroundColor: C.surface, borderRadius: 12, padding: 12,
                    borderWidth: 1, alignItems: 'flex-start' },
  waitCardLabel:  { fontSize: 9, fontWeight: '700', color: C.textSub, letterSpacing: 0.8, marginBottom: 4 },
  waitCardValue:  { fontSize: 22, fontWeight: '800', width: '100%' },
  waitCardUnit:   { fontSize: 10, color: C.textDim, marginTop: 2 },

  // Forecast detail cards
  recommendationBanner: { borderRadius: 10, borderWidth: 1, paddingHorizontal: 14, paddingVertical: 10, marginBottom: 12 },
  recommendationText:   { fontSize: 12, fontWeight: '600', lineHeight: 17 },
  detailRow:      { flexDirection: 'row', gap: 8, marginBottom: 14 },
  detailCard:     { flex: 1, backgroundColor: C.surface, borderRadius: 12, padding: 10,
                    borderWidth: 1, alignItems: 'flex-start' },
  detailLabel:    { fontSize: 8, fontWeight: '700', color: C.textSub, letterSpacing: 0.8, marginBottom: 4 },
  detailValue:    { fontSize: 16, fontWeight: '800', color: C.text, width: '100%' },
  detailSub:      { fontSize: 9, color: C.textDim, marginTop: 3 },

  // Chart
  chartCard:      { backgroundColor: C.surface, borderRadius: 12, borderWidth: 1,
                    marginBottom: 14, overflow: 'hidden' },
  chartLegend:    { flexDirection: 'row', alignItems: 'center', gap: 4 },
  legendDot:      { width: 7, height: 7, borderRadius: 4 },
  legendTxt:      { fontSize: 10, color: C.textDim },

  // Camera snapshots
  cameraRow:   { flexDirection: 'row', gap: 10, marginBottom: 4 },
  cameraCard:  { flex: 1, backgroundColor: C.surface, borderRadius: 12, borderWidth: 1, overflow: 'hidden' },
  cameraLabel: { fontSize: 9, fontWeight: '700', color: C.textSub, letterSpacing: 0.8, padding: 8, paddingBottom: 4 },
  cameraImage: { width: '100%', backgroundColor: C.surface2 },

  // Horizon toggle
  horizonToggle:        { flexDirection: 'row', gap: 4 },
  horizonBtn:           { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6,
                          backgroundColor: C.surface2, borderWidth: 1, borderColor: C.border },
  horizonBtnActive:     { backgroundColor: C.accent, borderColor: C.accent },
  horizonBtnText:       { fontSize: 10, fontWeight: '600', color: C.textSub },
  horizonBtnTextActive: { color: '#fff' },

  // Lane scenarios
  scenariosCard:  { backgroundColor: C.surface, borderRadius: 12, padding: 14, borderWidth: 1, marginBottom: 4 },
  scenarioRow:    { flexDirection: 'row', alignItems: 'center', gap: 10,
                    paddingVertical: 10, borderRadius: 8 },
  scenarioRowActive: { backgroundColor: C.surface2, paddingHorizontal: 8, marginHorizontal: -8 },
  scenarioLaneBox:   { width: 64, paddingVertical: 5, borderRadius: 8, alignItems: 'center',
                       backgroundColor: C.surface2 },
  scenarioLaneText:  { color: C.textSub, fontSize: 11, fontWeight: '600' },
  scenarioBarBg:     { flex: 1, height: 6, backgroundColor: C.border, borderRadius: 3 },
  scenarioBarFg:     { height: 6, borderRadius: 3 },
  scenarioRight:     { width: 56, alignItems: 'flex-end' },
  scenarioWait:      { fontSize: 14, fontWeight: '700' },
  scenarioOpen:      { fontSize: 9, color: C.accent, fontWeight: '700', letterSpacing: 0.5 },

  // Summary (Today)
  summaryCard:    { backgroundColor: C.surface, borderRadius: 12, borderWidth: 1,
                    overflow: 'hidden', marginBottom: 14 },
  summaryRow_:    { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
                    paddingVertical: 16, paddingHorizontal: 16 },
  summaryLbl:     { fontSize: 11, color: C.textSub, fontWeight: '600', letterSpacing: 0.8 },
  summaryValRow:  { flexDirection: 'row', alignItems: 'baseline' },
  summaryVal:     { fontSize: 18, fontWeight: '700', color: C.text },
  summaryValBig:  { fontSize: 26 },
  summaryUnit:    { fontSize: 12, color: C.textSub },
  summaryExtra:   { fontSize: 11, color: C.textSub },
  divider:        { height: 1, marginHorizontal: 0 },

  // Equipment
  equipCard:      { backgroundColor: C.surface, borderRadius: 12, borderWidth: 1,
                    overflow: 'hidden', marginBottom: 4 },
  equipRow:       { flexDirection: 'row', alignItems: 'center', gap: 12,
                    paddingVertical: 16, paddingHorizontal: 14 },
  equipIcon:      { width: 44, height: 44, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  equipLabelRow:  { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  equipLabel:     { color: C.text, fontSize: 14, fontWeight: '600' },
  equipPct:       { fontSize: 14, fontWeight: '700' },
  equipBarBg:     { height: 5, backgroundColor: C.border, borderRadius: 3, marginBottom: 5 },
  equipBarFg:     { height: 5, borderRadius: 3 },
  equipCount:     { color: C.textDim, fontSize: 11 },
  emptyText:      { padding: 20, textAlign: 'center', fontSize: 13 },

  // Gender donut
  genderRow:       { flexDirection: 'row', alignItems: 'center' },
  genderLegend:    { flex: 1, gap: 14, paddingLeft: 8 },
  genderLegendRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  genderDot:       { width: 10, height: 10, borderRadius: 5 },
  genderPct:       { fontSize: 20, fontWeight: '800' },
  genderLbl:       { fontSize: 12, fontWeight: '600', color: C.text },
  genderCount:     { fontSize: 11, color: C.textDim },

  // Demographics
  demoCard:       { backgroundColor: C.surface, borderRadius: 12, borderWidth: 1,
                    overflow: 'hidden', marginBottom: 4, padding: 14 },
  demoSubTitle:   { fontSize: 10, fontWeight: '700', color: C.textSub, letterSpacing: 1, marginBottom: 6 },
  demoSampleNote: { fontSize: 10, color: C.textDim, marginBottom: 10 },
  demoRow:        { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 3 },
  demoRowLabel:   { fontSize: 13, fontWeight: '600', color: C.text, width: 70 },
  demoBarWrap:    { flex: 1, height: 6, backgroundColor: C.border, borderRadius: 3 },
  demoBarFg:      { height: 6, borderRadius: 3 },
  demoPct:        { fontSize: 13, fontWeight: '700', width: 38, textAlign: 'right' },
  demoCount:      { fontSize: 11, color: C.textDim, marginBottom: 2, marginLeft: 78 },
  ageRow:         { flexDirection: 'row', gap: 10, justifyContent: 'center', marginTop: 4 },
  ageCard:        { flex: 1, alignItems: 'center', gap: 4 },
  agePct:         { fontSize: 14, fontWeight: '800' },
  ageBarBg:       { width: 28, height: 80, backgroundColor: C.border, borderRadius: 4,
                    justifyContent: 'flex-end', overflow: 'hidden' },
  ageBarFg:       { width: '100%', borderRadius: 4 },
  ageLabel:       { fontSize: 11, fontWeight: '600', color: C.text },
  ageCount:       { fontSize: 10, color: C.textDim },
});