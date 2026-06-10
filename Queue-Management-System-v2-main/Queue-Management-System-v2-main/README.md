# IQMS — Intelligent Queue Management System

A real-time queue management and forecasting system deployed in a retail environment. The system uses computer vision to detect customers at store entrances and checkout lanes, predicts wait times using a multi-model ML ensemble, and exposes the data through a mobile app and a live monitoring dashboard.

---

## System Overview

```
Camera feed (OWLv2 + tracking)
        │
        ▼
  PostgreSQL / TimescaleDB
  (service_events, entrance_events, queue_predictions)
        │
        ├──► Streamlit Dashboard   (real-time monitoring, lane control)
        │
        ├──► FastAPI Backend       (REST API)
        │           │
        │           ▼
        │    IQMSManager App      (React Native / Expo)
        │
        └──► ML Pipeline          (Prophet + LSTM + XGBoost ensemble)
```

---

## Components

### 1. FastAPI Backend (`api.py`)
REST API that bridges the database and the mobile app. All data the app displays is served from here.

**Endpoints:**
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Service health check |
| GET | `/live-lanes` | Real-time per-lane queue status |
| GET | `/forecast` | 15-min wait time forecast + lane scenarios |
| GET | `/forecast-chart` | 60-min time series for the in-app chart |
| GET | `/alerts` | Current alert level (OK / WARNING / URGENT / CRITICAL) |
| GET | `/day-recap` | Today's summary: customers, peak hour, equipment mix |
| POST | `/set-lanes` | Update active lane count (syncs with dashboard) |
| POST | `/alert-response` | Record staff response to an alert |

**Run locally:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

### 2. Streamlit Dashboard (`dashboard.py`)
Live monitoring dashboard used by store supervisors. Refreshes every 60 seconds automatically.

**Features:**
- Real-time queue depth and wait time display
- 15/30/45-min forecast with lane scenario comparison
- Lane selector (1–4 active lanes) — synced with the mobile app
- Day recap: customer volume, peak hours, equipment type breakdown
- Demographics and hourly entry charts
- Model component breakdown (Prophet / LSTM / XGBoost / Ensemble)
- Historical backtest and model accuracy metrics

**Run locally:**
```bash
streamlit run dashboard.py
```

**Environment variables:**
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=iqms
DB_USER=postgres
DB_PASSWORD=your_password
STORE_TZ=Europe/Paris
REFRESH_SEC=60
```

---

### 3. IQMSManager Mobile App (`master` branch)
Cross-platform mobile application built with React Native and Expo. Accessible remotely via a Cloudflare tunnel — no VPN or local network required.

**Screens:**
- **Live** — Per-lane queue status with fill indicators and wait times
- **Forecast** — 15-min wait predictions, 60-min chart, lane scenario selector
- **Today** — Daily recap with equipment mix, demographics, and hourly volume
- **Alerts** — Current alert level with staff response options (opening lane / false alarm)

**Lane sync:** Tapping a lane scenario in the app calls `POST /set-lanes` and updates the active lane count in both the app and the dashboard in real time.

**Run locally:**
```bash
npm install --legacy-peer-deps
npx expo start --web
```

**Remote access (Cloudflare tunnel):**
```bash
cloudflared tunnel --url http://localhost:8081
```

**Tech stack:**
- React Native + Expo SDK 54
- victory-native v35 (charts)
- react-native-svg

---

### 4. ML Forecast Pipeline
Ensemble model combining three independent forecasters to predict queue wait times.

| Model | Role | Strength |
|-------|------|----------|
| **Prophet** | Long-horizon trend + calendar seasonality | Daily/weekly patterns |
| **LSTM** | Recent sequence memory | Short-term momentum |
| **XGBoost** | Feature-based reactive prediction | Current queue state |
| **Ensemble** | Weighted combination of all three | Balanced accuracy |

**Key metrics (June 2026 evaluation):**
- 97.6% of predictions in OK state — near-zero actual wait
- ALERT state (2.2% of time): model underestimates by ~4.8× on average
- Correction factor for ALERT events: median **4.60×**, range 1×–10×

**Known limitation:** Model has a ~11-min baseline floor during empty-queue periods and underestimates severely during true congestion. A queue-depth-conditional correction factor is planned.

---

### 5. Data Analysis (`data-analysis` branch)
Standalone Python analysis of the production database exports.

**Script:** `generate_analysis.py`  
**Output:** 18 PNG charts saved to `analysis_output/`

**Charts generated:**
1. Dwell time distribution (full range + log scale)
2. Equipment type breakdown — service events
3. Dwell time by equipment type (boxplot)
4. Lane performance (volume + mean/median dwell)
5. Daily service volume over time
6. Hourly traffic patterns (service + entrance)
7. Day-of-week patterns (volume + avg dwell)
8. Model components over time (last 1000 predictions)
9. Model component correlation heatmap
10. Prediction status distribution (OK / ALERT / BUSY)
11. Forecast vs actual wait — by status
12. Correction factor distribution — ALERT events only
13. Model bias by status
14. Average predicted wait by hour of day
15. Equipment type — entrance events
16. Queue depth distribution
17. Has-bag distribution — entrance events
18. Dwell time percentiles (P10 → P99)

**Run:**
```bash
pip install pandas matplotlib seaborn numpy
python generate_analysis.py
```

---

## Database Schema (TimescaleDB)

| Table | Description |
|-------|-------------|
| `service_events` | One row per checkout interaction — lane, dwell time, equipment type |
| `entrance_events` | One row per camera detection at store entrance |
| `queue_predictions` | One row per forecast slot — Prophet/LSTM/XGBoost/ensemble outputs |
| `dashboard_state` | Single-row live state written by dashboard, read by API |
| `alert_responses` | Staff responses to alert events |

---

## Repository Branch Structure

| Branch | Contents |
|--------|----------|
| `main` | Full backend: pipeline, dashboard, API, camera integration |
| `lojaine/dev` | Active development branch — latest dashboard and API changes |
| `master` | IQMSManager mobile app (React Native / Expo) |
| `data-analysis` | Production CSV exports + analysis script + 18 PNG charts |

---

## Remote Access Setup (Production)

The production system runs on a dedicated machine with:
- PostgreSQL / TimescaleDB for data storage
- FastAPI served via uvicorn on port 8000
- ngrok tunnel exposing the API publicly
- Streamlit dashboard served on port 8501
- Cloudflare tunnel exposing the Expo web build publicly

The mobile app connects to the ngrok API URL defined in `App.js`:
```javascript
const API_URL = 'https://cheese-annuity-sulk.ngrok-free.dev';
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | PostgreSQL 15 + TimescaleDB |
| Backend API | FastAPI + psycopg2 |
| Dashboard | Streamlit |
| Mobile App | React Native + Expo |
| ML Models | Prophet, LSTM (TensorFlow), XGBoost |
| Camera AI | OWLv2 (object detection) + custom tracker |
| Tunneling | ngrok (API), Cloudflare Tunnel (app) |
| CI/Build | EAS Build (Expo Application Services) |

---

## Author

**Lojaine** — Internship project, 2026  
Supervisor: Arnaud
