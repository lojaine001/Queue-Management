# IQMS: Intelligent Queue Management System

Counts customers at the store entrance and in the checkout lanes from two cameras, measures
waiting and service times, forecasts arrivals and waiting times, and shows everything on a
dashboard, a web/phone app and push alerts.

| Component | Folder | Runs as |
|---|---|---|
| Checkout pipeline (head detection + tracking per lane) | `Head-Detector/` | `main.py` |
| Entrance pipeline (person counting, age/gender) | `Queue-Management-System-v2-main/Queue-Management-System-v2-main/` | `queue_management_v2.py` |
| API (port 8000), operator dashboard (port 8501), push alerts, forecast | same folder | `api.py`, `dashboard.py`, `push_alert_checker.py`, `ensemble_predict.py` |
| Forecast scheduler (every 3 min) and shared forecasting code | `run_scheduler.py`, `prediction/` | run with the entrance app's venv |
| Web / phone app (port 3001, public through ngrok) | `IQMS-Web/` | React, built with Vite, served by Caddy |
| Simulator (development only) | `simulator/`, `run_simulator.py` | own venv |
| Evaluation tools | `eval/` | see `eval/README.md` |

More documentation: `docs/` (developer guide, user guide, design reviews).

## 1. Install on a new machine

### Software to install first
| Software | Version used in production | Notes |
|---|---|---|
| Windows 11 | | Intel Core Ultra with NPU for the cameras (see "No NPU" below) |
| Python | **3.10** (3.10.6), 64-bit | python.org installer. Pinned packages need 3.10. |
| PostgreSQL | 17 | with the **TimescaleDB** extension |
| Node.js + npm | 26.x | for the web app |
| ngrok | 3.x | only for public access to the web app |
| Caddy | 2.11.7 | web server for the built web app; `install.ps1` downloads it into `tools\bin\` |
| Git | any | |
| Intel NPU driver | latest | only on machines with an NPU |

### Steps
```powershell
git clone <repository url> Queue-Management
cd Queue-Management

# first run: creates the venvs, installs packages, the web app and Caddy, and creates .env
# with a random IQMS_API_KEY (see "Security")
.\tools\install.ps1

# edit .env: at least DB_PASSWORD (the PostgreSQL password); see .env.example for every setting
notepad .env

# copy the two detector models into place (see "Models" below), then the second run:
# creates the database, checks the models, registers the daily check
.\tools\install.ps1 -RegisterTasks
```
`install.ps1` can be run again at any time; it skips what is already done and ends with a list
of anything left to do. `-DryRun` shows the steps without doing them. `-Exact` installs the
complete production package sets (`requirements.lock.txt`) instead of only what the code imports.

### Cameras
Each pipeline reads its camera from its `config.yml`: `camID`, `ip_address`, `username`,
`password`, plus the lane polygons (ROIs). `Head-Detector/config.yml` is the checkout camera,
`Queue-Management-System-v2-main/Queue-Management-System-v2-main/config.yml` the entrance camera.
Runtime options (tracker, OpenVINO device, logging) are in each folder's `config2.yml`.

### Models
Not in git. `tools/models.json` lists them:

| Model | Used by | Where it goes |
|---|---|---|
| `yolov9_s_discrete_headpose_nonms_1x3x480x640.onnx` | checkout pipeline | `Head-Detector/models/` |
| `yolov9t.onnx` | entrance pipeline | `Queue-Management-System-v2-main/Queue-Management-System-v2-main/models/` |
| `retinaface_mnet_v2.onnx`, `age_gender.onnx` | entrance faces | downloaded automatically on first start (uniface releases on GitHub) |

Copy the two detector models from an existing installation into those folders before starting.
Optional: if they are kept in a Hugging Face model repo, `install.ps1 -ModelsRepo <account>/<repo>`
downloads missing ones (set `HF_TOKEN` for a private repo; SHA256 values in `tools/models.json`
are checked when filled in).

### Database
`install.ps1` creates the database named in `.env` (`DB_NAME`, default `iqms`). The apps enable
TimescaleDB and create their tables on first start.

### No NPU
Set `openvino_device: CPU` and `openvino_precision: FP32` in both `config2.yml` files.

### ngrok
```powershell
ngrok config add-authtoken <token>
$env:NGROK_DOMAIN = "<your reserved domain>.ngrok-free.dev"   # or set it as a user environment variable
```
Without ngrok, start with `-NoTunnel`.

## 2. Run

| Command | What it does |
|---|---|
| `.\start_all.ps1` | checks that `.env` has `IQMS_API_KEY` and Caddy is installed (if not, it stops before touching anything), stops anything left over, then starts API, scheduler, push alerts, dashboard, web app (build + Caddy), ngrok and both cameras, and prints a status overview |
| `.\start_all.ps1 -NoViewImg` | same, cameras without preview windows (output to `logs\`) |
| `.\start_all.ps1 -SkipCameras` / `-NoTunnel` | without the cameras / without ngrok |
| `.\stop_all.ps1` | stops everything |
| `.\status_all.ps1` | status overview |
| `.\run_cameras.ps1` / `.\stop_cameras.ps1` | restart or stop only the two cameras |
| `.\tools\regression_test.ps1` | checks scripts, tests, build and the running services (`-StaticOnly` skips the live part) |

The `.bat` files do the same for double-clicking.

| Address | |
|---|---|
| http://localhost:3001 | web / phone app (Caddy: reachable only from this machine and through ngrok) |
| http://localhost:8501 | operator dashboard |
| http://localhost:8000/health | API health |

## 3. Security

**What is public.** The ngrok URL forwards to Caddy on port 3001: the built web app, and `/api/*`
forwarded to the API on port 8000. Every read-only (GET) endpoint can be reached by anyone with the
URL, including the camera snapshots (`/snapshot/checkout`, `/snapshot/entrance`) and the data export
(`/export`).

**What needs the key.** The endpoints that change data need the header `X-API-Key: <IQMS_API_KEY>`.
Without it the API answers 401 and logs the attempt (`logs\api_err.log`).

| Endpoint | Called by | How the key is sent |
|---|---|---|
| `POST /push-subscribe`, `POST /push-unsubscribe` | web app (alert subscriptions) | Caddy adds it (`IQMS-Web/Caddyfile`); it is never in the browser |
| `POST /set-lanes`, `POST /alert-response` | IQMSManager staff app, scripts | the caller must send it |

Because Caddy adds the key to the two push calls, anyone with the URL can still subscribe to or
unsubscribe from alerts through the web app; that is the feature. `/set-lanes` and
`/alert-response` are refused without the key, and so is every POST sent straight to port 8000.

**Setting secrets.** Secrets go in `.env`, which is not in git (template: `.env.example`):
- `IQMS_API_KEY`: required, the API refuses to start without it. `install.ps1` adds a random one.
  To make a new one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`, put it in
  `.env`, then run `.\start_all.ps1` (the API and Caddy read it when they start).
- `DB_PASSWORD`, `CHECKOUT_CAM_PASSWORD`, `VAPID_PRIVATE_KEY_PATH`.
- The camera logins are in each `config.yml`.

Never put the key in the web app's code or in git.

**Never expose:**
- the Vite dev server (`npm run dev`): development only, listens on `localhost`, never tunnel it;
- port 8000 (API), 8501 (dashboard: it can change forecast settings and has no login) and 5432
  (PostgreSQL; `pg_hba.conf` only accepts logins from this machine). Only Caddy on 3001 goes
  through ngrok;
- `.env`, the VAPID private key file and the camera passwords.

## 4. Logs and disk space

- `Head-Detector/LOGs/`, `.../Queue-Management-System-v2-main/LOGs/`: camera app logs, rotated at
  50 MB with 3 old copies (`LOG_MAX_MB`, `LOG_BACKUP_COUNT` in `.env`).
- `logs/`: output of the services started by `start_all.ps1`; the previous run is kept as `*.prev.log`.
- `.runtime/`: scheduler state and the last forecast run logs.
- `Head-Detector/ov_cache/`: OpenVINO compile cache, rebuilt automatically if deleted.

## 5. Tests

```powershell
.\tools\regression_test.ps1 -StaticOnly
```
or only the unit tests, with the entrance app's venv:
```powershell
$py = ".\Queue-Management-System-v2-main\Queue-Management-System-v2-main\venv\Scripts\python.exe"
& $py -m unittest discover -s prediction -t . -p "test_*.py"
& $py -m unittest discover -s tests -t . -p "test_*.py"
```
`tests/test_integration_ensemble.py` needs a database with forecast data. `tests/test_api_auth.py`
checks the API key without a database or a real key.
