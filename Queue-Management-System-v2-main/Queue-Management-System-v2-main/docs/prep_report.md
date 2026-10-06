# Prep report

Read-only investigation. No app/API code changed. Two files created: this
report and `tools/measure_snapshot_usage.py`.

## Blocker, read first

**Items 3 and 4 could not be completed from this machine (lojai's).**

- The DB reachable from here (`localhost:5432`, db `iqms`) only has two
  tables: `entrance_events` and `queue_predictions`. It has no
  `queue_state_snapshots`, `service_events`, `dashboard_state`, or
  `push_subscriptions` — it is a small local/test database, not production.
  The real production Postgres instance lives on the production machine
  (`arnau`), per `DB_HOST` defaulting to `localhost` in every script — each
  script is meant to run *on* that machine, connecting to its own local DB.
- The only API URL hardcoded anywhere in the app (`IQMSManager/App.js:9`,
  a Cloudflare quick-tunnel address) is dead — quick tunnels change on every
  restart, and this one is stale. Nothing else in the repo points at a live
  instance. `localhost:8000` here returns nothing (API isn't running on
  this machine either).

So item 3's query and item 4's live measurement are both written and ready
below, but have not actually been run against real data. Two ways to
unblock, your call:
- run them on `arnau` directly (fastest — see commands below), or
- give me a reachable URL/tunnel for the production API and DB.

I did **not** substitute the local test DB's numbers or invent plausible
ones for either item — flagging the blocker instead, per the no-fabrication
rule from the earlier data-backfill conversations.

---

## 2. Backend endpoints the app uses

All six live in `api.py` (848 lines, `statistique-on-prod`, commit `9263af6`).

### `GET /live-lanes`
Reads latest `queue_state_snapshots` row for `CHECKOUT_CAM_ID`
(`Bosch_Camera_exit`) for per-lane counts, plus `dashboard_state` for
`avg_wait_min`. Response:
```json
{
  "lanes": [
    { "lane_number": 1, "lane_id": "0", "status": "open",
      "waiting": 2, "fill": 2, "fill_max": 10, "avg_wait_min": 3.5 }
    // one per lane, lanes 1-4
  ],
  "snapshot": { "total_in_queue": 7, "avg_wait_min": 3.5, "open_lanes": 2 }
}
```
`status` is one of `closed | open | busy | busy_high`, decided by
`_lane_status()` (api.py:68-75) from wait time and queue depth.

### `GET /forecast`
Reads `dashboard_state` joined to `forecast_settings`, runs it through
`scenario_for()`/`freshness()` (imported from `prediction.forecast_state`,
not defined in `api.py` itself). Response:
```json
{
  "wait_now_min": 2.1, "wait_5_min": 3.0, "wait_10_min": 4.2, "wait_15_min": 5.0,
  "current_lanes": 2, "lane_scenarios": { "1": {...}, "2": {...}, "3": {...}, "4": {...} },
  "queue_now": 7, "simulation_available": true,
  "stale": false, "updated_at": "2026-10-06T10:00:00+00:00"
}
```
(`stale`/`updated_at` come from `freshness()`, spread in — exact key set
depends on that function, not shown here since it lives outside `api.py`.)

### `GET /forecast-chart` (and `-3h`, `-12h`, `-2d` variants)
`/forecast-chart` and `/forecast-chart-3h` both call the same
`_shared_forecast_chart()` helper (api.py:617-630) with a 60 or 180 minute
horizon — forward-looking simulation data, from the shared forecast state,
not a DB query of its own beyond what `_forecast_state()` already does.
`-12h` and `-2d` are a different shape entirely: actual historical
`entrance_events` counts bucketed into 3-minute windows via Timescale's
`time_bucket()`, going backward from now:
```json
{ "slots": [ { "time": "14:03", "entries": 4 }, ... ] }
```
versus the forward-looking shape from `/forecast-chart`:
```json
{ "slots": [ { "time": "14:05", "prediction_for": "...", "arrivals": 3.2, "wait_min": 4.1 }, ... ],
  "current_lanes": 2, "stale": false, "updated_at": "..." }
```
Two genuinely different response shapes behind what looks like one family
of endpoints — worth knowing before writing a shared parser on the client.

### `GET /day-recap?date=YYYY-MM-DD`
By far the largest handler (api.py:340-614, ~11 separate queries per call):
daily total, vs-yesterday %, 7-day trend, peak hour, avg wait, lanes used,
busiest lane, alert-minutes, equipment mix, gender/age demographics, and
hourly entries. Full shape:
```json
{
  "date": "06 Oct 2026", "total_customers": 412, "vs_yesterday_pct": 8,
  "trend_7d": [{ "date": "2026-09-30", "count": 380 }, ...],
  "avg_wait_min": 4.2, "peak_hour": "17:00", "peak_hour_end": "18:00",
  "peak_count": 54, "peak_pct_of_total": 13,
  "equipment": [{ "type": "trolley", "label": "Trolley", "count": 120, "percent": 29, "color": "#06b6d4" }, ...],
  "lanes_today": 3, "busiest_lane": "1", "alert_minutes": 24,
  "demographics_gender": [...], "demographics_age": [...], "avg_age": 34.2,
  "entries_by_hour": [{ "hour": "09:00", "count": 12, "is_peak": false }, ...]
}
```
This is also the endpoint with every f-string-built SQL fragment in the
file — see item 5.

### `GET /snapshot/checkout` / `GET /snapshot/entrance`
Simplest handlers in the file (api.py:762-781). No DB involved at all —
reads a JPEG straight off disk (`snapshots/latest_checkout.jpg` /
`latest_entrance.jpg`, written by the detector process, not by `api.py`),
base64-encodes it inline, returns:
```json
{ "image": "data:image/jpeg;base64,<...>" }
```
or `{ "image": null }` if the file doesn't exist yet. No resizing,
no caching headers, no ETag — every poll re-reads and re-encodes the full
file from scratch. Relevant directly to items 3 and 4.

### `GET /alerts`
Reads `dashboard_state.wait_15m` only, thresholds it into
`red (>10) / orange (>7) / yellow (>5) / null`:
```json
{ "level": "yellow", "message": "Queue may reach 6 min — consider opening a lane.",
  "predicted_wait_min": 6.0, "horizon_min": 15 }
```
Fixed to the 15-minute horizon only — the per-device horizon picked in the
app's alert settings is a separate, client-side comparison against
`/forecast`, not something this endpoint knows about.

---

## 5. Security notes (flagged only, not fixed)

**Plaintext camera credentials, committed to git.**
`Head-Detector/config.yml` and
`Queue-Management-System-v2-main/Queue-Management-System-v2-main/config.yml`
are both tracked in the repo (`git ls-files` confirms it, last touched by
merge commit `e7cb2d6`) and contain real RTSP credentials in cleartext:
```yaml
username: service
password: '<redacted here — see config.yml itself for the real value>'
ip_address: '192.168.1.48:554/?inst=2'   # checkout camera
```
and the same password on the entrance camera's config
(`ip_address: '192.168.169.1:554/?inst=4'`). Anyone with read access to
the GitHub repo — including anyone the PAT used for pushing this session
was scoped to — can read both cameras' login and internal network address.
These are also now permanently in git history, not just the working tree;
rotating the password won't remove the old one from history.

**No authentication on any endpoint**, combined with `allow_origins=["*"]`
(api.py:41). Every GET is naturally read-only, but three POST endpoints are
wide open to anyone who has the URL:
- `POST /set-lanes` — changes the live lane count and the dashboard's
  current wait-time state for everyone, no auth check at all.
- `POST /push-subscribe` / `/push-unsubscribe` — anyone can register an
  arbitrary push endpoint to receive IQMS alerts, or delete someone else's
  subscription by guessing/reusing an endpoint URL.
- `POST /alert-response` — unauthenticated insert into `alert_responses`,
  trivially spammable.

None of this matters much while the only way in is a private tunnel URL
nobody else has — but that tunnel is explicitly meant to be reachable from
any device on the internet (per the original "put it online" request), so
the URL itself is the only thing standing between this and public write
access.

**SQL built with f-strings, `/day-recap` only** (api.py:368, 371, 406, 419,
429, 438, 450, 460, 474, 485, 503, 515) — eleven queries interpolate
`date_filter` / `alert_date_filter` directly into the SQL string instead of
using `%s` placeholders like every other query in the file does. Checked
whether this is actually exploitable: `ref_date` is always either a Python
`date` parsed via `datetime.strptime(date, "%Y-%m-%d")` (raises → clean 400
on anything that isn't a real date) or a `date` object read back out of
Postgres itself — never a raw string reaching the query. So **not
currently injectable** given the input is always one of those two, but it's
a real inconsistency against the parameterized-query pattern used
everywhere else in this same file, and the kind of thing that becomes
exploitable the moment someone adds a new code path that builds
`date_filter` differently without noticing this one isn't parameterized.

**`VAPID_PRIVATE_KEY_PATH` / DB password default.** Not a new finding, but
worth repeating here since it's adjacent: `DB_CONFIG` falls back to
`password=os.getenv("DB_PASSWORD", "0000")` (api.py:51) when `.env` isn't
loaded or the var isn't set — a hardcoded default production credential
baked into source, independent of the recent `load_dotenv()` fix.

---

## 3. Checkout camera_id blocking check — NOT RUN (see blocker above)

Ready to run on `arnau`:
```powershell
cd "C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction\Queue-Management\Queue-Management-System-v2-main\Queue-Management-System-v2-main"
.\venv\Scripts\activate
python -c "
import psycopg2, psycopg2.extras
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

print('--- queue_state_snapshots, distinct camera_id ---')
cur.execute(\"SELECT camera_id, COUNT(*) rows, MIN(timestamp) min_ts, MAX(timestamp) max_ts FROM queue_state_snapshots WHERE camera_id NOT LIKE 'SIM_%%' GROUP BY camera_id ORDER BY rows DESC\")
for r in cur.fetchall(): print(dict(r))

print('--- service_events, distinct camera_id ---')
cur.execute(\"SELECT camera_id, COUNT(*) rows, MIN(timestamp) min_ts, MAX(timestamp) max_ts FROM service_events WHERE camera_id NOT LIKE 'SIM_%%' GROUP BY camera_id ORDER BY rows DESC\")
for r in cur.fetchall(): print(dict(r))

print('--- Bosch_Camera_exit specifically ---')
for t in ('queue_state_snapshots', 'service_events'):
    cur.execute(f\"SELECT COUNT(*) rows, MIN(timestamp) min_ts, MAX(timestamp) max_ts FROM {t} WHERE camera_id = 'Bosch_Camera_exit'\")
    print(t, dict(cur.fetchone()))

print('--- service_events, any non-SIM rows in last 14 days ---')
cur.execute(\"SELECT COUNT(*) rows, MIN(timestamp) min_ts, MAX(timestamp) max_ts FROM service_events WHERE camera_id NOT LIKE 'SIM_%%' AND timestamp >= NOW() - INTERVAL '14 days'\")
print(dict(cur.fetchone()))
"
```
Context already confirmed from code (not the DB): `Head-Detector/config.yml`
has `camID: Bosch_Camera_exit` and is the config actually tracked in the
repo that matches `api.py`'s `CHECKOUT_CAM_ID` default — so the camera ID
names line up on paper. Whether it's actually writing rows is exactly what
this query answers, and that's still open.

## 4. Snapshot bandwidth measurement — NOT RUN (see blocker above)

`tools/measure_snapshot_usage.py` is written and syntax-checked, not yet
run against a live instance. On `arnau`, with the API already running on
port 8000:
```powershell
cd "C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction\Queue-Management\Queue-Management-System-v2-main\Queue-Management-System-v2-main"
.\venv\Scripts\activate
pip install requests pillow   # only if not already present
python tools\measure_snapshot_usage.py --url http://localhost:8000 --calls 20 --interval-s 5
```
The `--interval-s 5` default already matches the Live screen's real polling
interval (`IQMSManager/App.js:328`, `useApi([...], 5000)`). One thing to
know going in: that same call also polls `/snapshot/entrance` on the
identical 5-second interval, so whatever MB/hour figure this produces for
`/snapshot/checkout` alone is roughly half of what one open Live-screen
client actually costs in snapshot traffic.
