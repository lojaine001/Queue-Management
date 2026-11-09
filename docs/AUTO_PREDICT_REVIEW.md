# Auto-predict review — running Windows installation

**Follow-up:** shared wait calculation now runs inside the background job too. Combined live cycles took 77 and 62 seconds; the dashboard/API read the same saved result even with Streamlit stopped. See [Background wait forecasting](BACKGROUND_WAIT_FORECAST.md). Earlier arrival-only timings below remain as the audit trail.

## Fixes applied and re-evaluation — 27 September 2026

The original findings below are retained as the audit trail. The active Windows installation now uses the dashboard scheduler as its only automatic owner; `IQMS_EnsemblePredict` is disabled, with its previous task definition backed up in `%TEMP%/IQMS_EnsemblePredict.before-single-owner.xml`. Existing forecasts were allowed to finish before the replacement scheduler started.

| Priority | Resolution | Re-evaluation |
|---|---|---|
| 1. Concurrent launchers / shared artifacts | OS-owned publisher lock acquired before ML imports; separate singleton scheduler lock; immutable model generations selected through an atomic `models/current.json` pointer. | A competing Windows predictor exits 75 without loading models. Duplicate scheduler rejected. Crash releases lock. Failed/incomplete model generation leaves the previous generation selected. |
| 2. Slow cadence / retraining each bucket | Cache compatibility uses source, training span, bootstrap setting and model age, rather than changing data fingerprints. Default refresh age remains 24 hours. Explicit `--mode infer`, `--mode train --train-only`, and `auto` modes. LSTM inference calls the model directly, and the training-fit export is rebuilt only during training. Scheduler uses monotonic start deadlines and skips missed slots. | Initial model refresh completed; subsequent complete live predictions ran in 19.31 and 20.38 seconds, including interpreter/ML startup. Actual consecutive starts were 180.007337 seconds apart. Tests cover age/config invalidation and missed deadlines. |
| 3. Stop leaves descendants | Stop requests are tied to the scheduler instance and allow active forecasts to finish. Timeout or scheduler exception terminates and verifies the owned process tree. Start remains disabled while stopping. | Native Windows tests cover active graceful completion and a timed-out launcher with a nested child. Real idle scheduler stopped, restarted, and rejected a duplicate. Stop was also requested during an actual scheduled inference: the forecast committed, all owned work exited, and status became stopped before restart. |
| 4. PID-only green status / missing diagnostics | Atomic state records PID plus creation time, arguments, heartbeat, active job, next deadline, last result, duration and success. Bounded job deadline (default 1800 seconds), retained per-run logs and rotating scheduler log. Manual prediction/backtest also have process-tree deadlines. | PID-reuse test passes. Live failures were shown as failures, with traceback and next retry; subsequent publication returned to waiting. |
| 5. UI differs from process configuration | Banner reads actual runtime state. Active interval/training span override browser query settings and are locked while running. CLI rejects nonpositive values. | Browser opened with `?sched_interval=60` correctly displayed the active 3-minute interval and 30-day span. Training/interval controls were disabled; Start was disabled. All 13 charts rendered, with no page exceptions or browser errors. |
| 6. Publication conceals age / overwrites vintages | After training, inference reloads current data and regenerates the horizon. Past targets are excluded from live publication. `forecast_runs` retains each publication's run ID, model version, run start, input cutoff, forecast origin, publication time and forecast values in the same transaction as the live table. UI shows input and publication age. | Actual PostgreSQL transaction test passed. First successful batch saved 49 future points; live/history counts match, and earliest target is later than publication. Failed transaction does not update the local success record. |

### Operating the scheduler

Use the dashboard's Start and Stop controls. Stop means **finish the active forecast and then exit**. The deadline remains effective while stopping. A 3-minute setting is a target between starts; overrunning work skips missed slots, with no overlapping catch-up jobs. A model refresh can still take longer than an inference run.

Runtime files live in the ignored `Queue-Management/.runtime/` folder:

- `scheduler.json`: actual configuration, ownership, heartbeat and result.
- `scheduler.log`: rotating scheduler events (2 MB × 4 files).
- `logs/<run-id>.log`: last 20 forecast run logs.
- `publication.json`: last committed forecast's provenance/freshness.
- `publisher.lock` / `scheduler.lock`: OS-owned lock files; do not delete them while processes run.

Models live under the active application's `models/versions/<version>/`. `models/current.json` selects a complete generation. Original flat model files and previous generations are preserved for rollback. `backtest_predict.py` resolves one generation per process, and the dashboard resolves the current training-fit curve.

For an explicit refresh, run the active virtualenv's Python with `ensemble_predict.py --source REAL --days 30 --mode train --train-only`. The same publisher lock applies, so an overlapping job exits 75. The automatic scheduler's `auto` mode refreshes expired/incompatible models and then publishes inference based on fresh inputs.

### Verification and remaining limits

20 focused tests passed using the actual Windows Python environment: scheduler ownership, crash release, PID identity, deadlines, descendant cleanup, graceful completion, model cache compatibility, atomic model publication, forecast publication/rollback and isolated dashboard LSTM behavior. An actual PostgreSQL transaction test used a private temporary table and rolled back its data.

Two integration regressions introduced during implementation were caught and corrected before a successful live publication: a misplaced timestamp filter in console formatting and a history-insert SQL placeholder mismatch. Regression tests now cover both paths. The pre-existing live forecast remained available throughout those failed transactions.

Final runtime check: scheduler PID **10604**, waiting, interval **3 minutes**, span **30 days**. Three successful retained publications were verified (49, 48, then 47 future rows as closing time approaches); the latest complete run took **20.33 seconds**. The independent Windows task remained disabled.

The obsolete `.scheduler.pid` record was removed; ownership now comes from `.runtime/scheduler.json` and OS locks.

This work fixes scheduler reliability and provenance; it does not establish forecast accuracy or a long-run p50/p95 latency guarantee. Retraining remains a serial maintenance phase; its duration can cause a skipped interval. Historical forecast vintages are retained from this change onward, not reconstructed for earlier overwritten predictions. Model versions/history currently have no automatic retention deletion. Scope is the active Auto-predict path, not every duplicate project directory or all forecasting algorithms.

---


Reviewed 27 September 2026, approximately 10:14–10:19 Europe/Paris.

## Scope and evidence

This review follows the user's request to focus on the running `Auto-predict running — every 3 min · PID 9332` feature. It is not a claim that every duplicate project directory has been audited. Runtime configuration, process ancestry, the active scheduler/dashboard/ensemble source, Windows Task Scheduler, recent ensemble logs and read-only database queries were checked. No production scheduler, task, camera or prediction process was stopped or reconfigured during this review.

Active root: `C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction\Queue-Management`.

| Component | Evidence |
|---|---|
| Dashboard scheduler launcher | PID 9332, running `run_scheduler.py --interval 3 --days 30` from the active root |
| Actual scheduler interpreter | PID 20116, child of 9332; Windows virtualenv launcher delegates to Python 3.10 |
| Forecast observed during review | Launcher PID 27140 → interpreter PID 23636, `ensemble_predict.py --source REAL --days 30` |
| Independent Windows task | `IQMS_EnsemblePredict`, enabled, repeats every 10 minutes, runs `Queue-Management/run_ensemble.bat` |
| Shared model/output destinations | Both paths run the same ensemble script, use its `models` directory and write `queue_predictions` |
| Database at 10:15:01 | Latest saved batch: 10:13:09; 58 surviving rows, prediction timestamps 10:06–12:57 |

The two Python layers per command are the Windows virtualenv launcher and its interpreter, not two independent schedulers. The Windows scheduled task is a genuinely separate scheduling mechanism. Its `IgnoreNew` setting prevents overlap with another instance of that task; it does not coordinate with the dashboard scheduler or the manual Run Prediction button.

## Findings, in priority order

### 1. High: multiple launchers can train and publish concurrently

`IQMS_EnsemblePredict` and PID 9332 are both enabled. The manual Run Prediction button is a third entry point. There is no common run lock in `run_scheduler.py` or `ensemble_predict.py`.

Relevant code:
- `run_ensemble.bat:4`
- `run_scheduler.py:38`
- `Queue-Management-System-v2-main/Queue-Management-System-v2-main/dashboard.py:2449`
- `Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py:609` writes the LSTM file, then its separate scaler and metadata files.
- The same script at line 1215 overwrites predictions on conflict using only `prediction_for`.

Impact: competing training consumes CPU/RAM; one run can read model artifacts while another writes them; a slower, older run can publish after a newer run. The database commit groups one run's rows transactionally, but does not prevent an older run from winning later. Actual artifact corruption was not observed; this is a confirmed unprotected concurrency path.

Recommendation: choose one scheduling owner, then put a process-wide or database advisory run lock at the predictor entry point so every launcher participates. Publish versioned model/scaler/metadata bundles atomically. Keep the previous successful bundle until replacement is complete.

### 2. High: “every 3 min” does not describe the real forecast cadence

`run_scheduler.py:71–82` runs prediction synchronously and starts the three-minute wait only after prediction returns. Start-to-start duration is therefore `prediction runtime + 180 seconds`.

The independent task's `ensemble_predict.log` records the same predictor taking 573.9 seconds in its most recent completed logged run, and 989.2 seconds in an earlier run that morning. These are task-run timings, not timings attributed to PID 9332: its output is discarded. A 573.9-second run through PID 9332 would produce a 12 minute 34 second start-to-start interval.

The training cache also requires an exact match of the changing training data fingerprint:
- Prophet: `ensemble_predict.py:452–456`
- LSTM: `ensemble_predict.py:562–567`
- XGBoost: `ensemble_predict.py:698–702`

Recent task logs repeatedly show full retraining, including 20 LSTM epochs. A nominal 24-hour model age limit does not prevent retraining when new data changes the fingerprint.

Recommendation: separate training from inference. Run short inference jobs on a three-minute schedule using the last successful model; retrain on a separately defined cadence or drift condition. Use monotonic deadlines and skip/coalesce missed runs rather than overlapping them. Until then, label the setting “pause after each run.”

### 3. High: Stop can leave a forecast running

`dashboard.py:696–710` terminates only the PID in `.scheduler.pid`, removes the PID file and returns success without waiting for the full process tree or checking termination errors.

Reproduction used extracted scheduler-control functions and disposable Python processes, not the production PIDs. A launcher → interpreter → forecast-launcher → forecast-interpreter tree was created. After `_stop_scheduler()` returned true, the two forecast descendants remained alive. The test then explicitly cleaned them up. A simpler launcher/interpreter-only test did terminate; the failure concerns the additional active forecast descendants.

Impact: the banner can say stopped while a forecast still trains or writes results. Pressing Start can add another run. Stopping this scheduler also does not disable `IQMS_EnsemblePredict`.

Recommendation: implement an explicit stop policy. Prefer stopping future scheduling and allowing the active run to finish; provide a separate cancel action that terminates and waits for the entire owned process tree. Remove the PID record only after confirming the intended state. Display active forecast PID separately.

### 4. High: the green banner is liveness, not health

`dashboard.py:654–666` checks only `psutil.pid_exists(pid)`; it does not verify the executable, command line, creation time, latest result or last successful database publication. PID reuse can therefore identify an unrelated process.

`dashboard.py:687–688` discards both stdout and stderr. `run_scheduler.py:40` has no forecast timeout. A stalled or repeatedly failing predictor can leave a green scheduler banner indefinitely. Recent database writes cannot be attributed to PID 9332 because the independent task writes the same table.

Recommendation: persist structured scheduler state with scheduler PID and creation time, active run ID/PID, actual arguments, last start/finish/success, exit code, duration, error summary, next due time and heartbeat. Derive Running, Waiting, Failed, Overdue and Stopped states from this record. Rotate logs and bound forecast execution time.

### 5. Medium: displayed settings may differ from actual process arguments

The status banner uses the current browser session's `sched_interval` (`dashboard.py:2522`), initialized from query parameters or a default of three minutes (`dashboard.py:117–125`). It does not read PID 9332's arguments. In this inspection, the interval really is three minutes, but that agreement is incidental.

The hint says the training span is locked while running, yet the Training data span widget (`dashboard.py:2462–2468`) is not disabled. Changing it does not change the already-running process's `--days 30`.

Recommendation: show actual scheduler settings from persisted runtime state. Separate “running configuration” from “next start configuration,” or restart deliberately when applying changes. Validate positive interval and day values in the CLI.

### 6. Medium: publication time hides old input and forecast horizons

At 10:15:01, the latest saved batch was timestamped 10:13:09, but its earliest prediction was for 10:06. The task log identifies that run as starting at 10:03:35 and finishing about 9.6 minutes later. Several forecast timestamps were already in the past when publication completed.

`ensemble_predict.py:1189` assigns `predicted_at` when saving. This accurately records publication time, but does not describe the age of the data or the forecast's original time origin. Long training before publication makes “newly saved” different from “based on current inputs.”

Recommendation: record `run_started_at`, `data_cutoff`, `forecast_origin`, `finished_at`, model version and run ID separately. Expose input age and last successful publication age in the UI. Preserve forecast runs for honest backtesting; the current upsert by `prediction_for` replaces previous vintages.

## Recommended implementation order

1. Establish one scheduling owner and a lock shared by all forecast entry points.
2. Add persistent state, rotating logs, success/failure reporting and a timeout.
3. Correct Stop/cancel behavior and verify it with real disposable Windows process trees.
4. Separate frequent inference from less frequent training; measure p50/p95 durations before promising a three-minute cadence.
5. Display actual runtime configuration and input/publication freshness.
6. Version model bundles and retain forecast vintages for evaluation.

## Focused regression checks

- Start twice simultaneously: exactly one scheduler owns the lock.
- Scheduled-task run overlaps manual run: only one publisher proceeds.
- Forecast fails: previous successful forecast stays available and status becomes failed/degraded.
- Forecast hangs: deadline terminates owned work and status becomes overdue/failed.
- Stop during an active Windows forecast: verify the chosen finish-or-cancel behavior; no accidental orphan or duplicate run.
- Open another browser or change query parameters: running interval and training span remain truthful.
- Run takes longer than three minutes: no overlap and no misleading next-run countdown.
- Kill the scheduler and reuse a PID: executable/creation-time identity check rejects the unrelated process.
- Older run finishes after a newer run: publication guard prevents regression.

## Limits

The source and operating-system evidence establish the issues above. A new complete training cycle was not launched for this review. Because PID 9332 sends output to the null device and lacks a per-run status record, its historical failures and durations cannot be reconstructed reliably from its own telemetry. The existing ensemble log belongs to the Windows task/batch launcher; database results combine all publishers.
