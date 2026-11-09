# Background wait forecasts

Implemented and verified on 27 September 2026 in the active Windows installation.

## Behavior

The scheduler's prediction job now performs two phases under its existing ownership lock and deadline:

1. Generate and publish the arrival forecast, with its model version and retained forecast run.
2. Build the detailed wait forecast for every supported lane count and publish the shared result atomically.

`prediction/wait_forecast.py` performs the second phase without importing Streamlit. It reuses extracted dwell-model preparation from `prediction/dwell_models.py`, the isolated LSTM worker, and the existing queue simulator. The live queue and time origin are refreshed after model preparation, so that training time does not age those inputs before simulation. Expired forecast slots are removed.

The dashboard and the API consume the resulting `dashboard_state.forecast_json`. The dashboard no longer trains its forecast LSTM or publishes wait simulations. Its separate historical XGBoost chart and camera/statistics queries remain available. Browser-only demographic/traffic updates cannot change forecast values or freshness timestamps.

## Shared settings and concurrency

`forecast_settings` stores one validated configuration and revision for arrival-model selection, arrival calibration, smoothing, dwell-model mode, and dwell-model selection. Explicit dashboard control changes persist there; opening a browser or changing URL parameters does not replace that configuration. A pending configuration is identified in the dashboard until the next prediction run applies it.

A worker captures the revision before calculation and checks it under a database lock before publishing. If settings changed meanwhile, it discards the obsolete result and leaves the previous complete forecast available. It reads the current lane selection inside the publication transaction and uses that lane's already-computed scenario; an API lane change is not overwritten by the worker's earlier assumptions.

The existing scheduler Stop policy, full process-tree deadline, hidden Windows launches, logs and failure reporting also cover this phase. Its success status means that both arrival and shared-wait publication completed.

## Freshness and provenance

The shared payload and `.runtime/wait_publication.json` record:

- Background publisher identity, unique publication ID and settings revision.
- Calculation time, source arrival publication, checkout snapshot, service-event cutoff and entrance-data cutoff.
- Arrival run/model version, training span, effective configuration, dwell-model status and calculation duration.

The API reports each source's age separately. A recent calculation cannot hide an old arrival publication or camera snapshot: either being over ten minutes old marks the shared forecast stale. Event cutoffs are reported separately because no recent entrance or completed transaction can also mean the shop is quiet.

## Verification

- **38 tests passed** in the production Windows virtualenv, including scheduler ownership/deadlines, model publication, API consistency, missing/stale data, shared settings validation, revision rejection, per-lane simulation, source-age handling and a regression guard that prevents the dashboard from publishing shared waits or invoking forecast LSTM training.
- An actual scheduled run published while the Streamlit server was completely stopped: port 8501 had no listener. The API returned publication `994c08f867df4794b2c54e90b6b2c95c`, publisher `background`, a fresh 10-minute wait of **2.7 minutes**, and a checkout snapshot age of **1 second**. The dashboard was restarted after the check.
- Shared-wait preparation took **54.13 seconds** in that run. Complete scheduled cycles took **77.36 seconds** and **62.02 seconds**, both within the three-minute target.
- Reopening the dashboard preserved the publication ID and calculation timestamp. The browser displayed the same **2.7-minute** forecast, rendered **13 charts**, and reported no page exceptions or browser errors. A subsequent full dashboard refresh took **2.4 seconds**.
- PostgreSQL integration tests used private temporary tables to verify publication against the latest lane selection and rejection after a settings revision changed. Production staffing settings were preserved.

## Follow-up status

The four implementation follow-ups below have now been addressed; see [implementation and verification](FOLLOWUP_IMPLEMENTATION.md). Calibration has a single guarded application point, full wait vintages are retained transactionally, mobile freshness/pending indicators are implemented, and demographic/traffic summaries refresh in the background. Accuracy improvements still require prospective evidence and independently measured waits.

### Original follow-up list (historical)

- The mobile UI should visibly display the API's stale/pending information; those fields are available, but this change did not edit the mobile app.
- Evaluate prediction accuracy against observations. Arrival forecast vintages are retained in `forecast_runs`; retain the full shared-wait vintages too before evaluating the exact waits shown to users over time.
- Some ancillary demographic and traffic summaries still refresh through the dashboard. This change makes wait forecasting independent of the browser, not every statistics route.
- These timings are observed live runs, not a long-term latency guarantee. Periodic retraining or unusually slow work can still skip a schedule slot without overlap.
