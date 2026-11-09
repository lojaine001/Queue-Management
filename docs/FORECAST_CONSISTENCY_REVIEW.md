# Forecast consistency — fixes and verification

27 September 2026. Scope: the active Streamlit dashboard and FastAPI mobile backend under `Queue-Management-System-v2-main/Queue-Management-System-v2-main`.

## Confirmed problems

1. `/forecast` read the dashboard simulation while `/forecast/wait` and the future chart routes read the ensemble's different wait calculation. Initial live evidence: the 10-minute estimates were **4.8 minutes versus 0 minutes**.
2. Mobile lane scenarios were approximated as `current_wait × current_lanes / candidate_lanes`. Queue evolution is nonlinear, so this could claim every scenario has zero wait when the currently selected lane count clears the queue.
3. The dashboard's alternate-lane simulations reused the selected lane count's initial waiting backlog. Each candidate needs its own `max(total_people − candidate_lanes, 0)`.
4. `/set-lanes` changed the displayed lane count without changing its waits, and reset `updated_at` even though no new data had been processed. A dashboard calculation already in flight could overwrite that selection.
5. Cached prediction rows could include elapsed slots, and horizon labels were selected by row index rather than forecast timestamps. Missing `wait_0m` could also crash `/forecast` through `round(None, 1)`.

## Changes

- The dashboard publishes its actual per-lane wait series and horizon values in `dashboard_state.forecast_json`, in the same transaction as the existing scalar state.
- `/forecast`, `/forecast/wait`, `/forecast-chart`, and `/forecast-chart-3h` use that shared simulation. Lane scenarios are no longer fabricated by proportional scaling. Existing response fields remain, with calculation time and freshness metadata added.
- Each lane scenario has its own initial backlog. Horizon selection uses timestamps relative to the saved calculation time, consistently across the dashboard and API; expired cached rows are removed before simulation. Out-of-range horizons return an unavailable result, rather than silently clamping to the last prediction.
- The dashboard loads up to 60 forecast slots so the shared three-hour endpoint can extend beyond one hour, subject to available predictions and shop hours.
- Lane selection changes its 0/5/10/15-minute waits atomically under a row lock, without resetting the calculation time. Missing/stale simulations reject the change with HTTP 409. Dashboard publication checks the lane selection it originally read, preventing an older in-flight calculation from overwriting a newer API selection.
- Missing waits remain null. Invalid negative/nonfinite horizons return HTTP 422. Shared-state write failures produce a diagnostic rather than being silently ignored.

## Re-evaluation

- **9 focused Windows regression tests passed**, covering nonlinear lane scenarios, API horizon agreement, atomic lane changes, stale-data rejection, missing data, invalid horizons, timestamp offsets, timezone serialization, and per-lane backlog.
- Live HTTP verification on one shared snapshot: waits at 0/5/10/15 minutes were **3.5 / 4.6 / 8.0 / 11.8 minutes**, and both forecast endpoints agreed at every horizon. Timestamp matches were within half a three-minute bucket of the requested target.
- The browser displayed **8.0 minutes at +10 minutes**, matching the API. All **13 charts** rendered, with no page exceptions, browser errors, or shared-state publication warning.
- Future chart endpoints returned **20 and 45 points**, respectively, for the one-hour and up-to-three-hour views during verification.
- PostgreSQL integration checks exercised the actual lane-change SQL for 1–4 lanes and the guarded dashboard upsert, using a private temporary copy of `dashboard_state`. Wait values changed correctly, source age stayed unchanged, and a stale dashboard publication was rejected. Production staffing settings were not changed by these checks.

## Follow-up: background publication completed

The browser dependency and missing source-age tracking identified below have now been addressed. The scheduler publishes the shared wait simulation, the dashboard consumes it, and all model controls persist one revisioned configuration. See [Background wait forecast implementation and verification](BACKGROUND_WAIT_FORECAST.md) for the server-off integration test and 38 passing tests.

## Remaining priorities

1. Make the mobile UI display the API's freshness metadata visibly.
2. Retain full shared-wait vintages and evaluate forecast accuracy against matched observations. Arrival forecast vintages are already retained; consistency and successful refreshes do not establish predictive accuracy.
3. Review the remaining browser-driven ancillary statistics and duplicated service launchers.

Historical recap routes continue to read their historical database series. Shared forecast horizons are anchored to the returned calculation/reference timestamp rather than silently reinterpreted on every API request.
