# Arrival calibration audit — 27 September 2026

**Follow-up:** The findings below describe the pre-fix audit snapshot. Corrections, verification and remaining evidence requirements are documented in [FOLLOWUP_IMPLEMENTATION.md](FOLLOWUP_IMPLEMENTATION.md).

## Verdict

Multiplying an arrival forecast by an observed/forecast ratio can correct a persistent aggregate bias. The present implementation does **not** establish that its correction improves future forecasts. It also applies the hourly factors in the wrong timezone. Treat the current coefficients as unvalidated, not an accuracy improvement guarantee.

This audit made no changes to live settings, calibration coefficients, model files, services, or production data. The shared setting was enabled at revision 7 when checked. Do not rerun `accuracy_eval.py` merely to inspect results: it writes the live calibration file automatically.

## What the switch actually does

`prediction/wait_forecast.py:23–42` blends the selected raw Prophet/LSTM/XGBoost columns using normalized weights. When enabled, it multiplies that blend by an hourly coefficient, falling back to the global coefficient. The observed-rate cap and smoothing run afterward. The resulting arrivals feed the background queue simulation; the dashboard and API read its published results. A settings change takes effect in a subsequent successful background publication.

This path applies calibration **once**. It does not use the already calibrated `queue_predictions.ensemble_yhat` column. An isolated call to the actual function, with raw model values 10/20/30 and weights 0.4/0.3/0.3, returned 19 with calibration off and 25.5132 with calibration on at 10:00 local time. Setting `ensemble_yhat` to 9999 did not change either result.

There is a separate application in `ensemble_predict.py:811–821`: the publisher always calibrates its stored ensemble, independently of this switch. Turning the switch off therefore disables correction for the shared wait forecast but does not disable correction everywhere in the project.

The file was generated on 10 September. Its global factor is 1.348 (+34.8%); its hourly factors range from 1.1448 to 1.4837 (+14.5% to +48.4%). The hourly factor replaces the global factor rather than multiplying it again. A percentage change in arrivals does not imply the same percentage change in waiting time, which also depends on capacity and backlog.

## Findings, in correction order

### 1. Hourly fitting and application use different timezones

`accuracy_eval.py:126–153` explicitly converts target timestamps to UTC, then groups by `.dt.hour`. The worker converts timestamps to `STORE_TZ` before `prepare_arrivals` uses `.hour` (`wait_forecast.py:85–87,35`). The ensemble publisher also uses local forecast hours.

For example, 10:00 in Paris on 27 September is 08:00 UTC. The current code uses key `10` (1.3428), whereas the fitted coefficient for that time of day is key `8` (1.233). The discrepancy changes with daylight saving time. The calibration file contains no timezone declaration.

Correction: choose an explicit fitting/application timezone, store it in a versioned artifact, and use aware timestamps consistently. A fixed two-hour shift is not a general solution. Correcting timezone alone would not resolve the validity problems below.

### 2. Re-fitting consumes calibrated forecasts but saves an absolute raw-model multiplier

The evaluator reads `ensemble_yhat` (`accuracy_eval.py:94`), which the producer has already multiplied. `_write_calibration_json` then writes `actual_sum / predicted_sum` as a replacement factor. The next prediction applies this residual ratio to the raw models, losing the previous correction.

An isolated execution of the real writer function, redirected to a temporary file, demonstrated the problem: raw forecasts of 10 and actuals of 15 produce k=1.5; evaluating the corrected forecasts of 15 produces k=1.0, which overwrites 1.5. Subsequent raw forecasts revert to 10. This is a re-fitting feedback problem, not double multiplication in the shared worker.

Correction: fit absolute coefficients from retained **raw** model forecasts with known weights and model versions. Separate evaluation from explicit publication of a calibration artifact; an evaluation command should not silently replace the live correction.

### 3. The evaluation population is unsuitable for claiming prospective accuracy

The SQL in `accuracy_eval.py:82–120`:

- Selects the latest mutable prediction per target bucket, without requiring it to have been available before that bucket.
- Has no source filter to exclude reconstructed backtests. `backtest_predict.py` writes historical predictions using loaded models and a different two-model blend; these are not equivalent to live three-model forecasts.
- Inner-joins to event counts, excluding buckets with no qualifying events rather than scoring a known zero. Missing telemetry must also be distinguished from an observed zero.
- Counts all cameras despite defining `CAMERA_ID`; it has no camera or simulator filter. The live producer excludes simulator cameras but also includes all other cameras. The intended target population needs to be made explicit before calibrating it.

The saved calibration-era CSV contains 2,525 rows for 27 August–9 September, **390 nonpositive forecast lead times (15.4%)**, and **no zero-arrival buckets**. The current 14-day database slice contains 2,445 past prediction rows, including 1,074 with `predicted_at >= prediction_for`.

In the current 14-day event slice eligible under the evaluator's dwell threshold, counts are 4,476 for `Bosch_Camera_Entrance` and 28,356 for `Bosch_Camera_exit`. These counts establish that the target is not entrance-camera-only; they do not establish whether individual customers are duplicated. A multiplier cannot resolve an incorrectly defined event population.

Correction: evaluate immutable forecast vintages issued before the target bucket, at explicit lead times, against a consistent camera/source/dwell definition. Include complete, observed zero buckets and exclude periods of missing coverage and incomplete buckets. Keep retrospective gap-filling out of prospective accuracy evaluation.

### 4. The claimed benefit is in-sample and does not evaluate the deployed hourly correction

`accuracy_eval.py:138–146` fits the global ratio on every evaluated row and reports corrected error on those same rows. With k = sum(actual) / sum(predicted), corrected aggregate bias is zero by construction. It is not independent evidence of prediction skill.

The saved CSV shows MAE falling from 3.9799 to 3.7202 arrivals per three-minute bucket, about 6.5%. Those are same-data results for the **global** factor. The file deployed afterward contains separate hourly factors, whose actual runtime behavior is not what that reported corrected MAE measures.

Correction: fit on an earlier calibration window, freeze the correction, and compare raw/global/hourly alternatives on a later untouched window. Report errors and bias at the horizons the application uses. Use multiple days and sufficient observations per hour; the current threshold of five rows does not establish stability. Reassess after model changes. Arrival error alone also does not validate waiting-time accuracy.

### 5. Coefficients have no compatibility or validity checks

The file lacks model version, selected models/weights, camera/source definition, fitting period, sample counts, holdout results, and timezone. The worker applies the same coefficients after changing the selected models or retraining them. The artifact predates the active model generation, with no evidence that the correction transfers to that generation.

Neither runtime path checks that `bucket_minutes` matches its own bucket size or that factors are finite and nonnegative. Parsed JSON with invalid numeric values can fail the worker or contaminate predictions. The flag `calibration_applied` records that a file was used, not that it passed validation; publications do not retain a calibration artifact identifier.

Correction: use a validated, versioned artifact tied to the forecasting configuration. Record the artifact identity in published results. Refuse incompatible/invalid corrections and expose the reason with the actual applied status.

### 6. The UI explanation and debug panel need repair

The toggle help says only that it multiplies by a learned k, without explaining publication timing, validation status, or its limited scope. `_arrival_calib_debug_rows` is set to an empty list and never populated (`dashboard.py:1808`), so the debug expander reports no arrival forecast even when one exists. Its unused explanatory branches still describe a saved-ensemble path that the shared worker no longer follows (`dashboard.py:2918–2938`). The adjacent trend note also still says computation happens in the dashboard.

Correction: show the actual published settings and applied artifact, fit date/timezone, factor range, and validation state. Distinguish pending settings from the last published result. Explain that the background worker blends raw model forecasts, optionally corrects arrivals, and then simulates waits.

## Independent current-data check

At 08:59:54 UTC, retained history contained only ten runs, all from one model generation, starting at 08:32:55 UTC that morning. A read-only probe selected one vintage per completed target bucket, published at least nine minutes beforehand, with an earlier data cutoff. Actuals used the current producer's non-simulator/dwell-filter population and a left join retaining zeros.

Only **five** targets qualified, 10:42–10:54 Paris time:

| Arrival prediction | MAE per bucket | Mean prediction minus actual |
| --- | ---: | ---: |
| Raw weighted blend | 3.1214 | +1.1206 |
| Current local-hour calibration | 4.8770 | +4.2471 |
| Same artifact indexed by UTC hour | 4.3147 | +3.2457 |

This probe isolates the multiplier before the shared worker's cap/smoothing. It is not an evaluation of final waiting times. It shows that calibration worsened this small sample; it is far too short to establish sustained performance or select a replacement coefficient. In particular, correcting the timezone is not sufficient evidence to enable this artifact.

## Recommended decision

Do not treat the existing enabled setting as a validated accuracy feature. Prefer uncalibrated raw forecasts as the comparison baseline while rebuilding the evaluator and gathering adequate prospective evidence. Fix the target definition, time alignment, raw-denominator fitting, and holdout evaluation before selecting and publishing replacement coefficients. Do not merely regenerate the existing file with the current evaluator.

Read-only SQL results and isolated-function probes are saved locally in `.runtime/calibration_audit_evidence.json`; the reproducible script is `.runtime/audit_arrival_calibration.py`. It never imports the evaluator's executable module, executes its writer only against a temporary file, and opens the production database in read-only mode.
