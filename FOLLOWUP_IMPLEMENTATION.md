# Calibration, forecast history, mobile status and background statistics

Implemented in the active Queue-Management installation and the sibling IQMSManager mobile source on 27 September 2026.

## Calibration and arrival target

- Stored ensemble/model forecasts are now raw. The shared wait worker is the sole calibration application point, so turning its correction off leaves no independently calibrated stored ensemble.
- Raw weights are validated and normalized. Real arrival training and retrospective reconstruction now filter the configured entrance camera and the existing ten-second dwell threshold. Old mixed-camera model caches fail compatibility checks; a new model generation was trained and published successfully.
- Calibration defaults to off. A requested correction is applied only with a compatible schema-2 artifact: matching target, model generation and selected weights; declared timezone; finite bounded factors; expiry; and passing prospective validation. Missing model outputs also prevent correction. The old September 10 file is preserved but rejected.
- Hour lookup converts aware timestamps into the artifact timezone, including daylight-saving changes. Global and hourly factors are alternatives, not compounded corrections.
- Publications and the API record requested/applied state, artifact ID and rejection reason. The dashboard explains this distinction and the next-publication timing. A checked toggle cannot bypass validation. During live verification revision 9 requested calibration, but the worker correctly published **uncalibrated** arrivals and reported the legacy-artifact rejection.

## Prospective accuracy evaluation

`accuracy_eval.py` is now an import-safe CLI. It does not write the live calibration file, change settings, train models, or use mutable/backfilled queue predictions.

It reads immutable raw arrival vintages, selecting a forecast actually available before each target at a specified lead (default ten minutes, with a three-minute vintage tolerance). Input cutoffs must precede publication. It matches the configured camera/target and scores complete observed buckets, including zeros. Entrance-camera snapshot coverage requires a span of at least 150 seconds, at least six samples and no internal gap above 30 seconds; absent/poor coverage is excluded rather than assumed zero. This is a declared telemetry-coverage heuristic, not proof that every visitor was detected.

Fitting and holdout dates are disjoint. The candidate gate requires at least seven fitting days and seven later holdout days, at least 100 rows in each, and improvement in holdout MAE without worse absolute bias. Hourly coefficients require at least 30 rows across seven fitting days; otherwise the global factor applies. The measured holdout scores use the same hourly/global lookup as runtime.

Evaluation currently requires the same model generation. With the normal daily model refresh, a generation will generally not accumulate fourteen days; calibration therefore remains unavailable under this conservative gate. Validating a correction across retrained generations requires an explicit compatibility policy and separate evidence. This change does not silently assume that transfer is safe.

From the Queue-Management directory, using the active Windows virtualenv:

```powershell
.\Queue-Management-System-v2-main\Queue-Management-System-v2-main\venv\Scripts\python.exe .\Queue-Management-System-v2-main\Queue-Management-System-v2-main\accuracy_eval.py --report .runtime\accuracy.json
```

`--candidate-output PATH` writes a candidate only if the holdout gate passes; the live `calibration.json` is explicitly forbidden as an output. Generation of a candidate does not enable it. Current production evaluation reports **insufficient_data**, which is the correct result for the newly corrected target/model.

## Full wait-forecast retention and measured waits

Every successful shared publication now inserts an immutable `wait_forecast_runs` row in the same transaction as live state. It retains the full payload: all lane scenarios/slots, effective settings and revision, arrivals/dwell inputs, queue state, selected lanes at publication, model/run lineage and calibration status. A failed archive insert rolls back the live update.

Archive publication timestamps use `clock_timestamp()`, not transaction-start `NOW()`: dwell preparation can run within the transaction for tens of seconds. One initial row created before this correction was conservatively dated at repair time; its original timestamp is backed up in `.runtime/history_timestamp_repair.json`. No historical availability claim is inferred from that faulty earlier timestamp.

The evaluator can score independently supplied wait observations against matching retained lane scenarios and pre-observation vintages:

```text
observed_at,lanes,wait_minutes,measurement_source
2026-09-27T12:00:00+02:00,4,3.5,manual_queue_entry_to_service_start
```

Pass such a CSV using `--wait-observations PATH`. The timestamp must include a timezone; waits must exclude service duration. Matching uses the publication's selected lane count, the requested forecast lead, and a target-time tolerance of 90 seconds. Lane changes after publication are not reconstructed; observations must represent the matching staffing scenario. Without independent observations, the report explicitly marks wait accuracy unavailable. Service duration and a queue-derived proxy are not presented as measured waiting time.

## Mobile status

- The API exposes published/current settings revisions and `pending_settings` along with forecast/source freshness.
- IQMSManager displays current, unavailable, stale and pending conditions in English and French, with the last calculation time.
- Pending/stale/unavailable forecasts suppress staffing recommendations and disable lane-selection actions. HTTP/network failures retain an error indicator; cached data is not relabelled fresh during a retry. Requests have deadlines, and overlapping polls are prevented.
- Lane-change failures are shown instead of silently ignored.
- Verification covered the exported mobile web application, including a 390×844 viewport. No physical Android/iOS device build was run.

## Browser-independent statistics

The scheduled job refreshes entrance demographics and today's hourly traffic before model work. `dashboard_statistics` stores the summary with its own timestamp and camera; legacy dashboard-state JSON fields are updated for compatibility without changing forecast freshness. The dashboard no longer writes these summaries.

`GET /statistics/summary` returns the background summary and its own stale/age information. A statistics failure is logged and does not prevent the core forecast attempt. Summaries refresh on scheduled prediction jobs rather than on page views.

## Verification evidence

- 47 Python regression tests and six mobile status tests passed.
- Private PostgreSQL tables verified full archive/live-state atomicity, rollback on archive failure, preservation of current staffing, and rejection of obsolete settings.
- The actual evaluator SQL was tested with covered zero counts, wrong-camera events, late forecasts, future input leakage and missing/gapped coverage. Independent synthetic wait observations verified the wait-matching query.
- The corrected model retrain/publication succeeded. The first cycle took 193.59 seconds and skipped the missed schedule slot without overlapping; a following inference cycle published successfully.
- With port 8501 stopped, a scheduled job refreshed statistics at **09:12:52 UTC** and published shared forecast `c390df7231c24e14b4a6ddb623eee57e` at approximately **09:13:38 UTC**. Its archived payload matched the live/API result. Streamlit was restarted and its health check returned `ok`.
- The dashboard rendered 13 charts without a page exception. The mobile web export built successfully; browser-only simulated stale/pending responses displayed both warnings, disabled all four lane actions and removed staffing recommendations. French wording and the Live tab also rendered correctly, with no browser page errors.

Runtime evidence is in `.runtime/background_followup_evidence.json`, `.runtime/accuracy_after_fixes.json` and `.runtime/check_remaining_transactions.py`. Screenshots are in the Windows temporary directory as `rtsp-calibration-fixed.png` and `rtsp-mobile-status-fixed.png`.

Implementation is complete for these four follow-ups. Empirical accuracy improvement is **not** established: it requires enough suitable prospective forecasts and, for actual waiting times, independent observations.
