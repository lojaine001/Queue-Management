# Beta Archive

This folder holds archived or compatibility-only files that are not part of the active application paths.

Current contents:

- `compat/forecasting_shared.py`
  - old compatibility shim kept only for historical reference after the shared prediction code moved into `prediction/`

- `Queue-Management-System-v2-main/Queue-Management-System-v2-main/beta/pipeline_b/queue_management_v3.py`
  - older v3 pipeline variant archived after `queue_management_v2.py` remained the active entry point

Active code should prefer:

- `prediction/`
- `simulator/`
- `Head-Detector/`
- `Queue-Management-System-v2-main/Queue-Management-System-v2-main/queue_management_v2.py`
