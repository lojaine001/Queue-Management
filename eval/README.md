# Tracking evaluation (step 1: measure the current checkout pipeline)

Everything here runs on the production machine. Nothing in this folder is imported by
the live system; the only live-code change is the optional frame log (`frame_log_enabled`
in `Head-Detector/config2.yml`, off by default).

## Workflow

| # | Where | Command | Output |
|---|---|---|---|
| 1 | prod, store open | set `frame_log_enabled: true`, restart Head-Detector | `Head-Detector/LOGs/frame_log_<cam>_<date>.csv` |
| 2 | prod, store open | `.\eval\record_clip.ps1 -Camera checkout -Minutes 1 -Label sizetest` | MB/min estimate |
| 3 | prod, store open | `.\eval\record_clip.ps1 -Camera checkout -Minutes 20 -Label peak` | `eval/data/raw/*.mkv` + `.json` |
| 4 | prod | set `frame_log_enabled: false` again, restart | |
| 5 | prod, **after closing** (NPU is shared) | `python eval/detect_cache.py --video eval/data/raw/<clip>.mkv` (Head-Detector venv) | `eval/data/cache/<clip>.dets.npz` |
| 6 | anywhere | `python eval/track_replay.py --cache <npz> --run-name 000_baseline [--frame-log ... --sidecar ...]` | `eval/runs/000_baseline/<clip>/` |
| 7 | prod | annotate in CVAT, then `evaluate.py` (next step) | metrics |

Sanity check without footage: `python eval/tests/test_replay_synthetic.py`.

## Storage budget

Keep to ~3 clips: checkout quiet + peak (20 min each), entrance 15 min. Check the 1-minute
size test first; a sub-stream is usually a few MB per minute. Caches are a few MB per clip.
`record_clip.ps1` refuses to record below `-MinFreeGB` (default 3 GB).

## Privacy

- `eval/data/` and `eval/runs/` are git-ignored; footage never goes to GitHub.
- Videos (`.mkv`) stay on the production machine, and each sidecar has a `delete_after` date.
- `.dets.npz`, `tracks.csv`, `events.csv` and ground-truth exports contain only coordinates,
  times and IDs, with no images, so they are the files to copy elsewhere if you need to rerun replays.
- Record video only, no audio (the script drops it).

## Comparability rules

- `000_baseline` = production settings, unchanged. Every later run changes one key, via `--override`.
- Use the same `.dets.npz` for every tracker run; re-run `detect_cache.py` only for detector experiments.
- Use norfair 2.3.0 (production version); `track_replay.py` warns otherwise.
