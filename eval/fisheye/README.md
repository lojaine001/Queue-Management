# Fisheye evaluation (step 2)

Tests person detectors on public overhead fisheye datasets that come with labels, so results are
measured automatically. Runs on any machine with the datasets; no camera or database needed.

## Data

WEPDTOF (16 real scenes, 8 of them shops) and CEPDOF (8 classroom videos), from the BU VIP lab.
Expected under `D:\data_new_system` (or pass `--root`, or set `FISHEYE_DATA`). See `datasets.py`
for the layout. Keep the data out of git.

**Licence:** both datasets (and the RAPiD code and weights) are for non-commercial research only.
Using them to compare models is fine; shipping a model trained on them needs a decision first.

## Steps

```powershell
# look at the labels
python eval\fisheye\view.py --dataset WEPDTOF --scene convenience_store

# today's detectors (one process per detector)
python eval\fisheye\run_detector.py --detector head     --model <path>\yolov9_s_discrete_headpose_post_0100_1x3x480x640.onnx
python eval\fisheye\run_detector.py --detector entrance --model <path>\yolov9t.onnx
python eval\fisheye\run_detector.py --detector head     --model ... --dataset CEPDOF --step 10

# RAPiD (top-view detector). Needs github.com/duanzhiihao/RAPiD cloned next to backend-v3 and
# weights/pL1_MWHB1024_Mar11_4000.ckpt from its v0.1 release. Runs on the GPU, full picture at 1024.
python eval\fisheye\run_rapid.py --dataset WEPDTOF
python eval\fisheye\run_rapid.py --dataset CEPDOF --step 10

# scores per scene
python eval\fisheye\score_detections.py --dets-dir eval\data\fisheye\dets\head\WEPDTOF

# see hits and misses
python eval\fisheye\view.py --dataset WEPDTOF --scene convenience_store --dets eval\data\fisheye\dets\head\WEPDTOF\convenience_store.npz
```

Outputs: detections in `eval/data/fisheye/dets/<detector>/<dataset>/`, scores in
`eval/runs/fisheye/<detector>/<dataset>/scores.csv|json`.

## What the scores mean

A detection counts for a labelled person when its box centre is inside that person's rotated
label box (one detection per person). Only people and detections inside the scene's ROI map count.

| Score | Meaning |
|---|---|
| recall | share of labelled people found |
| precision | share of detections that are a real person |
| count MAE | mean of \|detections - people\| per frame, i.e. how far off a head count would be |
| count bias | mean of (detections - people); negative means undercounting |
| exact | share of frames with exactly the right count |

## Comparability

- Same frames, same matching rule and the same ROI maps for every detector.
- Today's detectors see each picture resized to 640x480, as in the live loops.
- The laptop runs `yolov9_s_discrete_headpose_post_0100` on CPU. Production runs the `_nonms_` export
  of that model on the NPU with NMS done in Python, so boxes can differ slightly.
- Tune settings on CEPDOF and keep WEPDTOF for the final comparison.
