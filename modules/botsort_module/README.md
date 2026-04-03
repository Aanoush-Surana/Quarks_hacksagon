# botsort_module

Self-contained BoT-SORT tracking module for the **roadAI** project.

Operates on **pre-computed segmentation JSON** — no video inference inside this package.  
A separate segmentation step (e.g. `extract_trajectories.py` via YOLO) produces detections with `track_id: null`; this module fills those IDs in.

---

## Module Layout

```
botsort_module/
├── __init__.py       public API exports
├── schema.py         DetectionRecord / FrameRecord / SegmentationDocument dataclasses
├── tracker.py        BotSortTracker  — wraps ultralytics BotSort for pre-computed dets
├── io.py             load_json / save_json / validate_raw / make_empty_document
├── utils.py          iou_xyxy, to_native, bbox helpers, mask_to_polygon
├── pipeline.py       run_tracking_pipeline() + CLI (python -m botsort_module.pipeline)
└── config/
    └── botsort.yaml  bundled default BoT-SORT configuration
```

---

## JSON Contract

### Input — from segmentation step (`track_id` is always `null`)

```json
{
  "metadata": {
    "video":       "Data/videoplayback.mp4",
    "weights":     "krishSegmentationFinal/best.pt",
    "timestamp":   "2025-01-15T14:30:00",
    "fps":         30.0,
    "resolution":  [1920, 1080],
    "frame_count": 900
  },
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {
          "detection_id": 0,
          "class_id":     1,
          "class_name":   "car",
          "confidence":   0.87,
          "bbox_xyxy":    [206, 124, 295, 233],
          "bbox_xywh":    [250, 178, 89, 109],
          "mask_polygon": [[206,124], [295,124], [295,233], [206,233]],
          "mask_area_px": 9701,
          "track_id":     null
        }
      ]
    }
  ]
}
```

### Output — after BoT-SORT (`track_id` populated)

Same structure; `track_id` is now an `int` for every matched detection:

```json
{ "track_id": 7 }
```

Detections not matched by the tracker (below `track_low_thresh`) remain `null`.

---

## Quick Start

### CLI

```powershell
# Track a segmentation JSON (no video  — pure bbox tracking)
C:\Users\thang\miniconda3\python.exe -m botsort_module.pipeline `
    --input  output/segmentation.json `
    --output output/tracked.json

# With real frames (enables Kalman warm-up, optional ReID)
C:\Users\thang\miniconda3\python.exe -m botsort_module.pipeline `
    --input  output/segmentation.json `
    --output output/tracked.json `
    --video  Data/videoplayback.mp4 `
    --fps    30
```

### Programmatic

```python
from botsort_module import run_tracking_pipeline
from pathlib import Path

doc = run_tracking_pipeline(
    input_path  = Path("output/segmentation.json"),
    output_path = Path("output/tracked.json"),
    video_path  = Path("Data/videoplayback.mp4"),   # optional
)

# Inspect results
print(f"Unique track IDs: {doc.unique_track_ids}")

for frame in doc.frames:
    for det in frame.detections:
        print(frame.frame_id, det.track_id, det.class_name, det.bbox_xyxy)
```

### Build a document from scratch

```python
from botsort_module import BotSortTracker, make_empty_document
from botsort_module.schema import FrameRecord, DetectionRecord
from botsort_module.utils import bbox_xyxy_to_xywh

doc = make_empty_document(video="clip.mp4", fps=30.0)

for frame_id, raw_dets in enumerate(my_detections):
    frame = FrameRecord(frame_id=frame_id, detections=[
        DetectionRecord(
            detection_id = i,
            class_id     = d["class_id"],
            class_name   = d["class_name"],
            confidence   = d["conf"],
            bbox_xyxy    = d["xyxy"],
            bbox_xywh    = bbox_xyxy_to_xywh(d["xyxy"]),
            mask_polygon = d.get("polygon", []),
            mask_area_px = d.get("area", 0),
            track_id     = None,                  # ← always null at this stage
        )
        for i, d in enumerate(raw_dets)
    ])
    doc.frames.append(frame)

tracker = BotSortTracker(fps=30.0)

for frame in doc.frames:
    pairs = tracker.update(frame.detections)
    for det_idx, tid in pairs:
        frame.detections[det_idx].track_id = tid

from botsort_module import save_json
save_json(doc, "output/tracked.json")
```

---

## Integration with Existing Pipeline

`extract_trajectories.py` and `advanced_pipeline.py` are **unchanged** — this module is purely additive.

To wire the two together (produce a segmentation JSON *and* track it):

```python
# inside process_video() in extract_trajectories.py — add after temporal fusion
from botsort_module.schema import DetectionRecord, FrameRecord, SegmentationDocument
from botsort_module.io import save_json

# Build a SegmentationDocument from fused_outputs and save it
seg_frames.append(FrameRecord(
    frame_id   = fidx,
    detections = [
        DetectionRecord(
            detection_id = i,
            class_id     = fobj["class_id"],
            class_name   = fobj["class_name"],
            confidence   = fobj["confidence"],
            bbox_xyxy    = list(fobj["bbox"]),
            bbox_xywh    = bbox_xyxy_to_xywh(list(fobj["bbox"])),
            mask_polygon = [],          # fill from full_frame_mask if needed
            mask_area_px = 0,
            track_id     = tid,         # already assigned by model.track()
        )
        for i, (tid, fobj) in enumerate(fused_outputs.items())
    ]
))
```

---

## Configuration

Edit `botsort_module/config/botsort.yaml` or pass `cfg_overrides` to `BotSortTracker`:

```python
tracker = BotSortTracker(
    fps=30.0,
    cfg_overrides={
        "track_buffer":  60,     # keep lost tracks longer
        "match_thresh":  0.7,    # stricter IoU matching
        "track_high_thresh": 0.6,
    }
)
```

---

## Requirements

- `ultralytics >= 8.0`
- `torch` (CPU-only is fine)
- `opencv-python`
- `numpy`, `yaml` (PyYAML)

All are already installed in this project environment.
