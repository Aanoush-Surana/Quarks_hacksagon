"""
botsort_module
==============
Self-contained BoT-SORT tracking module that operates on pre-computed
segmentation JSON documents.

Public API
----------
    from botsort_module import (
        BotSortTracker,           # core tracker class
        SegmentationDocument,     # root data model
        FrameRecord,              # per-frame container
        DetectionRecord,          # per-detection record (track_id=null before tracking)
        VideoMetadata,            # JSON header
        load_json,                # Path → SegmentationDocument
        save_json,                # SegmentationDocument → Path
        make_empty_document,      # factory for building docs programmatically
        run_tracking_pipeline,    # full end-to-end function
    )

CLI
---
    python -m botsort_module.pipeline --input seg.json --output tracked.json

Module layout
-------------
    botsort_module/
    ├── __init__.py      ← you are here
    ├── schema.py        ← DetectionRecord / FrameRecord / SegmentationDocument
    ├── tracker.py       ← BotSortTracker (wraps ultralytics BotSort)
    ├── io.py            ← load_json / save_json / validate_raw
    ├── utils.py         ← iou_xyxy, to_native, bbox helpers
    ├── pipeline.py      ← run_tracking_pipeline + CLI main()
    └── config/
        └── botsort.yaml ← bundled default BoT-SORT configuration
"""

from .schema import (
    DetectionRecord,
    FrameRecord,
    SegmentationDocument,
    VideoMetadata,
)
from .tracker import BotSortTracker
from .io import load_json, save_json, make_empty_document
from .pipeline import run_tracking_pipeline

__all__ = [
    # Data model
    "DetectionRecord",
    "FrameRecord",
    "SegmentationDocument",
    "VideoMetadata",
    # Tracker
    "BotSortTracker",
    # I/O
    "load_json",
    "save_json",
    "make_empty_document",
    # Pipeline
    "run_tracking_pipeline",
]
