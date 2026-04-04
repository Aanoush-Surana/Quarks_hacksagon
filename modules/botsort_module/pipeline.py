"""
pipeline.py
===========
End-to-end BoT-SORT tracking pipeline driven by a pre-computed segmentation
JSON document.

Workflow
--------
1.  Load segmentation JSON  (track_id = null on every detection)
2.  Optionally open the source video for real frame images (enables ReID)
3.  For every frame in the JSON, call BotSortTracker.update()
4.  Write assigned track_id back onto each DetectionRecord
5.  Save the updated JSON to disk


Programmatic usage
------------------
    from botsort_module.pipeline import run_tracking_pipeline
    doc = run_tracking_pipeline(
        input_path  = Path("output/segmentation.json"),
        output_path = Path("output/tracked.json"),
        video_path  = Path("Data/videoplayback.mp4"),
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from .io import load_json, save_json
from .schema import SegmentationDocument
from .tracker import BotSortTracker


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_tracking_pipeline(
    input_path:  Path,
    output_path: Path,
    video_path:  Optional[Path] = None,
    cfg_path:    Optional[Path] = None,
    fps:         Optional[float] = None,
    with_reid:   bool  = False,
    verbose:     bool  = True,
) -> SegmentationDocument:
    """
    Run BoT-SORT over a segmentation JSON and write back track IDs.

    Parameters
    ----------
    input_path  : Path to the input JSON  (track_id = null)
    output_path : Path to write the output JSON (track_id populated)
    video_path  : Optional video file.  When provided, the real BGR frame is
                  passed to the tracker — required for ReID mode and useful for
                  Kalman filter warm-up on the first few frames.
    cfg_path    : Optional BoT-SORT YAML config (uses bundled default if None)
    fps         : Frame rate override.  Auto-detected from video or metadata.
    with_reid   : Enable ReID appearance matching (needs video_path + weights)
    verbose     : Print progress to stdout

    Returns
    -------
    The mutated SegmentationDocument with track_ids filled in.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    # ── Load document ─────────────────────────────────────────────────────────
    doc = load_json(input_path)

    # ── Resolve FPS ───────────────────────────────────────────────────────────
    cap: Optional[cv2.VideoCapture] = None
    if video_path is not None:
        video_path = Path(video_path)
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if fps is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            print(f"  ⚠️  Video not found: {video_path}  (running without frames)")

    if fps is None:
        fps = float(doc.metadata.fps or 30.0)

    # Infer orig_shape from metadata for the detection stub
    orig_shape: tuple = (1080, 1920)   # sensible default
    if doc.metadata.resolution and len(doc.metadata.resolution) == 2:
        w, h = doc.metadata.resolution
        orig_shape = (int(h), int(w))  # (H, W) — OpenCV convention

    # ── Create tracker ────────────────────────────────────────────────────────
    tracker = BotSortTracker(
        cfg_path   = cfg_path,
        fps        = fps,
        with_reid  = with_reid,
    )

    if verbose:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  BoT-SORT Tracking Pipeline")
        print(sep)
        print(f"  Input     : {input_path}")
        print(f"  Output    : {output_path}")
        print(f"  Frames    : {doc.total_frames}")
        print(f"  Detections: {doc.total_detections}")
        print(f"  FPS       : {fps:.1f}")
        print(f"  ReID      : {'ON' if with_reid else 'OFF'}")
        print(f"  Video     : {video_path if cap else 'not provided'}")
        print(sep)

    # ── Frame loop ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    total_assigned = 0

    for frame_record in doc.frames:
        fid  = frame_record.frame_id
        dets = frame_record.detections

        # Fetch real frame image when video is available
        frame_img = None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame_img = cap.read()
            if not ret:
                frame_img = None

        # Reset all track_ids in this frame to None before assigning
        for det in dets:
            det.track_id = None

        # Update tracker → list of (det_idx, track_id)
        assigned = tracker.update(dets, frame_img=frame_img, orig_shape=orig_shape)

        for det_idx, track_id in assigned:
            dets[det_idx].track_id = track_id
            total_assigned += 1

        # Progress every 100 frames
        if verbose and fid % 100 == 0:
            n_tracks = len(assigned)
            print(
                f"    frame {fid:5d}/{doc.total_frames}"
                f"  |  dets: {len(dets):3d}"
                f"  |  tracked: {n_tracks:3d}"
            )

    if cap is not None:
        cap.release()

    elapsed = time.perf_counter() - t0

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(doc, output_path)

    if verbose:
        unique_tids = len(doc.unique_track_ids)
        print(f"\n  ✅ Done in {elapsed:.1f}s")
        print(f"     Detections assigned : {total_assigned:,}")
        print(f"     Unique track IDs    : {unique_tids}")
        print(f"     Output              : {output_path}")
        print("─" * 60)

    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Test Configuration ───────────────────────────────────────────────────
    # Replace these paths with your actual test files
    input_file = "path/to/your/segmentation.json"
    output_file = "path/to/save/tracked.json"
    video_file = "path/to/your/video.mp4" # Set to None to run without video
    
    print(f"Testing BoT-SORT pipeline locally...")
    run_tracking_pipeline(
        input_path  = Path(input_file),
        output_path = Path(output_file),
        video_path  = Path(video_file) if video_file else None,
        cfg_path    = None,
        fps         = None, # Will auto-detect from video or JSON
        with_reid   = False,
        verbose     = True,
    )


if __name__ == "__main__":
    main()
