"""
schema.py
=========
Dataclasses representing the full segmentation + tracking JSON contract.

JSON Layout
-----------
{
  "metadata": { "video": "...", "weights": "...", "timestamp": "...", "fps": 30.0,
                "resolution": [1920, 1080], "frame_count": 900 },
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {
          "detection_id":  0,
          "class_id":      1,
          "class_name":    "car",
          "confidence":    0.87,
          "bbox_xyxy":     [x1, y1, x2, y2],
          "bbox_xywh":     [cx, cy, w, h],
          "mask_polygon":  [[x, y], ...],
          "mask_area_px":  1234,
          "track_id":      null        ← null before BoT-SORT; int after
        }
      ]
    }
  ]
}

Rules
-----
- `track_id` is always `null` in the raw segmentation output.
- After the BoT-SORT pass it is an `int` for detections that were matched to a
  track, or still `null` for low-confidence detections that were not assigned.
- All coordinate values are absolute pixel integers unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Top-level schema classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """
    One bounding-box / segmentation detection for a single object instance.

    Attributes
    ----------
    detection_id : int
        Zero-indexed position of this detection within its frame.
    class_id : int
        Integer class label from the YOLO model.
    class_name : str
        Human-readable class label (e.g. ``"car"``, ``"auto_rickshaw"``).
    confidence : float
        YOLO detection confidence in [0, 1].
    bbox_xyxy : list[float]
        Bounding box as ``[x1, y1, x2, y2]`` in absolute pixel coords.
    bbox_xywh : list[float]
        Bounding box as ``[cx, cy, w, h]`` in absolute pixel coords.
    mask_polygon : list[list[float]]
        Contour polygon as ``[[x, y], ...]``.  Empty list if no mask.
    mask_area_px : int
        Number of foreground pixels inside the mask.
    track_id : int or None
        ``None`` (JSON ``null``) before BoT-SORT runs.
        Set to a positive integer after the tracker assigns an identity.
    """

    detection_id:  int
    class_id:      int
    class_name:    str
    confidence:    float
    bbox_xyxy:     List[float]          # [x1, y1, x2, y2]
    bbox_xywh:     List[float]          # [cx, cy,  w,  h]
    mask_polygon:  List[List[float]]    # [[x,y], ...]
    mask_area_px:  int
    track_id:      Optional[int] = None  # null until BoT-SORT assigns one

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (all values are native Python)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionRecord":
        """Construct from a raw JSON dict. Missing keys get sensible defaults."""
        return cls(
            detection_id  = int(d["detection_id"]),
            class_id      = int(d["class_id"]),
            class_name    = str(d["class_name"]),
            confidence    = float(d["confidence"]),
            bbox_xyxy     = [float(v) for v in d["bbox_xyxy"]],
            bbox_xywh     = [float(v) for v in d["bbox_xywh"]],
            mask_polygon  = [[float(x), float(y)] for x, y in d.get("mask_polygon", [])],
            mask_area_px  = int(d.get("mask_area_px", 0)),
            track_id      = d.get("track_id"),  # preserve None (null) or existing int
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_tracked(self) -> bool:
        """True once a track_id has been assigned."""
        return self.track_id is not None

    @property
    def center(self):
        """``(cx, cy)`` computed from bbox_xyxy."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameRecord:
    """
    All detections for one video frame.

    Attributes
    ----------
    frame_id : int
        Zero-indexed frame number in the source video.
    detections : list[DetectionRecord]
        Ordered list of detections for this frame.
    """

    frame_id:   int
    detections: List[DetectionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id":   self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameRecord":
        return cls(
            frame_id   = int(d["frame_id"]),
            detections = [DetectionRecord.from_dict(det) for det in d.get("detections", [])],
        )

    @property
    def tracked_count(self) -> int:
        return sum(1 for det in self.detections if det.is_tracked)

    @property
    def has_any_track(self) -> bool:
        return any(det.is_tracked for det in self.detections)


@dataclass
class VideoMetadata:
    """
    Source video / model provenance information stored in the JSON header.

    Attributes
    ----------
    video : str
        Path or name of the source video file.
    weights : str
        Path to the YOLO model weights used for segmentation.
    timestamp : str
        ISO-8601 string recording when segmentation was run.
    fps : float or None
        Video frame rate (used to initialise the tracker).
    resolution : list[int] or None
        ``[width, height]`` of the source video.
    frame_count : int or None
        Total number of frames in the source video.
    """

    video:       str
    weights:     str
    timestamp:   str
    fps:         Optional[float]     = None
    resolution:  Optional[List[int]] = None   # [width, height]
    frame_count: Optional[int]       = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoMetadata":
        return cls(
            video       = str(d.get("video", "")),
            weights     = str(d.get("weights", "")),
            timestamp   = str(d.get("timestamp", "")),
            fps         = float(d["fps"])         if d.get("fps")         is not None else None,
            resolution  = [int(v) for v in d["resolution"]] if d.get("resolution") else None,
            frame_count = int(d["frame_count"])   if d.get("frame_count") is not None else None,
        )


@dataclass
class SegmentationDocument:
    """
    Complete JSON document: provenance metadata + per-frame detections.

    This is the root object that flows through the pipeline:
      1. Segmentation step writes it with ``track_id = null`` everywhere.
      2. BoT-SORT step reads it, runs the tracker, writes ``track_id = int``.

    Attributes
    ----------
    metadata : VideoMetadata
    frames   : list[FrameRecord]
    """

    metadata: VideoMetadata
    frames:   List[FrameRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "frames":   [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationDocument":
        return cls(
            metadata = VideoMetadata.from_dict(d.get("metadata", {})),
            frames   = [FrameRecord.from_dict(f) for f in d.get("frames", [])],
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def total_detections(self) -> int:
        return sum(len(f.detections) for f in self.frames)

    @property
    def total_tracked(self) -> int:
        """Detections that have been assigned a track_id."""
        return sum(det.is_tracked for f in self.frames for det in f.detections)

    @property
    def unique_track_ids(self):
        """Set of all unique track IDs across all frames."""
        return {
            det.track_id
            for f in self.frames
            for det in f.detections
            if det.track_id is not None
        }

    def reset_track_ids(self) -> None:
        """Set every detection's track_id back to None (useful for re-tracking)."""
        for frame in self.frames:
            for det in frame.detections:
                det.track_id = None
