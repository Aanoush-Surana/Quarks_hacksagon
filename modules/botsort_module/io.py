"""
io.py
=====
JSON loading, saving, and validation for the segmentation + tracking document.

Public functions
----------------
load_json(path)         → SegmentationDocument
save_json(doc, path)    → None
validate_raw(raw_dict)  → None  (raises ValueError on schema violations)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schema import SegmentationDocument
from .utils import to_native


# ---------------------------------------------------------------------------
# Required top-level keys and their expected types
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS: Dict[str, type] = {
    "metadata": dict,
    "frames":   list,
}

_REQUIRED_METADATA_KEYS = {"video", "weights", "timestamp"}

_REQUIRED_DETECTION_KEYS = {
    "detection_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_xyxy",
    "bbox_xywh",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_raw(raw: Any, *, strict: bool = False) -> None:
    """
    Validate the structure of a raw JSON dict against the schema.

    Parameters
    ----------
    raw    : Any   — the parsed JSON object
    strict : bool  — if True, also validate every detection in every frame
                     (slower but catches per-detection key errors)

    Raises
    ------
    ValueError with a descriptive message if any check fails.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object at root, got {type(raw).__name__}")

    # Top-level keys
    for key, expected_type in _REQUIRED_TOP_KEYS.items():
        if key not in raw:
            raise ValueError(f"Missing required top-level key: '{key}'")
        if not isinstance(raw[key], expected_type):
            raise ValueError(
                f"Top-level key '{key}' must be {expected_type.__name__}, "
                f"got {type(raw[key]).__name__}"
            )

    # Metadata keys
    meta = raw["metadata"]
    missing_meta = _REQUIRED_METADATA_KEYS - set(meta.keys())
    if missing_meta:
        raise ValueError(f"metadata is missing required keys: {missing_meta}")

    # Frames
    frames = raw["frames"]
    if not isinstance(frames, list):
        raise ValueError("'frames' must be a JSON array")

    if strict:
        for frame_idx, frame in enumerate(frames):
            if "frame_id" not in frame:
                raise ValueError(f"Frame at index {frame_idx} is missing 'frame_id'")
            for det_idx, det in enumerate(frame.get("detections", [])):
                missing = _REQUIRED_DETECTION_KEYS - set(det.keys())
                if missing:
                    raise ValueError(
                        f"Frame {frame['frame_id']}, detection {det_idx} "
                        f"is missing required keys: {missing}"
                    )
                # bbox_xyxy must have exactly 4 elements
                if len(det.get("bbox_xyxy", [])) != 4:
                    raise ValueError(
                        f"Frame {frame['frame_id']}, detection {det_idx}: "
                        "bbox_xyxy must have exactly 4 elements [x1,y1,x2,y2]"
                    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_json(
    path: "str | Path",
    *,
    strict_validate: bool = False,
) -> SegmentationDocument:
    """
    Load a segmentation JSON file and return a ``SegmentationDocument``.

    Parameters
    ----------
    path            : str or Path — path to the ``.json`` file
    strict_validate : bool        — run per-detection validation (slower)

    Returns
    -------
    SegmentationDocument

    Raises
    ------
    FileNotFoundError  — if the file does not exist
    ValueError         — if the JSON fails schema validation
    json.JSONDecodeError — if the file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    validate_raw(raw, strict=strict_validate)
    return SegmentationDocument.from_dict(raw)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_json(
    doc:    SegmentationDocument,
    path:   "str | Path",
    *,
    indent: int  = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Serialise a ``SegmentationDocument`` to disk as a JSON file.

    Parameters
    ----------
    doc          : SegmentationDocument
    path         : str or Path — output path (parent directories are created)
    indent       : int         — JSON indentation level (default: 2)
    ensure_ascii : bool        — passed to ``json.dump``

    Notes
    -----
    All numpy integer / float types are converted to Python-native types via
    :func:`botsort_module.utils.to_native` before serialisation to avoid
    ``TypeError: Object of type int64 is not JSON serializable``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = to_native(doc.to_dict())

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=indent, ensure_ascii=ensure_ascii)


# ---------------------------------------------------------------------------
# Convenience: create a blank document template
# ---------------------------------------------------------------------------

def make_empty_document(
    video:     str = "",
    weights:   str = "",
    timestamp: str = "",
    fps:       float = 30.0,
    resolution: "list[int] | None" = None,
) -> SegmentationDocument:
    """
    Create a brand-new SegmentationDocument with no frames.

    Useful for building segmentation output programmatically before passing
    it to the tracker pipeline.

    Example
    -------
    >>> from botsort_module.io import make_empty_document
    >>> from botsort_module.schema import FrameRecord, DetectionRecord
    >>> doc = make_empty_document(video="clip.mp4", fps=30.0)
    >>> doc.frames.append(FrameRecord(frame_id=0, detections=[...]))
    """
    from datetime import datetime
    from .schema import VideoMetadata

    return SegmentationDocument(
        metadata=VideoMetadata(
            video       = video,
            weights     = weights,
            timestamp   = timestamp or datetime.now().isoformat(timespec="seconds"),
            fps         = fps,
            resolution  = resolution,
        ),
        frames=[],
    )
