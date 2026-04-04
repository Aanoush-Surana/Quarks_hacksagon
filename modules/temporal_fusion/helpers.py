"""
helpers.py — Shared utilities for the temporal_fusion package.

Contains pure functions used by all other modules. No class definitions.
No imports from sibling modules (prevents circular imports).

Functions:
    clip_bbox          — Clip bbox to frame bounds, return None if zero-area.
    iou_pair           — Vectorised IoU between two bboxes.
    bbox_center        — Centre point of a bbox.
    deque_median       — Median of a deque without list conversion.
    resize_mask_to_bbox— Resize a 64×64 mask to bbox dimensions + threshold.
    color_for_id       — Deterministic RGB from track ID hash.
    extract_detections_from_result — Pull detection dicts from a YOLO result.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np


# Classes treated as large semantic regions — bbox is derived from the mask
# rather than from the YOLO detection-head bbox, which can disagree with
# the segmentation-head mask for large irregular shapes.
SEMANTIC_CLASSES: frozenset = frozenset({"drivable_area", "drivable area"})


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------

def clip_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> Optional[tuple[int, int, int, int]]:
    """Clip a bbox (x1, y1, x2, y2) to *frame_shape* (H, W).

    Returns:
        Clipped bbox tuple, or ``None`` if the result has zero area.
    """
    H, W = frame_shape[:2]
    x1 = max(0, min(int(bbox[0]), W))
    y1 = max(0, min(int(bbox[1]), H))
    x2 = max(0, min(int(bbox[2]), W))
    y2 = max(0, min(int(bbox[3]), H))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def iou_pair(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    """Compute IoU between two bboxes using numpy (no loops).

    Args:
        bbox_a: (x1, y1, x2, y2)
        bbox_b: (x1, y1, x2, y2)

    Returns:
        Intersection-over-Union as a float in [0, 1].
    """
    a = np.asarray(bbox_a, dtype=np.float32)
    b = np.asarray(bbox_b, dtype=np.float32)

    inter_x1 = np.maximum(a[0], b[0])
    inter_y1 = np.maximum(a[1], b[1])
    inter_x2 = np.minimum(a[2], b[2])
    inter_y2 = np.minimum(a[3], b[3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return float(inter_area / union)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the centre (cx, cy) of a bbox (x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


# ---------------------------------------------------------------------------
# Deque helpers
# ---------------------------------------------------------------------------

def deque_median(d: deque) -> float:
    """Compute the median of a deque *without* converting to a Python list.

    Uses ``np.fromiter`` for zero-copy ingestion in the hot path.

    Args:
        d: A deque of numeric values.

    Returns:
        The median as a Python float.  Returns 0.0 for an empty deque.
    """
    n = len(d)
    if n == 0:
        return 0.0
    arr = np.fromiter(d, dtype=np.float32, count=n)
    return float(np.median(arr))


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def resize_mask_to_bbox(
    mask_64: np.ndarray,
    bbox_h: int,
    bbox_w: int,
) -> np.ndarray:
    """Resize a (64, 64) probability mask to (bbox_h, bbox_w) and threshold.

    Args:
        mask_64: float32 mask of shape (64, 64), values in [0, 1].
        bbox_h:  Target height (pixels).
        bbox_w:  Target width  (pixels).

    Returns:
        uint8 array of shape (bbox_h, bbox_w) with values 0 or 255.
    """
    if bbox_h <= 0 or bbox_w <= 0:
        return np.zeros((max(1, bbox_h), max(1, bbox_w)), dtype=np.uint8)
    resized = cv2.resize(mask_64, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
    return ((resized > 0.5) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic RGB colour from a track ID (hash-based).

    Args:
        track_id: Integer track identifier.

    Returns:
        (R, G, B) tuple with values in [40, 255] to avoid very dark colours.
    """
    h = hash(track_id * 49979693)  # large prime for spread
    r = 40 + (h & 0xFF) % 216
    g = 40 + ((h >> 8) & 0xFF) % 216
    b = 40 + ((h >> 16) & 0xFF) % 216
    return (r, g, b)


# ---------------------------------------------------------------------------
# Detection extraction from YOLO result
# ---------------------------------------------------------------------------

def extract_detections_from_result(result, model) -> list[dict]:
    """Extract detection dicts from a single YOLO *result* object.

    This is intended for segmentation models (``task='segment'``).  Each
    detection dict contains all fields needed by the temporal-fusion pipeline.

    Args:
        result: A single element from the list returned by ``model.track()``
                or ``model.predict()``.
        model:  The ``ultralytics.YOLO`` model instance (must have
                ``task == 'segment'``).

    Returns:
        List of dicts, each with keys:
            ``bbox``        — (x1, y1, x2, y2) int tuple
            ``track_id``    — int or None
            ``confidence``  — float
            ``class_id``    — int
            ``class_name``  — str
            ``mask``        — np.ndarray uint8 (H, W), 0 or 255

        Returns an empty list when there are no detections or masks.
    """
    boxes = result.boxes
    masks = result.masks

    # Guard: no detections at all
    if boxes is None or len(boxes) == 0:
        return []

    # Guard: segmentation masks missing (seg failed or detect-only model)
    if masks is None:
        return []

    # Guard: tracker hasn't assigned IDs yet
    has_ids = boxes.id is not None

    names = model.names
    H, W = result.orig_shape  # original frame dimensions

    mask_data = masks.data.cpu().numpy()  # (N, mask_h, mask_w) float
    detections: list[dict] = []

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, f"cls_{cls_id}")
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        track_id: int | None = None
        if has_ids:
            track_id = int(box.id[0].item())

        # Resize raw mask to full frame using INTER_NEAREST, threshold → uint8
        raw_mask = mask_data[i]
        if (raw_mask.shape[0], raw_mask.shape[1]) != (H, W):
            resized = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            resized = raw_mask
        binary_mask = ((resized > 0.5) * 255).astype(np.uint8)
        binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255

        # For large semantic classes (drivable_area), YOLO's detection-head
        # bbox often disagrees with the segmentation-head mask.  Derive the
        # bbox from the actual mask pixels so downstream crop / project
        # operations are spatially consistent.
        if cls_name in SEMANTIC_CLASSES and binary_mask.any():
            ys, xs = np.where(binary_mask > 0)
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "track_id": track_id,
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name,
            "mask": binary_mask,
        })

    return detections
