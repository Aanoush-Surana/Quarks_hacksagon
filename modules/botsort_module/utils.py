"""
utils.py
========
Shared utility helpers for the botsort_module package.

Functions
---------
iou_xyxy          — vectorised IoU between one box and N boxes
to_native         — recursively convert numpy types → Python-native
bbox_xyxy_to_xywh — coordinate conversion helper
mask_to_polygon   — OpenCV contour → list-of-points polygon
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bounding-box geometry
# ---------------------------------------------------------------------------

def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.

    Parameters
    ----------
    box   : shape (4,)   — [x1, y1, x2, y2]
    boxes : shape (N, 4) — [x1, y1, x2, y2]

    Returns
    -------
    ious : shape (N,)  — float32 IoU values in [0, 1]
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(xi2 - xi1, 0.0)
    inter_h = np.maximum(yi2 - yi1, 0.0)
    inter   = inter_w * inter_h

    area_box   = (box[2]   - box[0])   * (box[3]   - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union      = area_box + area_boxes - inter + 1e-7

    return (inter / union).astype(np.float32)


def bbox_xyxy_to_xywh(xyxy: List[float]) -> List[float]:
    """
    Convert ``[x1, y1, x2, y2]`` → ``[cx, cy, w, h]``.

    All values remain in absolute pixel coordinates.
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return [cx, cy, w, h]


def bbox_xywh_to_xyxy(xywh: List[float]) -> List[float]:
    """
    Convert ``[cx, cy, w, h]`` → ``[x1, y1, x2, y2]``.
    """
    cx, cy, w, h = xywh
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def to_native(obj):
    """
    Recursively convert numpy integers / floats / arrays to Python-native types.

    Needed because ``json.dump`` cannot serialise numpy types.

    Examples
    --------
    >>> to_native(np.int64(5))
    5
    >>> to_native({"a": np.float32(1.5)})
    {"a": 1.5}
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def mask_to_polygon(
    binary_mask: np.ndarray,
    simplify_epsilon: float = 1.0,
) -> List[List[float]]:
    """
    Convert a binary mask (uint8, 0/255) to a contour polygon.

    Parameters
    ----------
    binary_mask       : np.ndarray — uint8, shape (H, W)
    simplify_epsilon  : float      — Douglas-Peucker epsilon; 0 disables it

    Returns
    -------
    list of [x, y] pairs representing the outer contour.  Empty list if no
    contour is found.
    """
    try:
        import cv2  # optional if user doesn't need this helper
    except ImportError:
        return []

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Largest contour
    contour = max(contours, key=cv2.contourArea)

    if simplify_epsilon > 0:
        arc = cv2.arcLength(contour, closed=True)
        contour = cv2.approxPolyDP(contour, simplify_epsilon * arc / 1000.0, closed=True)

    # Shape: (N, 1, 2) → flatten to [[x,y], ...]
    return [[float(pt[0][0]), float(pt[0][1])] for pt in contour]


# ---------------------------------------------------------------------------
# Track ↔ detection matching
# ---------------------------------------------------------------------------

def match_tracks_to_detections(
    track_bboxes: np.ndarray,
    det_bboxes:   np.ndarray,
    iou_threshold: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Greedy 1-to-1 matching: pair each track bbox to the highest-IoU detection.

    Parameters
    ----------
    track_bboxes  : (T, 4) float32 — [x1,y1,x2,y2] for each returned STrack
    det_bboxes    : (D, 4) float32 — [x1,y1,x2,y2] for each input detection
    iou_threshold : float          — matches below this IoU are discarded

    Returns
    -------
    list of (detection_index, track_index) integer pairs
    """
    if track_bboxes.shape[0] == 0 or det_bboxes.shape[0] == 0:
        return []

    matched: List[Tuple[int, int]] = []
    used_dets: set = set()

    for t_idx in range(track_bboxes.shape[0]):
        ious = iou_xyxy(track_bboxes[t_idx], det_bboxes)
        # Zero out already-used detections
        for used in used_dets:
            ious[used] = -1.0
        best_d = int(np.argmax(ious))
        if ious[best_d] >= iou_threshold:
            matched.append((best_d, t_idx))
            used_dets.add(best_d)

    return matched
