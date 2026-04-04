"""
mask_postprocessor.py — Full-frame mask projection from 64×64 temporal posterior.

Projects the 64×64 binary temporal mask back to the original frame dimensions
via the detection bounding box.  No morphological post-processing is applied.

Stuff detections (state == "stuff") bypass the 64×64 projection entirely
and use their raw full-frame mask directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from .helpers import clip_bbox


def project_and_fill(
    outputs: dict[int, dict],
    frame_shape: tuple[int, int],
) -> dict[int, dict]:
    """Project 64×64 temporal masks to full-frame resolution.

    Args:
        outputs:      Dict returned by ``TemporalMaskFusion.update()``,
                      keyed by track_id.
        frame_shape:  ``(H, W)`` of the original video frame.

    Returns:
        The same *outputs* dict with ``full_frame_mask`` populated for each
        object (uint8 array of shape ``(H, W)`` with values 0 or 255).

    Side-effects:
        Mutates *outputs* in place (sets ``full_frame_mask``).
    """
    H, W = frame_shape[:2]

    for tid, obj in outputs.items():
        # --- Stuff detections: use raw mask directly ---
        if obj.get("state") == "stuff":
            raw_mask = obj.pop("raw_mask", None)
            if raw_mask is not None and hasattr(raw_mask, "shape") and raw_mask.shape == (H, W):
                # Apply morphological operations to reduce jitter and smooth edges for stuff classes (like vegetation)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                obj["full_frame_mask"] = raw_mask
            else:
                obj["full_frame_mask"] = np.zeros((H, W), dtype=np.uint8)
            continue

        # If full_frame_mask was already populated upstream (e.g. raw mask
        # bypass for semantic classes like drivable_area), keep it as-is
        # and skip the 64×64 → bbox projection.
        existing = obj.get("full_frame_mask")
        if existing is not None:
            if hasattr(existing, "shape") and existing.shape == (H, W):
                continue

        bbox = obj.get("bbox")
        if bbox is None:
            obj["full_frame_mask"] = np.zeros((H, W), dtype=np.uint8)
            continue

        clipped = clip_bbox(bbox, (H, W))
        if clipped is None:
            obj["full_frame_mask"] = np.zeros((H, W), dtype=np.uint8)
            continue

        x1, y1, x2, y2 = clipped
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        if bbox_h <= 0 or bbox_w <= 0:
            obj["full_frame_mask"] = np.zeros((H, W), dtype=np.uint8)
            continue

        # --- Resize 64×64 binary mask to bbox dimensions ---
        mask_64 = obj.get("temporal_mask_binary_64")
        if mask_64 is None:
            obj["full_frame_mask"] = np.zeros((H, W), dtype=np.uint8)
            continue

        # mask_64 is uint8 0/255 — resize with INTER_LINEAR then re-threshold
        resized = cv2.resize(
            mask_64.astype(np.float32) / 255.0,
            (bbox_w, bbox_h),
            interpolation=cv2.INTER_LINEAR,
        )
        region_mask = ((resized > 0.5) * 255).astype(np.uint8)

        # --- Paste into full-frame canvas ---
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = region_mask
        obj["full_frame_mask"] = full_mask

    return outputs
