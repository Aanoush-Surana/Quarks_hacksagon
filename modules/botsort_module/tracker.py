"""
tracker.py
==========
BotSortTracker — wraps the ultralytics BoT-SORT algorithm so it can be driven
by **pre-computed** detection lists (from JSON) instead of raw video frames.

Key design choice
-----------------
The official ``model.track()`` API fuses detection and tracking into one call.
Here we bypass detection and feed a minimal ``_DetectionStub`` directly into
``ultralytics.trackers.bot_sort.BotSort.update()``, passing the same interface
the tracker expects from a ``Results.boxes`` object.

This is intentionally version-robust:
- The stub exposes *both* ``results.attr`` and ``results.boxes.attr`` access
  patterns (the self-reference ``self.boxes = self`` trick), covering different
  ultralytics major versions.
- An ``IterableSimpleNamespace`` (or fallback ``SimpleNamespace``) is used for
  the tracker config to avoid importing private ultralytics classes.

Requirements
------------
  ultralytics >= 8.0
  PyTorch (CPU-only is fine; tracker is pure-numpy internally)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .schema import DetectionRecord
from .utils import match_tracks_to_detections

# ---------------------------------------------------------------------------
# Ultralytics imports — handled gracefully so import errors are informative
# ---------------------------------------------------------------------------

try:
    try:
        # Newer ultralytics versions (e.g. 8.4+)
        from ultralytics.trackers.bot_sort import BOTSORT as _BotSort
    except ImportError:
        # Older ultralytics versions
        from ultralytics.trackers.bot_sort import BotSort as _BotSort
except ImportError as exc:
    raise ImportError(
        "ultralytics >= 8.0 is required for BotSort tracking. "
        "Install it with: pip install ultralytics"
    ) from exc

# IterableSimpleNamespace was added in later ultralytics versions; fall back
# to stdlib SimpleNamespace which satisfies the same .attr access pattern.
try:
    from ultralytics.utils import IterableSimpleNamespace as _NS
except ImportError:
    from types import SimpleNamespace as _NS  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Detection stub
# ---------------------------------------------------------------------------

class _DetectionStub:
    # Minimal proxy that looks like ``ultralytics.engine.results.Boxes``
    def __init__(
        self,
        xyxy,
        conf,
        cls,
        orig_shape: Tuple[int, int] = (1080, 1920),
    ) -> None:
        self.xyxy  = torch.as_tensor(xyxy, dtype=torch.float32)
        if self.xyxy.ndim == 1:
            self.xyxy = self.xyxy.unsqueeze(0)
            
        self.conf  = torch.as_tensor(conf, dtype=torch.float32)
        self.cls   = torch.as_tensor(cls,  dtype=torch.float32)
        self.id    = None          # tracker will fill this in
        self.orig_shape = orig_shape

        # Self-reference so both results.xyxy and results.boxes.xyxy work
        self.boxes  = self
        self.masks  = None
        self.probs  = None

    @property
    def xywh(self) -> torch.Tensor:
        """Dynamically compute xywh format as required by init_track in new ultralytics."""
        xywh = torch.empty_like(self.xyxy)
        if self.xyxy.numel() > 0:
            xywh[..., 0] = (self.xyxy[..., 0] + self.xyxy[..., 2]) / 2  # x center
            xywh[..., 1] = (self.xyxy[..., 1] + self.xyxy[..., 3]) / 2  # y center
            xywh[..., 2] = self.xyxy[..., 2] - self.xyxy[..., 0]        # width
            xywh[..., 3] = self.xyxy[..., 3] - self.xyxy[..., 1]        # height
        return xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetectionStub":
        """Newer ultralytics explicitly slices the results object itself."""
        return _DetectionStub(
            xyxy=self.xyxy[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
            orig_shape=self.orig_shape
        )


def _build_stub(
    detections: List[DetectionRecord],
    orig_shape: Tuple[int, int] = (1080, 1920),
) -> _DetectionStub:
    """Build a ``_DetectionStub`` from a list of ``DetectionRecord`` objects."""
    if not detections:
        return _DetectionStub(
            xyxy = np.zeros((0, 4), dtype=np.float32),
            conf = np.zeros((0,),   dtype=np.float32),
            cls  = np.zeros((0,),   dtype=np.float32),
            orig_shape = orig_shape,
        )

    xyxy = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
    conf = np.array([d.confidence for d in detections], dtype=np.float32)
    cls  = np.array([d.class_id   for d in detections], dtype=np.float32)

    return _DetectionStub(xyxy=xyxy, conf=conf, cls=cls, orig_shape=orig_shape)


# ---------------------------------------------------------------------------
# Public tracker class
# ---------------------------------------------------------------------------

class BotSortTracker:
    """
    Drop-in BoT-SORT wrapper for pre-computed detection lists.

    Typical usage
    -------------
    >>> tracker = BotSortTracker(fps=30.0)
    >>> for frame in doc.frames:
    ...     pairs = tracker.update(frame.detections)      # [(det_idx, tid), ...]
    ...     for det_idx, tid in pairs:
    ...         frame.detections[det_idx].track_id = tid

    Parameters
    ----------
    cfg_path : Path or str, optional
        YAML config for BoT-SORT.  Defaults to the bundled
        ``botsort_module/config/botsort.yaml``.
    fps : float
        Video frame rate.  Used to size the Kalman filter's motion model.
    with_reid : bool
        Enable appearance ReID.  Requires model_weights in the config and
        a real frame image passed to ``update()``.
    cfg_overrides : dict, optional
        Key/value pairs that override the YAML config without editing the file.
        Example: ``{"track_buffer": 60, "match_thresh": 0.7}``
    """

    _DEFAULT_CFG = Path(__file__).parent / "config" / "botsort.yaml"

    def __init__(
        self,
        cfg_path:      Optional[Path] = None,
        fps:           float = 30.0,
        with_reid:     bool  = False,
        cfg_overrides: Optional[Dict] = None,
    ) -> None:
        cfg_path = Path(cfg_path) if cfg_path else self._DEFAULT_CFG
        if not cfg_path.exists():
            raise FileNotFoundError(f"BoT-SORT config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg_dict: dict = yaml.safe_load(fh)

        # Apply any caller-supplied overrides
        if cfg_overrides:
            cfg_dict.update(cfg_overrides)

        cfg_dict["with_reid"] = with_reid
        
        # Newer ultralytics expects 'model' instead of 'model_weights' for ReID
        if with_reid and "model" not in cfg_dict:
            cfg_dict["model"] = cfg_dict.get("model_weights", "osnet_x0_25_msmt17.pt")

        args = _NS(**cfg_dict)
        self._tracker    = _BotSort(args, frame_rate=fps)
        self._fps        = fps
        self._with_reid  = with_reid
        self._frame_idx  = 0

        # Cache the inferred orig_shape (updated per-frame if video is provided)
        self._default_orig_shape: Tuple[int, int] = (1080, 1920)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[DetectionRecord],
        frame_img:  Optional[np.ndarray] = None,
        orig_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Process one frame: advance the tracker and return track assignments.

        Parameters
        ----------
        detections : list[DetectionRecord]
            All detections for the current frame (``track_id`` may be None).
        frame_img  : np.ndarray (H, W, 3) BGR, optional
            The raw video frame.  Required only when ReID is enabled.
        orig_shape : (H, W), optional
            Frame resolution.  Inferred from ``frame_img`` if not provided.

        Returns
        -------
        list of (detection_index, track_id) tuples.
            Only detections that were matched to a track appear here.
            Detections not matched remain unassigned (track_id stays None).
        """
        if orig_shape is None and frame_img is not None:
            orig_shape = frame_img.shape[:2]
        if orig_shape is None:
            orig_shape = self._default_orig_shape

        stub = _build_stub(detections, orig_shape)

        try:
            online_targets = self._tracker.update(stub, frame_img)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"BotSort.update() raised {type(exc).__name__}: {exc}. "
                "Frame skipped — tracker state preserved.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._frame_idx += 1
            return []

        self._frame_idx += 1

        if len(online_targets) == 0:
            return []

        # ------------------------------------------------------------------
        # Match returned targets back to input detections by highest IoU.
        # Newer ultralytics returns np.ndarray of shape (N, 8) [x1,y1,x2,y2, tid, seq, cls, idx]
        # Older returns a list of STrack objects.
        # ------------------------------------------------------------------
        if isinstance(online_targets, np.ndarray):
            track_bboxes = online_targets[:, :4].astype(np.float32)
            track_ids    = online_targets[:, 4].astype(int)
        else:
            track_bboxes = np.array(
                [self._strack_xyxy(t) for t in online_targets],
                dtype=np.float32,
            )
            track_ids    = np.array([int(t.track_id) for t in online_targets], dtype=int)

        det_bboxes = np.array(
            [d.bbox_xyxy for d in detections],
            dtype=np.float32,
        ) if detections else np.zeros((0, 4), dtype=np.float32)   # shape (D, 4)

        matched_pairs = match_tracks_to_detections(track_bboxes, det_bboxes)

        return [
            (det_idx, int(track_ids[t_idx]))
            for det_idx, t_idx in matched_pairs
        ]

    def reset(self) -> None:
        """
        Hard-reset the tracker state.

        Call this between videos to avoid track ID carry-over.
        """
        cfg_dict: dict = {}
        with open(self._DEFAULT_CFG, "r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh)
        cfg_dict["with_reid"] = self._with_reid
        args = _NS(**cfg_dict)
        self._tracker   = _BotSort(args, frame_rate=self._fps)
        self._frame_idx = 0

    @property
    def frame_index(self) -> int:
        """Number of frames processed so far."""
        return self._frame_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strack_xyxy(track) -> List[float]:
        """
        Extract [x1, y1, x2, y2] from an STrack object.

        Different ultralytics versions expose different property names.
        Priority: .xyxy → .tlbr → .to_tlbr()
        """
        if hasattr(track, "xyxy"):
            b = track.xyxy
        elif hasattr(track, "tlbr"):
            b = track.tlbr
        elif hasattr(track, "to_tlbr"):
            b = track.to_tlbr()
        else:
            raise AttributeError(
                f"STrack object {type(track)} has no known bbox property "
                "(tried: .xyxy, .tlbr, .to_tlbr())"
            )
        # Could be a Tensor or ndarray
        if hasattr(b, "cpu"):
            return b.cpu().numpy().tolist()
        return list(b)
