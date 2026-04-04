"""
detection_prefilter.py — Pre-BoT-SORT detection filter.

Addresses **Problem 4**: occlusion-driven detection flicker that creates
spurious BoT-SORT track IDs.  Sits between raw YOLO output and the tracker.

Suppressed detections are NOT dropped — they are separated into a second list
so the temporal-fusion posterior can still incorporate mask geometry while
BoT-SORT is shielded from the noisy re-appearances.

Stuff (semantic) classes are separated into a third list and never enter the
tracker or the Bayesian fusion — they are rendered as raw mask overlays only.
"""

from __future__ import annotations

from typing import Optional

from .helpers import iou_pair


# Default stuff classes — large semantic regions that should NOT be tracked
# as individual objects by BoT-SORT.
_DEFAULT_STUFF_CLASSES: frozenset[str] = frozenset({
    "sky", "vegetation", "drivable area", "drivable_area",
    "curb", "fallback", "road",
})


class DetectionPrefilter:
    """Filter raw YOLO detections before they reach BoT-SORT.

    Prevents spurious track-ID creation caused by flickering occluded objects
    by suppressing detections that reappear at high IoU after a short absence
    while the fusion core marks them as occluded / hallucinated.

    Stuff (semantic) classes are split out entirely — they never enter the
    tracker or the temporal fusion state machine.

    Parameters:
        flicker_iou_threshold:       IoU floor for a reappearance to be
                                     considered a candidate flicker.
        flicker_suppression_frames:  Max absence gap (in frames) for the
                                     flicker heuristic to trigger.
        min_confidence_to_pass:      Hard confidence floor — anything below
                                     is dropped entirely.
        occlusion_confidence_gate:   Confidence a detection must exceed when
                                     its fusion state is ``"occluded"``.
        stuff_classes:               Class names treated as semantic "stuff"
                                     (not tracked). ``None`` → default set.
    """

    def __init__(
        self,
        flicker_iou_threshold: float = 0.45,
        flicker_suppression_frames: int = 4,
        min_confidence_to_pass: float = 0.25,
        occlusion_confidence_gate: float = 0.45,
        stuff_classes: Optional[set[str]] = None,
    ) -> None:
        self.flicker_iou_threshold = flicker_iou_threshold
        self.flicker_suppression_frames = flicker_suppression_frames
        self.min_confidence_to_pass = min_confidence_to_pass
        self.occlusion_confidence_gate = occlusion_confidence_gate
        self.stuff_classes: frozenset[str] = (
            frozenset(stuff_classes) if stuff_classes is not None
            else _DEFAULT_STUFF_CLASSES
        )

        # Internal per-track state: track_id -> {...}
        self._tracks: dict[int, dict] = {}

        # Per-frame stats (reset each call to filter)
        self._suppressed_this_frame: int = 0
        self._suppressed_ids: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        raw_detections: list[dict],
        current_frame_index: int,
        fusion_states: dict[int, str],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Filter detections, splitting into clean + suppressed + stuff.

        Args:
            raw_detections:      List of detection dicts from
                                 ``extract_detections_from_result()``.
            current_frame_index: Monotonically increasing frame counter.
            fusion_states:       ``{track_id: state_str}`` from
                                 ``TemporalMaskFusion.get_states()``.
                                 Pass ``{}`` if fusion is not yet initialised.

        Returns:
            ``(clean_detections, suppressed_detections, stuff_detections)``

            * *clean_detections* — pass these to BoT-SORT.
            * *suppressed_detections* — pass these to
              ``TemporalMaskFusion.update()`` (mask data is preserved but
              ``last_confirmed_frame`` is **not** reset).
            * *stuff_detections* — semantic classes that bypass tracking
              and Bayesian fusion entirely.

        Side-effects:
            Updates internal per-track bookkeeping.

        Addresses:
            **Problem 4** — occlusion-driven detection flicker.
        """
        self._suppressed_this_frame = 0
        self._suppressed_ids = []

        # --- Split stuff vs instance detections up-front ---
        stuff: list[dict] = []
        instance_dets: list[dict] = []
        for det in raw_detections:
            if det.get("class_name", "") in self.stuff_classes:
                stuff.append(det)
            else:
                instance_dets.append(det)

        clean: list[dict] = []
        suppressed: list[dict] = []

        seen_this_frame: set[int] = set()

        for det in instance_dets:
            conf: float = det.get("confidence", 0.0)
            track_id: Optional[int] = det.get("track_id")

            # --- Step 1: hard confidence floor ---
            if conf < self.min_confidence_to_pass:
                continue  # dropped entirely

            # No track_id → can't do temporal analysis, pass through
            if track_id is None:
                clean.append(det)
                continue

            seen_this_frame.add(track_id)
            suppress = False

            # --- Step 2a: flicker after short absence ---
            if track_id in self._tracks:
                trk = self._tracks[track_id]
                frames_absent = trk["frames_absent"]

                if (
                    frames_absent > 0
                    and frames_absent <= self.flicker_suppression_frames
                ):
                    iou = iou_pair(det["bbox"], trk["last_bbox"])
                    fstate = fusion_states.get(track_id, "")
                    if iou >= self.flicker_iou_threshold and fstate in (
                        "occluded",
                        "hallucinated",
                    ):
                        suppress = True

            # --- Step 2b: occlusion confidence gate ---
            if (
                not suppress
                and fusion_states.get(track_id) == "occluded"
                and conf < self.occlusion_confidence_gate
            ):
                suppress = True

            # --- Step 3: update internal state ---
            self._tracks[track_id] = {
                "last_seen_frame": current_frame_index,
                "last_bbox": det["bbox"],
                "frames_absent": 0,
            }

            if suppress:
                det["flicker_suppressed"] = True
                suppressed.append(det)
                self._suppressed_this_frame += 1
                self._suppressed_ids.append(track_id)
            else:
                det["flicker_suppressed"] = False
                clean.append(det)

        # --- Step 4: age-out tracks not seen this frame ---
        for tid in list(self._tracks.keys()):
            if tid not in seen_this_frame:
                self._tracks[tid]["frames_absent"] += 1

        return clean, suppressed, stuff

    def get_flicker_stats(self) -> dict:
        """Return per-frame flicker suppression statistics.

        Returns:
            ``{"total_suppressed_this_frame": int,
               "track_ids_suppressed": list[int]}``
        """
        return {
            "total_suppressed_this_frame": self._suppressed_this_frame,
            "track_ids_suppressed": list(self._suppressed_ids),
        }
