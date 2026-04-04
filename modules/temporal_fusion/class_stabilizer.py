"""
class_stabilizer.py — Per-track class-label vote buffer.

Addresses **Problem 2**: class confusion between visually similar classes
(bus/truck, vegetation/fallback, motorcycle/bicycle) at low confidence.

The Bayesian posterior in temporal_fusion_core accumulates mask *geometry*.
Class label is a separate categorical signal that must be stabilised
independently via a weighted-vote buffer.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional


class ClassStabilizer:
    """Maintain a per-track vote buffer to stabilise class labels.

    Only applies stabilisation between pre-defined *confusion_pairs*.
    For non-confused classes the raw label is passed through immediately.

    Parameters:
        vote_buffer_size:   Number of frames of class-vote history per track.
        confidence_weighted: Weight votes by detection confidence.
        switch_threshold:   Fraction of weighted votes the new class must
                            achieve before a class switch is accepted.
        confusion_pairs:    Symmetric pairs of class names that are known
                            to be visually confusable.
    """

    def __init__(
        self,
        vote_buffer_size: int = 20,
        confidence_weighted: bool = True,
        switch_threshold: float = 0.65,
        confusion_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        self.vote_buffer_size = vote_buffer_size
        self.confidence_weighted = confidence_weighted
        self.switch_threshold = switch_threshold

        if confusion_pairs is None:
            confusion_pairs = [
                ("bus", "truck"),
                ("vegetation", "fallback"),
                ("motorcycle", "bicycle"),
            ]

        # Build a fast look-up: for each class name, which class(es) it is
        # confused with.  Symmetric.
        self._confused_with: dict[str, set[str]] = defaultdict(set)
        for a, b in confusion_pairs:
            self._confused_with[a].add(b)
            self._confused_with[b].add(a)

        # Per-track state
        self._tracks: dict[int, dict] = {}
        # track_id -> {
        #   "vote_history": deque of (class_id, class_name, confidence),
        #   "stable_class_id": int,
        #   "stable_class_name": str,
        # }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stabilize(
        self,
        track_id: int,
        raw_class_id: int,
        raw_class_name: str,
        confidence: float,
    ) -> tuple[int, str]:
        """Return the stabilised (class_id, class_name) for *track_id*.

        Args:
            track_id:       Integer track identifier.
            raw_class_id:   Class index from the detector.
            raw_class_name: Human-readable class name from the detector.
            confidence:     Detection confidence for this frame.

        Returns:
            ``(stable_class_id, stable_class_name)``

        Side-effects:
            Appends to the internal vote history for *track_id*.

        Addresses:
            **Problem 2** — class confusion between similar classes.
        """
        # --- New track: initialise and return raw immediately ---
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "vote_history": deque(maxlen=self.vote_buffer_size),
                "stable_class_id": raw_class_id,
                "stable_class_name": raw_class_name,
            }
            # Append the first vote
            self._tracks[track_id]["vote_history"].append(
                (raw_class_id, raw_class_name, confidence)
            )
            return raw_class_id, raw_class_name

        trk = self._tracks[track_id]
        trk["vote_history"].append((raw_class_id, raw_class_name, confidence))

        stable_name = trk["stable_class_name"]

        # --- Check if raw and stable form a confusion pair ---
        if raw_class_name == stable_name:
            # Same class — no confusion, just keep stable
            return trk["stable_class_id"], stable_name

        is_confused = raw_class_name in self._confused_with.get(stable_name, set())

        if not is_confused:
            # Not a confusion pair → accept the raw class immediately
            trk["stable_class_id"] = raw_class_id
            trk["stable_class_name"] = raw_class_name
            return raw_class_id, raw_class_name

        # --- Confusion pair: compute weighted vote fraction ---
        history = trk["vote_history"]
        raw_weight = 0.0
        total_weight = 0.0

        for _cid, cname, conf in history:
            w = conf if self.confidence_weighted else 1.0
            total_weight += w
            if cname == raw_class_name:
                raw_weight += w

        if total_weight <= 0:
            return trk["stable_class_id"], trk["stable_class_name"]

        weighted_fraction = raw_weight / total_weight

        if weighted_fraction >= self.switch_threshold:
            # Enough evidence to switch
            trk["stable_class_id"] = raw_class_id
            trk["stable_class_name"] = raw_class_name
            return raw_class_id, raw_class_name

        # Hold the previous stable label
        return trk["stable_class_id"], trk["stable_class_name"]

    def reset(self, track_id: int) -> None:
        """Remove all state for *track_id* (called when a track is dropped).

        Args:
            track_id: The track to purge.
        """
        self._tracks.pop(track_id, None)

    def get_stability_report(self) -> dict[int, dict]:
        """Per-track stability diagnostics.

        Returns:
            ``{track_id: {"stable_class": str, "vote_entropy": float}}``

            *vote_entropy* is the Shannon entropy (base-2) over the
            class-vote distribution.  High entropy ⇒ unstable label.
        """
        report: dict[int, dict] = {}
        for tid, trk in self._tracks.items():
            history = trk["vote_history"]
            if len(history) == 0:
                report[tid] = {
                    "stable_class": trk["stable_class_name"],
                    "vote_entropy": 0.0,
                }
                continue

            # Accumulate per-class counts
            counts: dict[str, float] = defaultdict(float)
            for _cid, cname, conf in history:
                counts[cname] += conf if self.confidence_weighted else 1.0

            total = sum(counts.values())
            entropy = 0.0
            if total > 0:
                for c in counts.values():
                    p = c / total
                    if p > 0:
                        entropy -= p * math.log2(p)

            report[tid] = {
                "stable_class": trk["stable_class_name"],
                "vote_entropy": round(entropy, 4),
            }
        return report
