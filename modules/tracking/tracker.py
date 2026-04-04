import logging


class TrackerModule:
    """
    Statistics aggregator for BoT-SORT tracked detections.

    Actual tracking is performed by YOLO's built-in model.track() with
    BoT-SORT.  This module reads the track IDs produced by the model,
    maintains lifetime unique-object counts per class, and injects a
    ``tracking_stats`` dict into the frame_data payload.
    """

    # Area-type segmentation classes that should NOT be counted as objects
    EXCLUDE_CLASSES = {"drivable_area", "drivable area"}

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- Lifetime statistics ---
        self.seen_ids = set()           # every unique track-ID ever seen
        self.class_counts = {}          # {class_name: unique_count}
        self._id_class_map = {}         # {track_id: class_name} for lookup

        self.logger.info("TrackerModule initialized (BoT-SORT stats aggregator)")

    # ------------------------------------------------------------------
    def process_frame(self, frame, tracking_data):
        """
        Read track IDs from *tracking_data* (set by inference.py),
        update lifetime statistics, and inject ``tracking_stats`` back
        into *tracking_data*.

        The frame is returned unchanged — all visual rendering of IDs
        is already done in ``SegmentationModel.process_frame``.
        """
        detections = tracking_data.get("detections", [])
        active_count = 0

        for det in detections:
            track_id  = det.get("track_id")
            cls_name  = det.get("class_name", "unknown")

            # Skip area segments from unique-object counting
            if cls_name in self.EXCLUDE_CLASSES:
                continue

            if track_id is not None:
                active_count += 1

                # First time we see this track ID → new unique object
                if track_id not in self.seen_ids:
                    self.seen_ids.add(track_id)
                    self._id_class_map[track_id] = cls_name
                    self.class_counts[cls_name] = (
                        self.class_counts.get(cls_name, 0) + 1
                    )

        # Count only trackable (non-area) detections this frame
        trackable = sum(
            1 for d in detections
            if d.get("class_name") not in self.EXCLUDE_CLASSES
        )

        tracking_data["tracking_stats"] = {
            "active_tracked":       active_count,
            "total_unique_objects":  len(self.seen_ids),
            "class_counts":         dict(self.class_counts),
            "detections_this_frame": trackable,
        }

        return frame
