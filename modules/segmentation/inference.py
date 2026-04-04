import os
import cv2
import json
import logging
import numpy as np
import time
import torch
from pathlib import Path
from ultralytics import YOLO

class SegmentationModel:
    def __init__(self, weights_path, device=None, conf_thresh=0.25, iou_thresh=0.45):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Device selection
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # FP16 only works on CUDA
        self.use_half = torch.cuda.is_available()
        self.logger.info(f"Loading YOLOv8 model from {weights_path} (device={self.device}, half={self.use_half})")
        
        # TensorRT Optimization Check (CUDA only)
        weights_file = Path(weights_path)
        engine_path = weights_file.with_suffix('.engine')
        
        if engine_path.exists() and torch.cuda.is_available():
            self.logger.info(f"Loading TensorRT engine from {engine_path}")
            self.model = YOLO(str(engine_path), task='segment')
        else:
            self.logger.info(f"Loading standard YOLOv8 model from {weights_path}")
            self.model = YOLO(weights_path)
            # Export to TensorRT only if CUDA is actually available
            if torch.cuda.is_available():
                try:
                    self.logger.info("Attempting to export model to TensorRT (this may take a few minutes)...")
                    exported_path = self.model.export(format='engine', device=self.device, half=True)
                    self.logger.info(f"TensorRT export successful: {exported_path}")
                    self.model = YOLO(exported_path, task='segment')
                except Exception as e:
                    self.logger.warning(f"TensorRT export failed: {e}. Falling back to standard model.")
        
        self.names = self.model.names

        # Color palette
        self.DRIVABLE_NAME = "drivable_area"
        self.DRIVABLE_BGR = (50, 205, 50)

        self._PALETTE = [
            (86,180,233),(230,159,0),(240,228,66),(204,121,167),(0,114,178),
            (213,94,0),(0,158,115),(255,127,14),(148,103,189),(140,86,75),
            (127,127,127),(188,189,34),(23,190,207),(31,119,180),(255,187,120),
            (174,199,232),(255,152,150),(197,176,213),(196,156,148),(247,182,210),
        ]

    def get_colour(self, class_id, class_name):
        if class_name == self.DRIVABLE_NAME:
            return self.DRIVABLE_BGR

        c = self._PALETTE[class_id % len(self._PALETTE)]
        return (c[2], c[1], c[0])  # RGB → BGR

    def detect(self, frame):
        """Run YOLO inference + BoT-SORT tracking without rendering.

        Returns the raw ultralytics Result object for downstream
        temporal-fusion processing.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=640,
            device=self.device,
            retina_masks=True,
            half=self.use_half,
            persist=True,
            tracker="botsort.yaml",
            verbose=False
        )
        return results[0]

    def process_frame(self, frame, frame_idx, timestamp_sec):
        """
        Optimized segmentation processing (no accuracy loss)
        """
        H, W = frame.shape[:2]

        # BoT-SORT tracking with FP16 and Retina Masks
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=640,
            device=self.device,
            retina_masks=True,
            half=self.use_half,  # FP16 only on CUDA
            persist=True,  # Maintain track state across frames
            tracker="botsort.yaml",  # BoT-SORT tracker
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        frame_data = {
            "frame_id": frame_idx,
            "timestamp_sec": timestamp_sec,
            "detections": []
        }

        if masks is None or boxes is None:
            return frame, frame_data

        # Float overlay (only once, no copy needed as astype creates one)
        overlay = frame.astype(np.float32)
        mask_data = masks.data.cpu().numpy()
        mask_segments = masks.xy
        names = self.names

        # Pass 1: Apply mask transparency overlays
        for i, (box, mask) in enumerate(zip(boxes, mask_data)):
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, f"cls_{cls_id}")
            
            # Use INTER_NEAREST resize for mask speed if dimensions don't match
            if (mask.shape[0], mask.shape[1]) != (H, W):
                msk = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            else:
                msk = mask > 0.5
            
            # Skip tiny masks (no accuracy impact)
            if msk.sum() < 50:
                continue

            bgr_colour = self.get_colour(cls_id, cls_name)
            alpha = 0.60 if cls_name == self.DRIVABLE_NAME else 0.45
            overlay[msk] = overlay[msk] * (1 - alpha) + np.array(bgr_colour, dtype=np.float32) * alpha

        # Convert to uint8 once for final drawing
        final_frame = overlay.astype(np.uint8)

        # Pass 2: Draw contours, bboxes, and labels, compile JSON
        for i, (box, mask) in enumerate(zip(boxes, mask_data)):
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, f"cls_{cls_id}")
            conf = float(box.conf[0].item())
            bgr_colour = self.get_colour(cls_id, cls_name)

            # Track ID from BoT-SORT (None until tracker confirms)
            track_id = int(box.id[0].item()) if box.id is not None else None
            
            if (mask.shape[0], mask.shape[1]) != (H, W):
                msk = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            else:
                msk = mask > 0.5

            # Skip contours for tiny masks but keep bounding box and JSON
            if msk.sum() >= 50:
                contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(final_frame, contours, -1, bgr_colour, 2)

            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), bgr_colour, 2)
            
            # Label with track ID + class + confidence
            tid_str = f"ID:{track_id} " if track_id is not None else ""
            label = f"{tid_str}{cls_name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(final_frame, (x1, max(0, y1-th-bl-4)), (x1+tw+6, y1), bgr_colour, -1)
            tc = (0,0,0) if sum(bgr_colour) > 380 else (255,255,255)
            cv2.putText(final_frame, label, (x1+3, y1-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
            
            # Store data for JSON (includes track_id)
            poly_points = mask_segments[i].tolist() if i < len(mask_segments) else []
            frame_data["detections"].append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "track_id": track_id,
                "mask_polygon": poly_points
            })

        return final_frame, frame_data

    def render_fusion_outputs(self, frame, fusion_outputs, frame_idx, timestamp_sec):
        """Render temporal-fusion outputs onto a frame.

        Args:
            frame:           Original video frame (H, W, 3) uint8.
            fusion_outputs:  Dict from TemporalMaskFusion.update() after
                             project_and_fill().
            frame_idx:       Frame index.
            timestamp_sec:   Timestamp in seconds.

        Returns:
            (rendered_frame, frame_data_dict)
        """
        H, W = frame.shape[:2]
        frame_data = {
            "frame_id": frame_idx,
            "timestamp_sec": timestamp_sec,
            "detections": []
        }

        if not fusion_outputs:
            return frame.copy(), frame_data

        overlay = frame.astype(np.float32)

        # Pass 1: mask overlays
        for tid, obj in fusion_outputs.items():
            full_mask = obj.get("full_frame_mask")
            if full_mask is None:
                continue
            msk = full_mask > 127
            if msk.sum() < 50:
                continue

            cls_name = obj.get("stable_class_name", "unknown")
            bgr_colour = self.get_colour(obj.get("stable_class_id", 0), cls_name)
            state = obj.get("state", "visible")
            if state == "stuff":
                alpha = 0.25
            elif cls_name in (self.DRIVABLE_NAME, "drivable area"):
                alpha = 0.60
            elif state == "hallucinated":
                alpha = 0.25
            else:
                alpha = 0.45
            overlay[msk] = overlay[msk] * (1 - alpha) + np.array(bgr_colour, dtype=np.float32) * alpha

        final_frame = overlay.astype(np.uint8)

        # Pass 2: contours, bboxes, labels, JSON
        for tid, obj in fusion_outputs.items():
            state = obj.get("state", "visible")

            # Stuff classes: no bbox, no contour, no label — mask-only
            if state == "stuff":
                continue

            cls_name = obj.get("stable_class_name", "unknown")
            conf = obj.get("confidence", 0.0)
            bbox = obj.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            bgr_colour = self.get_colour(obj.get("stable_class_id", 0), cls_name)

            # Contours
            full_mask = obj.get("full_frame_mask")
            if full_mask is not None and full_mask.any():
                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(final_frame, contours, -1, bgr_colour, 2)

            # Bbox — dashed yellow for hallucinated
            if state == "hallucinated":
                _draw_dashed_rect(final_frame, (x1, y1), (x2, y2), (0, 255, 255), 2, 10)
            else:
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), bgr_colour, 2)

            # Label
            state_tag = ""
            if state == "hallucinated":
                state_tag = " [hall]"
            if obj.get("seg_was_skipped", False):
                state_tag += " [skip]"

            label = f"{cls_name} #{tid} {conf:.2f}{state_tag}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lbl_colour = (0, 255, 255) if state == "hallucinated" else bgr_colour
            cv2.rectangle(final_frame, (x1, max(0, y1-th-bl-4)), (x1+tw+6, y1), lbl_colour, -1)
            tc = (0, 0, 0) if sum(lbl_colour) > 380 else (255, 255, 255)
            cv2.putText(final_frame, label, (x1+3, y1-bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)

            frame_data["detections"].append({
                "class_id": obj.get("stable_class_id", 0),
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "track_id": tid,
                "state": state,
                "seg_was_skipped": obj.get("seg_was_skipped", False),
                "frames_since_seen": obj.get("frames_since_seen", 0),
            })

        return final_frame, frame_data


def _draw_dashed_rect(img, pt1, pt2, color, thickness, gap):
    """Draw a dashed rectangle on *img*."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)
