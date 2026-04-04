import os
import cv2
import json
import logging
import numpy as np
import time
import torch
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

        self.logger.info(f"Loading YOLOv8 model from {weights_path}")
        
        # TensorRT Optimization Check
        weights_file = Path(weights_path)
        engine_path = weights_file.with_suffix('.engine')
        
        if engine_path.exists():
            self.logger.info(f"Loading TensorRT engine from {engine_path}")
            self.model = YOLO(str(engine_path), task='segment')
        else:
            self.logger.info(f"Loading standard YOLOv8 model from {weights_path}")
            self.model = YOLO(weights_path)
            # Export to TensorRT if CUDA is available
            if 'cuda' in str(device).lower():
                try:
                    self.logger.info("Attempting to export model to TensorRT (this may take a few minutes)...")
                    # half=True for FP16 optimization during export
                    exported_path = self.model.export(format='engine', device=device, half=True)
                    self.logger.info(f"TensorRT export successful: {exported_path}")
                    # Reload the engine
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

    def process_frame(self, frame, frame_idx, timestamp_sec):
        """
        Optimized segmentation processing (no accuracy loss)
        """
        H, W = frame.shape[:2]

        # Inference
        # Predict with FP16 and Retina Masks (High-Res)
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=640,
            device=self.device,
            retina_masks=True,
            half=True,  # FP16 Inference
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
            
            # Label background and text
            label = f"{cls_name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(final_frame, (x1, max(0, y1-th-bl-4)), (x1+tw+6, y1), bgr_colour, -1)
            tc = (0,0,0) if sum(bgr_colour) > 380 else (255,255,255)
            cv2.putText(final_frame, label, (x1+3, y1-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
            
            # Store data for JSON
            poly_points = mask_segments[i].tolist() if i < len(mask_segments) else []
            frame_data["detections"].append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "mask_polygon": poly_points
            })

        return final_frame, frame_data
