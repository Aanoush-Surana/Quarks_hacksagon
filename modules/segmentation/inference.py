import os
import cv2
import json
import logging
import numpy as np
import time
from ultralytics import YOLO

class SegmentationModel:
    def __init__(self, weights_path, device=0, conf_thresh=0.25, iou_thresh=0.45):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        self.logger.info(f"Loading YOLOv8 model from {weights_path}")
        self.model = YOLO(weights_path)
        self.names = self.model.names
        
        # Colour palette for rendering (RGB format converted to BGR below if needed)
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
        # Return BGR
        c = self._PALETTE[class_id % len(self._PALETTE)]
        return (c[2], c[1], c[0]) 

    def process_frame(self, frame, frame_idx, timestamp_sec):
        """
        Runs segmentation on a single frame.
        Returns the rendered frame and the JSON data for this frame.
        """
        H, W = frame.shape[:2]

        # Predict
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=640,
            device=self.device,
            retina_masks=True,
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

        # Rendering Overlay
        overlay = frame.copy().astype(np.float32)

        if masks is not None and boxes is not None:
            mask_segments = masks.xy 
            for i, (box, mask) in enumerate(zip(boxes, masks.data.cpu().numpy())):
                cls_id = int(box.cls[0].item())
                cls_name = self.names.get(cls_id, f"cls_{cls_id}")
                conf = float(box.conf[0].item())
                bgr_colour = self.get_colour(cls_id, cls_name)
                
                # Ensure mask matches frame dimensions
                msk = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                alpha = 0.60 if cls_name == self.DRIVABLE_NAME else 0.45
                
                overlay[msk] = (
                    overlay[msk] * (1 - alpha) +
                    np.array(bgr_colour, dtype=np.float32) * alpha
                )

                overlay_u8 = overlay.astype(np.uint8)
                contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_u8, contours, -1, bgr_colour, 2)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(overlay_u8, (x1, y1), (x2, y2), bgr_colour, 2)
                
                # Text
                label = f"{cls_name} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay_u8, (x1, max(0, y1-th-bl-4)), (x1+tw+6, y1), bgr_colour, -1)
                tc = (0,0,0) if sum(bgr_colour) > 380 else (255,255,255)
                cv2.putText(overlay_u8, label, (x1+3, y1-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
                
                overlay = overlay_u8.astype(np.float32)
                
                # Store data for JSON
                poly_points = []
                if i < len(mask_segments):
                    poly_points = mask_segments[i].tolist()
                    
                frame_data["detections"].append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "mask_polygon": poly_points
                })

        final_frame = overlay.astype(np.uint8)
        
        return final_frame, frame_data
