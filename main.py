import os
import yaml
import logging
import argparse
import time
import cv2
import threading
import numpy as np
from queue import Queue, Empty, Full
from pathlib import Path

from modules.preprocess.cleaner import Preprocessor
from modules.segmentation.inference import SegmentationModel
from modules.tracking.tracker import TrackerModule
from modules.temporal_fusion import TemporalMaskFusion, DetectionPrefilter
from modules.temporal_fusion.class_stabilizer import ClassStabilizer
from modules.temporal_fusion.mask_postprocessor import project_and_fill
from modules.temporal_fusion.helpers import extract_detections_from_result


# 🔴 IMPORTANT: disable unless purely photometric
ENABLE_PREPROCESSING = False
SHOW_REALTIME_STREAM = True
RENDER_SEGMENTATION_MASKS = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineManager")

TRACK_CLASSES = {
    "car", "bus", "truck", "motorcycle",
    "bicycle", "autorickshaw", "rider", "person"
}


def ensure_dirs(config):
    os.makedirs(config['paths'].get('output_tracking', 'data/outputs/tracking'), exist_ok=True)


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {
            'paths': {
                'default_video_input': "data/inputs/sample_1.mp4",
                'default_weights': "weights/best.pt",
                'output_tracking': "data/outputs/tracking"
            }
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_segmentation_masks(frame, outputs, seg_model):
    """Draws colored alpha-blended segmentation masks on the frame."""
    overlay = frame.astype(np.float32)
    has_masks = False
    
    for tid, obj in outputs.items():
        full_mask = obj.get("full_frame_mask")
        if full_mask is None:
            continue
            
        msk = full_mask > 127
        if msk.sum() < 50:
            continue
            
        has_masks = True
        cls_name = obj.get("stable_class_name", "unknown")
        bgr_colour = seg_model.get_colour(obj.get("stable_class_id", 0), cls_name)
        
        state = obj.get("state", "visible")
        if state == "stuff" or state == "hallucinated":
            alpha = 0.25
        elif cls_name in (seg_model.DRIVABLE_NAME, "drivable area"):
            alpha = 0.60
        else:
            alpha = 0.45
            
        overlay[msk] = overlay[msk] * (1 - alpha) + np.array(bgr_colour, dtype=np.float32) * alpha

    return overlay.astype(np.uint8) if has_masks else frame.copy()


class AsyncPipeline:
    def __init__(self, video_path, weights_path, config):
        self.video_path = video_path
        self.weights_path = weights_path
        self.config = config

        ensure_dirs(config)

        self.preprocessor = Preprocessor(enabled=ENABLE_PREPROCESSING)
        self.seg_model = SegmentationModel(weights_path=weights_path)
        self.tracker = TrackerModule()
        self.fusion = TemporalMaskFusion()
        self.prefilter = DetectionPrefilter()
        self.stabilizer = ClassStabilizer()

        self.capture_queue = Queue(maxsize=15)
        self.display_queue = Queue(maxsize=1)

        self.running = True
        self.capture_done = False
        self.trajectories = {}

    # -----------------------
    # CAPTURE
    # -----------------------
    def capture_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self.capture_queue.put((frame_idx, frame_idx / fps, frame), timeout=1)
                frame_idx += 1
            except Full:
                continue

        self.capture_done = True
        cap.release()

    # -----------------------
    # SAFE COPY
    # -----------------------
    def _safe_copy_frame_data(self, frame_data):
        new_data = dict(frame_data)
        new_data["detections"] = [
            d.copy() for d in frame_data.get("detections", [])
        ]
        return new_data

    # -----------------------
    # INFERENCE
    # -----------------------
    def inference_thread(self):
        DETECTION_INTERVAL = 2
        last_frame_data = None

        while self.running:
            try:
                frame_idx, ts, frame = self.capture_queue.get(timeout=0.5)
            except Empty:
                if self.capture_done:
                    break
                continue

            proc_frame = self.preprocessor.process_frame(frame)
            orig_frame = frame

            H, W = orig_frame.shape[:2]
            t0 = time.perf_counter()

            # -----------------------
            # DETECTION
            # -----------------------
            if frame_idx % DETECTION_INTERVAL == 0 or last_frame_data is None:
                result = self.seg_model.detect(proc_frame)
                detections = extract_detections_from_result(result, self.seg_model.model)
                frame_data = {
                    "frame_id": frame_idx,
                    "timestamp_sec": ts,
                    "detections": detections
                }
                last_frame_data = self._safe_copy_frame_data(frame_data)
            else:
                frame_data = self._safe_copy_frame_data(last_frame_data)
                frame_data["frame_idx"] = frame_idx
                frame_data["timestamp"] = ts

            # -----------------------
            # TRACKING (FIXED)
            # -----------------------
            self.tracker.process_frame(orig_frame, frame_data)

            if RENDER_SEGMENTATION_MASKS:
                detections = frame_data.get("detections", [])
            else:
                detections = [
                    d for d in frame_data.get("detections", [])
                    if d.get("class_name") in TRACK_CLASSES
                ]

            # -----------------------
            # PREFILTER
            # -----------------------
            clean_dets, suppressed_dets, stuff_dets = self.prefilter.filter(
                detections, frame_idx, self.fusion.get_states()
            )

            # -----------------------
            # FUSION
            # -----------------------
            outputs = self.fusion.update(
                clean_dets,
                suppressed_dets,
                (H, W),
                frame_idx,
                stuff_detections=stuff_dets
            )

            # -----------------------
            # STABILIZATION
            # -----------------------
            for det in clean_dets:
                tid = det.get("track_id")
                if tid in outputs:
                    _, cname = self.stabilizer.stabilize(
                        tid,
                        det["class_id"],
                        det["class_name"],
                        det["confidence"]
                    )
                    outputs[tid]["stable_class_name"] = cname

            # -----------------------
            # CLEANUP (NEW)
            # -----------------------
            if frame_idx % 100 == 0:
                active_ids = {d.get("track_id") for d in clean_dets if d.get("track_id") is not None}
                self.fusion.cleanup(active_ids, frame_idx)

            # -----------------------
            # OPTIONAL MASK POST-PROCESS
            # -----------------------
            if RENDER_SEGMENTATION_MASKS:
                outputs = project_and_fill(outputs, (H, W))
                final_frame = apply_segmentation_masks(orig_frame, outputs, self.seg_model)
            else:
                final_frame = orig_frame

            # -----------------------
            # TRAJECTORIES & RENDER
            # -----------------------

            for tid, obj in outputs.items():
                if obj.get("state") == "stuff":
                    continue
                    
                bbox = obj.get("bbox")
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                cls_name = obj.get("stable_class_name", "unknown")
                cls_id = obj.get("stable_class_id", 0)
                conf = obj.get("confidence", 0.0)
                state = obj.get("state", "visible")
                bgr_colour = self.seg_model.get_colour(cls_id, cls_name)
                
                # 1. Trajectories
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.trajectories.setdefault(tid, []).append((cx, cy))
                if len(self.trajectories[tid]) > 30:
                    self.trajectories[tid].pop(0)

                # 2. Contour outlines
                if RENDER_SEGMENTATION_MASKS:
                    full_mask = obj.get("full_frame_mask")
                    if full_mask is not None and full_mask.any():
                        contours, _ = cv2.findContours(
                            full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(final_frame, contours, -1, bgr_colour, 2)

                # 3. Draw BBox (class-colored)
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), bgr_colour, 2)

                # 4. Label with background
                tid_str = f"ID:{abs(tid)} " if tid is not None else ""
                label = f"{tid_str}{cls_name} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(final_frame, (x1, max(0, y1 - th - bl - 4)),
                              (x1 + tw + 6, y1), bgr_colour, -1)
                tc = (0, 0, 0) if sum(bgr_colour) > 380 else (255, 255, 255)
                cv2.putText(final_frame, label, (x1 + 3, y1 - bl),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)

                # 5. Draw Trajectory Line
                pts = self.trajectories[tid]
                for i in range(1, len(pts)):
                    cv2.line(final_frame, pts[i - 1], pts[i], (255, 0, 0), 2)

            # -----------------------
            # HUD METRICS
            # -----------------------
            inference_ms = (time.perf_counter() - t0) * 1000
            fps = 1000.0 / max(inference_ms, 1e-6)
            
            fusion_metrics = self.fusion.get_metrics()
            unique_ids = len(self.trajectories)
            
            hud_lines = [
                f"FPS: {fps:.1f} | Latency: {inference_ms:.0f}ms",
                f"Unique IDs (Total): {unique_ids}",
                f"Fusion State -> Vis: {fusion_metrics['visible_count']}  Occ: {fusion_metrics['occluded_count']}  Hall: {fusion_metrics['hallucinated_count']}"
            ]
            
            for idx, text in enumerate(hud_lines):
                cv2.putText(final_frame, text,
                            (10, 30 + (idx * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            try:
                self.display_queue.put_nowait(final_frame)
            except Full:
                pass

            self.capture_queue.task_done()

    # -----------------------
    # RUN
    # -----------------------
    def run_pipeline(self):
        t1 = threading.Thread(target=self.capture_thread)
        t2 = threading.Thread(target=self.inference_thread)

        t1.start()
        t2.start()

        try:
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=0.1)
                    cv2.imshow("Pipeline", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                except Empty:
                    if not t2.is_alive() and self.display_queue.empty():
                        break
        finally:
            self.running = False
            t1.join()
            t2.join()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=cfg["paths"]["default_video_input"])
    parser.add_argument("--weights_path", default=cfg["paths"]["default_weights"])
    args = parser.parse_args()

    AsyncPipeline(args.video_path, args.weights_path, cfg).run_pipeline()