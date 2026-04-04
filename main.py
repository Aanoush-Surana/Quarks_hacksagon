import os
import sys
import yaml
import logging
import argparse
import time
import cv2
import json
import threading
import torch
from queue import Queue, Empty
from pathlib import Path

from modules.preprocess.cleaner import Preprocessor
from modules.segmentation.inference import SegmentationModel
from modules.tracking.tracker import TrackerModule

# Temporal Fusion pipeline
from modules.temporal_fusion import (
    TemporalMaskFusion,
    DetectionPrefilter,
    extract_detections_from_result,
)
from modules.temporal_fusion.class_stabilizer import ClassStabilizer
from modules.temporal_fusion.mask_postprocessor import project_and_fill


ENABLE_PREPROCESSING = False
# --- CONFIGURATION FLAG ---
# Set to True to display stream in real-time (Do not save as video)
# Set to False to save output as video (Do not display real-time stream)
SHOW_REALTIME_STREAM = True
# --------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineManager")

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
USE_PREPROCESSING = False
# ==========================================

def ensure_dirs(config):
    """Ensure that all data directories exist."""
    dirs_to_create = [
        config['paths'].get('output_preprocess', 'data/outputs/preprocess'),
        config['paths'].get('output_segmentation', 'data/outputs/segmentation'),
        config['paths'].get('output_tracking', 'data/outputs/tracking'),
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Proceeding with defaults.")
        return {
            'paths': {
                'default_video_input': "data/inputs/sample_1.mp4",
                'default_weights': "weights/best.pt",
                'output_preprocess': "data/outputs/preprocess",
                'output_segmentation': "data/outputs/segmentation",
                'output_tracking': "data/outputs/tracking"
            },
            'pipeline': {
                'use_preprocessing': True,
                'preprocess_resolution': [640, 640],
                'conf_thresh': 0.25,
                'iou_thresh': 0.45
            }
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AsyncPipeline:
    def __init__(self, video_path, weights_path, config, show_stream=True):
        self.config = config
        self.video_path = video_path
        self.weights_path = weights_path
        self.show_stream = show_stream
        
        ensure_dirs(config)
        
        # Output paths
        vid_name = Path(video_path).stem
        self.out_vid_path = os.path.join(config['paths']['output_tracking'], f"{vid_name}_final.mp4")
        self.out_json_path = os.path.join(config['paths']['output_tracking'], f"{vid_name}_results.json")
        
        # Modules
        res = config['pipeline'].get('preprocess_resolution', [640, 640])
        self.preprocessor = Preprocessor(
            target_resolution=tuple(res),
            enabled=ENABLE_PREPROCESSING
        )
        
        conf_th = config['pipeline'].get('conf_thresh', 0.25)
        iou_th = config['pipeline'].get('iou_thresh', 0.45)
        self.seg_model = SegmentationModel(
            weights_path=weights_path, 
            conf_thresh=conf_th, 
            iou_thresh=iou_th
        )
        self.tracker = TrackerModule()
        
        # Temporal Fusion modules
        self.fusion = TemporalMaskFusion()
        self.prefilter = DetectionPrefilter()
        self.stabilizer = ClassStabilizer()
        logger.info("Temporal Fusion pipeline initialised.")
        
        # Queues
        self.capture_queue = Queue(maxsize=15)
        self.inference_queue = Queue(maxsize=15)
        self.display_queue = Queue(maxsize=1) # Latest frame only for display
        self.json_queue = Queue()
        
        # Control flags
        self.running = True
        self.capture_done = False
        self.inference_done = False
        
        self.start_time = 0
        self.frames_processed = 0
        self.total_frames = 0
        self.video_fps = 30.0
        self._last_log_count = -1

    def capture_thread(self):
        """Thread 1: Manages Video Capture (No frame skipping)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_path}")
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_fps = fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        logger.info(f"Capture Thread: {self.video_path} ({self.total_frames} frames @ {fps:.1f} FPS)")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Block and wait if the queue is full (NO FRAME SKIPPING)
            timestamp = frame_idx / float(fps)
            self.capture_queue.put((frame_idx, timestamp, frame))
            frame_idx += 1
            
        self.capture_done = True
        cap.release()
        logger.info("Capture thread finished.")

    def inference_thread(self):
        """Thread 2: Temporal-Fusion inference pipeline.

        Per-frame order:
          1. Preprocess
          2. YOLO detect + BoT-SORT  (raw result)
          3. extract_detections_from_result
          4. DetectionPrefilter  → clean_dets, suppressed_dets
          5. Seg-skip check
          6. TemporalMaskFusion.update
          7. ClassStabilizer.stabilize
          8. project_and_fill
          9. render_fusion_outputs
         10. Stats / periodic cleanup
        """
        logger.info("Starting Temporal-Fusion Inference Thread.")
        
        while self.running:
            try:
                item = self.capture_queue.get(timeout=0.5)
                frame_idx, timestamp, frame = item
            except Empty:
                if self.capture_done: break
                continue
            
            # 1. Preprocess
            proc_frame = self.preprocessor.process_frame(frame)
            H, W = proc_frame.shape[:2]
            t0 = time.perf_counter()
            
            # 2. YOLO detection + BoT-SORT tracking (raw result)
            raw_result = self.seg_model.detect(proc_frame)
            
            # 3. Extract detections
            raw_dets = extract_detections_from_result(raw_result, self.seg_model.model)
            
            # 4. Prefilter — split into clean (→ tracker) + suppressed (→ fusion only) + stuff
            fusion_states = self.fusion.get_states()
            clean_dets, suppressed_dets, stuff_dets = self.prefilter.filter(
                raw_dets, frame_idx, fusion_states
            )
            
            # 5. Seg-skip check
            skip_ids = self.fusion.get_seg_skip_set(frame_idx)
            
            # 6. Fusion update
            outputs = self.fusion.update(
                clean_dets, suppressed_dets, (H, W), frame_idx,
                skip_ids=skip_ids, stuff_detections=stuff_dets
            )
            
            # 7. Class stabilise
            for det in clean_dets:
                tid = det.get("track_id")
                if tid is not None:
                    s_cid, s_cname = self.stabilizer.stabilize(
                        tid, det["class_id"], det["class_name"], det["confidence"]
                    )
                    if tid in outputs:
                        outputs[tid]["stable_class_name"] = s_cname
                        outputs[tid]["stable_class_id"] = s_cid
            
            # 8. Mask post-process (hole-fill + full-frame projection)
            outputs = project_and_fill(outputs, (H, W))
            
            # 9. Render
            seg_frame, frame_data = self.seg_model.render_fusion_outputs(
                proc_frame, outputs, frame_idx, timestamp
            )
            
            # 10. Stats aggregation (TrackerModule for unique-object counts)
            tracked_frame = self.tracker.process_frame(seg_frame, frame_data)
            inference_ms = (time.perf_counter() - t0) * 1000
            
            if "tracking_stats" in frame_data:
                frame_data["tracking_stats"]["inference_ms"] = round(inference_ms, 1)
            
            # 11. Periodic diagnostics (every 50 frames)
            if frame_idx > 0 and frame_idx % 50 == 0:
                vram_str = ""
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    vram_str = f" | VRAM: {vram_mb:.0f}MB"
                logger.info(
                    f"[Frame {frame_idx}] Fusion: {self.fusion.get_metrics()} | "
                    f"Stability: {self.stabilizer.get_stability_report()} | "
                    f"Flicker: {self.prefilter.get_flicker_stats()}{vram_str}"
                )
            
            # 12. Periodic cleanup (every 100 frames)
            if frame_idx > 0 and frame_idx % 100 == 0:
                active_ids = {
                    det.get("track_id")
                    for det in clean_dets
                    if det.get("track_id") is not None
                }
                self.fusion.cleanup(active_ids, frame_idx)
                # Reset stabilizer for dropped tracks
                for tid in list(self.stabilizer._tracks.keys()):
                    if tid not in active_ids and tid not in self.fusion._state:
                        self.stabilizer.reset(tid)
            
            # Push to display/writer queue and json queue
            self.inference_queue.put((frame_idx, tracked_frame))
            self.json_queue.put(frame_data)
            
            # Non-blocking display queue (latest frame + stats)
            if self.display_queue.empty():
                self.display_queue.put((frame_idx, tracked_frame, frame_data))
                
            self.capture_queue.task_done()
            self.frames_processed += 1
            
        self.inference_done = True
        logger.info("Inference thread finished.")

    def output_thread(self):
        """Thread 3: Background Video Writing."""
        if not self.show_stream:
            res = self.config['pipeline'].get('preprocess_resolution', [640, 640])
            W, H = tuple(res)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(self.out_vid_path, fourcc, 30.0, (W, H))
            logger.info(f"Starting Background Writer Thread. Path: {self.out_vid_path}")
        else:
            logger.info("Background Writer disabled. Real-time stream display is ON.")
        
        while self.running:
            try:
                item = self.inference_queue.get(timeout=0.5)
                _, frame = item
            except Empty:
                if self.inference_done: break
                continue
            
            if not self.show_stream:
                out_vid.write(frame)
            self.inference_queue.task_done()
            
        if not self.show_stream:
            out_vid.release()
        logger.info("Writer thread finished.")

    def json_thread(self):
        """Thread 4: Asynchronous JSON Data flow."""
        all_results = []
        logger.info("Starting JSON Metadata Thread.")
        
        while self.running:
            try:
                data = self.json_queue.get(timeout=0.5)
            except Empty:
                if self.inference_done: break
                continue
            
            all_results.append(data)
            self.json_queue.task_done()
            
        with open(self.out_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"JSON Logger finished. Path: {self.out_json_path}")

    def _draw_hud(self, frame, frame_idx, fps, stats):
        """Draw a semi-transparent diagnostic HUD panel on the frame."""
        h, w = frame.shape[:2]

        active   = stats.get("active_tracked", 0)
        unique   = stats.get("total_unique_objects", 0)
        det_cnt  = stats.get("detections_this_frame", 0)
        inf_ms   = stats.get("inference_ms", 0)
        cls_map  = stats.get("class_counts", {})

        lines = [
            f"FPS: {fps:.1f}  |  Frame: {frame_idx}/{self.total_frames}",
            f"Active: {active}  |  Unique Objects: {unique}  |  Det/Frame: {det_cnt}",
            f"Inference: {inf_ms:.0f}ms  |  Source: {self.video_fps:.0f}fps",
        ]
        if cls_map:
            cls_str = "  ".join(f"{k}: {v}" for k, v in sorted(cls_map.items()))
            lines.append(f"Classes: {cls_str}")

        # Semi-transparent dark panel
        line_h, pad = 22, 8
        panel_h = len(lines) * line_h + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, txt in enumerate(lines):
            y = pad + (i + 1) * line_h - 4
            cv2.putText(frame, txt, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 255, 200), 1, cv2.LINE_AA)

    def run_pipeline(self):
        logger.info("-" * 40)
        logger.info("STARTING ASYNC STREAMING PIPELINE")
        logger.info("-" * 40)
        
        # Start Threads
        threads = [
            threading.Thread(target=self.capture_thread, name="Capture"),
            threading.Thread(target=self.inference_thread, name="Inference"),
            threading.Thread(target=self.output_thread, name="Writer"),
            threading.Thread(target=self.json_thread, name="JSON")
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
            
        # Display Loop in Main Thread
        self.start_time = time.time()
        
        try:
            while self.running:
                # Check for termination condition
                if self.inference_done and self.inference_queue.empty() and self.display_queue.empty():
                    break
                    
                if self.show_stream:
                    try:
                        frame_idx, frame, frame_data = self.display_queue.get(timeout=0.01)
                        
                        elapsed = time.time() - self.start_time
                        fps = self.frames_processed / elapsed if elapsed > 0 else 0
                        stats = frame_data.get("tracking_stats", {})
                        
                        # Draw full diagnostic HUD
                        self._draw_hud(frame, frame_idx, fps, stats)
                        
                        cv2.imshow("Pipeline Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested termination.")
                            self.running = False
                    except Empty:
                        pass
                else:
                    time.sleep(0.1)

                # Periodic console log (deduplicated, every 100 frames)
                if (self.frames_processed > 0
                        and self.frames_processed != self._last_log_count
                        and self.frames_processed % 100 == 0):
                    self._last_log_count = self.frames_processed
                    elapsed = time.time() - self.start_time
                    fps = self.frames_processed / elapsed
                    logger.info(
                        f"[{self.frames_processed}/{self.total_frames}] "
                        f"FPS: {fps:.1f} | Q_cap: {self.capture_queue.qsize()} "
                        f"| Q_inf: {self.inference_queue.qsize()}"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.running = False
            for t in threads:
                try:
                    t.join(timeout=1.0)
                except (KeyboardInterrupt, Exception):
                    pass
            cv2.destroyAllWindows()
            
        logger.info("-" * 40)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
        logger.info("-" * 40)

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    parser = argparse.ArgumentParser(description="High-Performance YOLOv8 Streaming Pipeline.")
    parser.add_argument("--video_path", type=str, default=config["paths"]["default_video_input"],
                        help="Path to the raw input video.")
    parser.add_argument("--weights_path", type=str, default=config["paths"]["default_weights"],
                        help="Path to the YOLOv8 weights (.pt file).")
    
    args = parser.parse_args()
    
    pipeline = AsyncPipeline(args.video_path, args.weights_path, config, show_stream=SHOW_REALTIME_STREAM)
    pipeline.run_pipeline()
