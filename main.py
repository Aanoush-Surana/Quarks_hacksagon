import os
import sys
import yaml
import logging
import argparse
import time
import cv2
import json
import threading
from queue import Queue, Empty
from pathlib import Path

from modules.preprocess.cleaner import Preprocessor
from modules.segmentation.inference import SegmentationModel
from modules.tracking.tracker import TrackerModule


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
USE_PREPROCESSING = False   # Change to False to bypass preprocessing
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
            device='cuda', # Forced CUDA
            conf_thresh=conf_th, 
            iou_thresh=iou_th
        )
        self.tracker = TrackerModule()
        
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

    def capture_thread(self):
        """Thread 1: Manages Video Capture (No frame skipping)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_path}")
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        
        logger.info(f"Starting Capture Thread from: {self.video_path}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
<<<<<<< HEAD
            # Block and wait if the queue is full (NO FRAME SKIPPING)
            timestamp = frame_idx / float(fps)
            self.capture_queue.put((frame_idx, timestamp, frame))
            frame_idx += 1
            
        self.capture_done = True
        cap.release()
        logger.info("Capture thread finished.")
=======
        timestamp_sec = frame_idx / float(fps)
        
        # 1. Preprocess (Conditional)
        proc_frame = preprocessor.process_frame(frame) if USE_PREPROCESSING else frame
        
        # 2. Segmentation
        seg_frame, frame_data = seg_model.process_frame(proc_frame, frame_idx, timestamp_sec)
        
        # 3. Tracking
        tracked_frame = tracker.process_frame(seg_frame, frame_data)
        
        # Overlays
        cv2.putText(tracked_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
>>>>>>> e1cba7dbfc0e1ba780cb2ebfdfbfc20b0a60bbd5

    def inference_thread(self):
        """Thread 2: Performs Inference, Preprocessing, and Rendering."""
        logger.info("Starting High-Performance Inference Thread.")
        
        while self.running:
            try:
                # Wait for a frame from capture queue
                item = self.capture_queue.get(timeout=0.5)
                frame_idx, timestamp, frame = item
            except Empty:
                if self.capture_done: break
                continue
            
            # 1. Preprocess
            proc_frame = self.preprocessor.process_frame(frame)
            
            # 2. Segmentation (Optimized with FP16/CUDA/TensorRT)
            seg_frame, frame_data = self.seg_model.process_frame(proc_frame, frame_idx, timestamp)
            
            # 3. Tracking (Placeholder logic)
            tracked_frame = self.tracker.process_frame(seg_frame, frame_data)
            
            # Push to display/writer queue and json queue
            self.inference_queue.put((frame_idx, tracked_frame))
            self.json_queue.put(frame_data)
            
            # Non-blocking display queue (update only if empty to show latest)
            if self.display_queue.empty():
                self.display_queue.put((frame_idx, tracked_frame))
                
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
                    # Real-time display from display_queue
                    try:
                        frame_idx, frame = self.display_queue.get(timeout=0.01)
                        
                        # Display Stats Overlay
                        elapsed = time.time() - self.start_time
                        fps = self.frames_processed / elapsed if elapsed > 0 else 0
                        cv2.putText(frame, f"FPS: {fps:.2f} | Frame: {frame_idx}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow("High-Performance Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested termination.")
                            self.running = False
                    except Empty:
                        pass
                else:
                    time.sleep(0.1) # Sleep to avoid maxing out CPU in main thread

                if self.frames_processed % 30 == 0 and self.frames_processed > 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_processed / elapsed
                    logger.info(f"Performance: {fps:.2f} FPS | Q_cap: {self.capture_queue.qsize()} | Q_inf: {self.inference_queue.qsize()}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.running = False
            for t in threads:
                t.join(timeout=1.0)
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
