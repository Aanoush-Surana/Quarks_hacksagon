import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

from modules.preprocess.cleaner import Preprocessor
from modules.segmentation.inference import SegmentationModel
from modules.tracking.tracker import TrackerModule

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

import cv2
import json

def run_pipeline(video_path, weights_path, config):
    ensure_dirs(config)

    logger.info("-" * 40)
    logger.info("STARTING STREAMING PIPELINE")
    logger.info("-" * 40)
    
    if not os.path.exists(video_path):
        logger.error(f"Input video not found: {video_path}")
        sys.exit(1)
        
    if not os.path.exists(weights_path):
        logger.error(f"Model weights not found: {weights_path}")
        sys.exit(1)

    # Output paths
    vid_name = Path(video_path).stem
    out_vid_path = os.path.join(config['paths']['output_tracking'], f"{vid_name}_final.mp4")
    out_json_path = os.path.join(config['paths']['output_tracking'], f"{vid_name}_results.json")

    # Initialize Modules
    res = config['pipeline'].get('preprocess_resolution', [640, 640])
    preprocessor = Preprocessor(target_resolution=tuple(res))
    
    conf_th = config['pipeline'].get('conf_thresh', 0.25)
    iou_th = config['pipeline'].get('iou_thresh', 0.45)
    seg_model = SegmentationModel(weights_path=weights_path, conf_thresh=conf_th, iou_thresh=iou_th)
    
    tracker = TrackerModule()

    # Setup Video Stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = tuple(res)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (W, H))

    all_tracking_data = []

    frame_idx = 0
    logger.info(f"Processing {total_frames} frames from {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
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

        out_vid.write(tracked_frame)
        all_tracking_data.append(frame_data)
        
        # Display the frame in a window
        cv2.imshow("Segmentation Stream", tracked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Early termination requested by user.")
            break
        
        if frame_idx % 30 == 0:
            logger.info(f"Processed frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
    
    # Save JSON results
    with open(out_json_path, 'w') as f:
        json.dump(all_tracking_data, f, indent=2)

    logger.info("-" * 40)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info(f"Output video: {out_vid_path}")
    logger.info(f"Output JSON: {out_json_path}")
    logger.info("-" * 40)


if __name__ == "__main__":
    config = load_config("config.yaml")
    
    parser = argparse.ArgumentParser(description="Run the YOLOv8 Segmentation & Tracking Pipeline.")
    parser.add_argument("--video_path", type=str, default=config["paths"]["default_video_input"],
                        help="Path to the raw input video.")
    parser.add_argument("--weights_path", type=str, default=config["paths"]["default_weights"],
                        help="Path to the YOLOv8 weights (.pt file).")
    
    args = parser.parse_args()
    
    run_pipeline(args.video_path, args.weights_path, config)
