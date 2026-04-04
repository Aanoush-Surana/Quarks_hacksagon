import sys
import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import time

# Ensure imports work
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.segmentation.inference import SegmentationModel
from modules.botsort_module.tracker import BotSortTracker
from modules.botsort_module.schema import DetectionRecord


# -------------------------------
# CONFIG
# -------------------------------
VIDEO_PATH   = r"../../Data/inputs/videoplayback.mp4"
WEIGHTS_PATH = r"../../weights/best.pt"
OUTPUT_VIDEO = "live_tracked_output_clean.mp4"


def main():

    if not Path(VIDEO_PATH).exists():
        print("Video not found")
        return
        
    if not Path(WEIGHTS_PATH).exists():
        print("Weights not found")
        return

    print("Initialising YOLOv8 Segmentation Model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg_model = SegmentationModel(
        weights_path=WEIGHTS_PATH,
        conf_thresh=0.25,
        device=device
    )
    
    print("Initialising BoT-SORT Tracker...")
    tracker = BotSortTracker(
        fps=30.0,
        with_reid=False
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Track state
    track_history = defaultdict(list)
    unique_ids = set()

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(10000, 3), dtype=np.uint8)
    
    frame_idx = 0
    print("Starting Tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.perf_counter()
        
        # -------------------------------
        # 1. SEGMENTATION
        # -------------------------------
        seg_frame, frame_data = seg_model.process_frame(
            frame, frame_idx, frame_idx / fps
        )
        
        # -------------------------------
        # 2. FORMAT DETECTIONS
        # -------------------------------
        detections = []

        for d_id, d in enumerate(frame_data["detections"]):
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            
            detections.append(
                DetectionRecord(
                    detection_id=d_id,
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xywh=[cx, cy, w, h],
                    mask_polygon=d["mask_polygon"],
                    mask_area_px=0,
                    track_id=None
                )
            )
            
        # -------------------------------
        # 3. TRACKING
        # -------------------------------
        assigned_pairs = tracker.update(detections, frame_img=frame)

        for det_idx, tid in assigned_pairs:
            detections[det_idx].track_id = tid
            unique_ids.add(tid)

        # -------------------------------
        # 4. VISUALIZATION (WITH TRAJECTORY)
        # -------------------------------
        rendered_frame = seg_frame.copy()

        for det in detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            cx, cy = map(int, det.center)

            color = tuple(int(c) for c in colors[tid % 10000])

            # Draw bounding box
            cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name} #{tid}"
            cv2.putText(rendered_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw center
            cv2.circle(rendered_frame, (cx, cy), 3, color, -1)

            # -------------------------------
            # TRAJECTORY (kept)
            # -------------------------------
            track_history[tid].append((cx, cy))

            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            traj = track_history[tid]
            for i in range(1, len(traj)):
                cv2.line(rendered_frame, traj[i-1], traj[i], color, 2)

        # FPS
        fps_live = 1.0 / max((time.perf_counter() - t0), 0.001)
        cv2.putText(rendered_frame, f"FPS: {fps_live:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tracking", cv2.resize(rendered_frame, (1280, 720)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(rendered_frame)

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx:04d} | FPS: {fps_live:.1f}")

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -------------------------------
    # FINAL COUNT
    # -------------------------------
    print("\n==============================")
    print(f"Total unique objects tracked: {len(unique_ids)}")
    print("==============================")


if __name__ == "__main__":
    main()