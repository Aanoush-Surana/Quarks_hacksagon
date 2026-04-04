import sys
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
VIDEO_PATH   = r"../../Data/inputs/sample_1.mp4"
WEIGHTS_PATH = r"../../weights/best.pt"
OUTPUT_VIDEO = "live_tracked_output_clean.mp4"

SHOW = True
TRACK_EVERY = 2
MAX_DETS = 50

# Only track movable objects
MOVABLE_CLASSES = {
    "car", "truck", "bus",
    "motorcycle", "bicycle",
    "autorickshaw", "rider", "person"
}


# -------------------------------
# FAST NMS
# -------------------------------
def fast_nms(detections, iou_thresh=0.5):
    if len(detections) == 0:
        return []

    boxes = []
    scores = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(det.confidence))

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_thresh)

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return [detections[i] for i in indices]


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = BotSortTracker(fps=fps, with_reid=False)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg_model = SegmentationModel(
        weights_path=WEIGHTS_PATH,
        conf_thresh=0.35,
        device=device
    )

    track_history = defaultdict(list)
    last_known_positions = {}
    unique_ids = set()
    id_frame_count = defaultdict(int)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(10000, 3), dtype=np.uint8)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        seg_frame, frame_data = seg_model.process_frame(
            frame, frame_idx, frame_idx / fps
        )

        # -------------------------------
        # DETECTIONS (FILTERED)
        # -------------------------------
        detections = []

        for d_id, d in enumerate(frame_data["detections"]):

            # 🔥 filter only movable classes
            if d["class_name"] not in MOVABLE_CLASSES:
                continue

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

        # limit detections
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)[:MAX_DETS]

        # NMS
        detections = fast_nms(detections, 0.5)

        # -------------------------------
        # TRACKING
        # -------------------------------
        if frame_idx % TRACK_EVERY == 0:
            assigned_pairs = tracker.update(detections, frame_img=frame)
        else:
            assigned_pairs = []

        for det_idx, tid in assigned_pairs:
            tid = int(tid)
            detections[det_idx].track_id = tid

            unique_ids.add(tid)
            id_frame_count[tid] += 1

            cx, cy = map(int, detections[det_idx].center)
            last_known_positions[tid] = (cx, cy)

        # -------------------------------
        # UPDATE TRAJECTORIES
        # -------------------------------
        for det in detections:
            if det.track_id is not None:
                tid = det.track_id
                cx, cy = map(int, det.center)

                track_history[tid].append((cx, cy))
                last_known_positions[tid] = (cx, cy)

                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)

        # -------------------------------
        # DRAW TRACKING FRAME
        # -------------------------------
        rendered_frame = seg_frame.copy()

        for tid, traj in track_history.items():
            color = tuple(int(c) for c in colors[tid % 10000])

            for i in range(1, len(traj)):
                cv2.line(rendered_frame, traj[i-1], traj[i], color, 2)

            if tid in last_known_positions:
                cx, cy = last_known_positions[tid]
                cv2.circle(rendered_frame, (cx, cy), 3, color, -1)

        for det in detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)

            color = tuple(int(c) for c in colors[tid % 10000])


            cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), color, 2)

            life = id_frame_count[tid]
            label = f"{det.class_name} #{tid} ({life})"

            cv2.putText(rendered_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -------------------------------
        # SIDE-BY-SIDE VIEW
        # -------------------------------
        combined = np.hstack((
            cv2.resize(frame, (width, height)),           # raw
            cv2.resize(rendered_frame, (width, height))   # tracking
        ))

        fps_live = 1.0 / max((time.perf_counter() - t0), 0.001)

        if SHOW:
            cv2.putText(combined, f"FPS: {fps_live:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Raw vs Tracking", cv2.resize(combined, (1280, 720)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.write(rendered_frame)

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total unique IDs: {len(unique_ids)}")


if __name__ == "__main__":
    main()