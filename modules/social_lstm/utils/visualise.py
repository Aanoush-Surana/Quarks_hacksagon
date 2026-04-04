"""
Visualise predicted trajectories on video frames.

Usage
─────
python utils/visualise.py \
    --video        data/videoplayback.mp4 \
    --predictions  predictions.json       \
    --output       output_video.mp4       \
    [--pixels_per_metre 10.0]             \
    [--show_samples]
"""

import json
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python")


# ── Colour palette per class ─────────────────────────────────────────────────

CLASS_COLOURS = {
    "car":        (0,   200, 255),   # amber
    "truck":      (0,   128, 255),   # orange
    "bus":        (0,    64, 200),   # dark orange
    "pedestrian": (0,   255, 128),   # green
    "bicycle":    (255, 200,   0),   # cyan-ish
    "motorcycle": (200, 100, 255),   # pink
    "unknown":    (180, 180, 180),   # grey
}

PRED_COLOUR   = (0, 255, 255)    # yellow for predicted path
OBS_COLOUR    = (255, 255, 255)  # white for observed history
SAMPLE_COLOUR = (100, 100, 255)  # light blue for samples


def metric_to_pixel(xy_metric: np.ndarray, ppm: float) -> np.ndarray:
    """Convert metric (x, y) back to pixel coordinates using ppm scale."""
    return (xy_metric * ppm).astype(np.int32)


def draw_trajectory(
    frame:          np.ndarray,
    points_px:      np.ndarray,          # (T, 2) pixel coords
    colour:         tuple,
    radius:         int   = 4,
    thickness:      int   = 2,
    alpha:          float = 1.0,
) -> np.ndarray:
    """Draw a polyline trajectory on the frame."""
    overlay = frame.copy() if alpha < 1.0 else frame

    for i, pt in enumerate(points_px):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(overlay, (x, y), radius, colour, -1)
        if i > 0:
            px0, py0 = int(points_px[i-1, 0]), int(points_px[i-1, 1])
            cv2.line(overlay, (px0, py0), (x, y), colour, thickness)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
    return overlay


def visualise(args):
    with open(args.predictions) as f:
        data = json.load(f)

    pred_by_frame = {}
    for win in data.get("predictions", []):
        sf = win["start_frame"]
        pred_by_frame[sf] = win["agents"]

    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    ppm    = args.pixels_per_metre
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in pred_by_frame:
            for agent in pred_by_frame[frame_idx]:
                cname  = agent.get("class_name", "unknown")
                colour = CLASS_COLOURS.get(cname, CLASS_COLOURS["unknown"])

                # Observed history
                obs_xy  = np.array(agent["obs_xy"],   dtype=np.float32)
                obs_px  = metric_to_pixel(obs_xy, ppm)
                frame   = draw_trajectory(frame, obs_px, OBS_COLOUR, radius=3)

                # Predicted mean
                pred_xy = np.array(agent["pred_mean"], dtype=np.float32)
                pred_px = metric_to_pixel(pred_xy, ppm)
                frame   = draw_trajectory(frame, pred_px, PRED_COLOUR, radius=5, thickness=2)

                # Stochastic samples
                if args.show_samples and "samples" in agent:
                    for sample in agent["samples"]:
                        s_px = metric_to_pixel(np.array(sample, dtype=np.float32), ppm)
                        frame = draw_trajectory(frame, s_px, SAMPLE_COLOUR,
                                                radius=2, thickness=1, alpha=0.4)

                # Track label
                tid    = agent["track_id"]
                if len(obs_px) > 0:
                    tx, ty = int(obs_px[-1, 0]), int(obs_px[-1, 1])
                    cv2.putText(
                        frame, f"{cname} #{tid}",
                        (tx + 6, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA
                    )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[visualise] Saved annotated video → '{args.output}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay Social LSTM predictions on video")
    p.add_argument("--video",             required=True)
    p.add_argument("--predictions",       required=True)
    p.add_argument("--output",            default="output_with_predictions.mp4")
    p.add_argument("--pixels_per_metre",  type=float, default=10.0)
    p.add_argument("--show_samples",      action="store_true",
                   help="Draw stochastic trajectory samples as well")
    visualise(p.parse_args())
