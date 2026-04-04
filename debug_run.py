"""
debug_run.py -- Single-threaded diagnostic runner.
Answers Q1-Q7 by inspecting pipeline internals at each frame.
"""

import os, sys, cv2, yaml, numpy as np, torch

# Force UTF-8 and unbuffered output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from modules.preprocess.cleaner import Preprocessor
from modules.segmentation.inference import SegmentationModel
from modules.temporal_fusion import (
    TemporalMaskFusion,
    DetectionPrefilter,
    extract_detections_from_result,
)
from modules.temporal_fusion.class_stabilizer import ClassStabilizer
from modules.temporal_fusion.mask_postprocessor import project_and_fill

DEBUG_CLASSES = {"sidewalk", "vegetation", "fallback", "pole"}
DEBUG_FRAMES_Q3 = {1, 5, 10, 20}
MAX_FRAMES = 25

config = yaml.safe_load(open("config.yaml"))
video_path = config["paths"]["default_video_input"]
weights_path = config["paths"]["default_weights"]

log_lines = []

def LOG(msg):
    log_lines.append(msg)
    print(msg, flush=True)

LOG("=" * 80)
LOG("DEBUG RUN -- answering Q1-Q7")
LOG(f"  video : {video_path}")
LOG(f"  weights: {weights_path}")
LOG(f"  max frames: {MAX_FRAMES}")
LOG("=" * 80)

# Initialise modules
seg_model = SegmentationModel(
    weights_path=weights_path, conf_thresh=0.25, iou_thresh=0.45
)
fusion = TemporalMaskFusion()
prefilter = DetectionPrefilter()
stabilizer = ClassStabilizer()
preprocessor = Preprocessor(target_resolution=(640, 640), enabled=False)

# Q6
LOG("")
LOG("=" * 80)
LOG(f"[Q6] posterior_threshold = {fusion.posterior_threshold}")
LOG(f"     (default in __init__, temporal_fusion_core.py line 59)")
LOG("=" * 80)

# Q7 static
LOG("")
LOG("=" * 80)
LOG(f"[Q7] Stuff classes in DetectionPrefilter : {sorted(prefilter.stuff_classes)}")
LOG(f"     Stuff classes in TemporalMaskFusion : {sorted(fusion.stuff_classes)}")
LOG(f"     'sidewalk' in stuff_classes : {'sidewalk' in prefilter.stuff_classes}")
LOG(f"     'pole'     in stuff_classes : {'pole' in prefilter.stuff_classes}")
LOG(f"     'vegetation' in stuff_classes: {'vegetation' in prefilter.stuff_classes}")
LOG(f"     'fallback'   in stuff_classes: {'fallback' in prefilter.stuff_classes}")
LOG(f"     -> sidewalk & pole go through INSTANCE (Bayesian posterior) path")
LOG(f"     -> vegetation & fallback go through STUFF BYPASS path")
LOG("=" * 80)
LOG("")

# Frame loop
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    LOG(f"ERROR: cannot open {video_path}")
    sys.exit(1)

frame_idx = 0

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    proc_frame = preprocessor.process_frame(frame)
    H, W = proc_frame.shape[:2]

    # YOLO detect
    raw_result = seg_model.detect(proc_frame)

    # Q2: raw mask tensor BEFORE any conversion
    if raw_result.masks is not None and raw_result.boxes is not None:
        raw_tensor_all = raw_result.masks.data
        names = seg_model.model.names
        for i in range(len(raw_result.boxes)):
            cls_id = int(raw_result.boxes[i].cls[0].item())
            cls_name = names.get(cls_id, f"cls_{cls_id}")
            if cls_name in DEBUG_CLASSES:
                t = raw_tensor_all[i]
                LOG(
                    f"[Q2] frame={frame_idx:3d}  class={cls_name:12s}  "
                    f"raw min={t.min().item():.6f}  max={t.max().item():.6f}  "
                    f"dtype={t.dtype}"
                )

    # Q1: after extract_detections_from_result
    raw_dets = extract_detections_from_result(raw_result, seg_model.model)
    for det in raw_dets:
        cn = det["class_name"]
        if cn in DEBUG_CLASSES:
            m = det["mask"]
            LOG(
                f"[Q1] frame={frame_idx:3d}  class={cn:12s}  "
                f"mask min={m.min()}  max={m.max()}  mean={m.mean():.4f}  "
                f"shape={m.shape}"
            )

    # Prefilter
    fusion_states = fusion.get_states()
    clean_dets, suppressed_dets, stuff_dets = prefilter.filter(
        raw_dets, frame_idx, fusion_states
    )

    # Q7 runtime
    for det in clean_dets:
        if det["class_name"] in {"sidewalk", "pole"}:
            LOG(
                f"[Q7-rt] frame={frame_idx:3d}  class={det['class_name']:12s}  "
                f"-> CLEAN (Bayesian posterior path)"
            )
    for det in suppressed_dets:
        if det["class_name"] in {"sidewalk", "pole"}:
            LOG(
                f"[Q7-rt] frame={frame_idx:3d}  class={det['class_name']:12s}  "
                f"-> SUPPRESSED (low-trust Bayesian)"
            )
    for det in stuff_dets:
        if det["class_name"] in {"sidewalk", "pole", "vegetation", "fallback"}:
            LOG(
                f"[Q7-rt] frame={frame_idx:3d}  class={det['class_name']:12s}  "
                f"-> STUFF (bypass, no Bayesian)"
            )

    # Q3 prep: save pre-update posterior means
    q3_pre = {}
    if frame_idx in DEBUG_FRAMES_Q3:
        for det in clean_dets + suppressed_dets:
            tid = det.get("track_id")
            cn = det.get("class_name")
            if cn in DEBUG_CLASSES and tid is not None:
                if tid in fusion._state:
                    q3_pre[tid] = {
                        "class_name": cn,
                        "pre_mean": float(fusion._state[tid]["posterior"].mean()),
                        "alpha": min(0.95, max(0.3, det.get("confidence", 0.0))),
                    }
                else:
                    q3_pre[tid] = {
                        "class_name": cn,
                        "pre_mean": 0.5,
                        "alpha": min(0.95, max(0.3, det.get("confidence", 0.0))),
                    }

    # Seg-skip + Fusion update
    skip_ids = fusion.get_seg_skip_set(frame_idx)
    outputs = fusion.update(
        clean_dets, suppressed_dets, (H, W), frame_idx,
        skip_ids=skip_ids,
        stuff_detections=stuff_dets,
    )

    # Q3 post
    if frame_idx in DEBUG_FRAMES_Q3:
        for tid, info in q3_pre.items():
            if tid in fusion._state:
                s = fusion._state[tid]
                mask_mean = 0.0
                if s["mask_buffer"]:
                    mask_mean = float(s["mask_buffer"][-1].mean())
                post_mean = float(s["posterior"].mean())
                LOG(
                    f"[Q3] frame={frame_idx:3d}  class={info['class_name']:12s}  "
                    f"alpha={info['alpha']:.4f}  curr_mask_mean={mask_mean:.4f}  "
                    f"pre_post_mean={info['pre_mean']:.4f}  "
                    f"post_post_mean={post_mean:.4f}"
                )
        for det in stuff_dets:
            if det["class_name"] in DEBUG_CLASSES:
                LOG(
                    f"[Q3] frame={frame_idx:3d}  class={det['class_name']:12s}  "
                    f"-> STUFF bypass, no Bayesian update"
                )

    # Class stabilise
    for det in clean_dets:
        tid = det.get("track_id")
        if tid is not None:
            s_cid, s_cname = stabilizer.stabilize(
                tid, det["class_id"], det["class_name"], det["confidence"]
            )
            if tid in outputs:
                outputs[tid]["stable_class_name"] = s_cname
                outputs[tid]["stable_class_id"] = s_cid

    # Q4 pre
    for tid, obj in outputs.items():
        cn = obj.get("stable_class_name", "")
        if cn in DEBUG_CLASSES:
            tmb = obj.get("temporal_mask_binary_64")
            if tmb is not None:
                LOG(
                    f"[Q4-pre] frame={frame_idx:3d}  class={cn:12s}  "
                    f"tmb64 min={tmb.min()}  max={tmb.max()}  "
                    f"mean={tmb.mean():.4f}"
                )
            else:
                LOG(
                    f"[Q4-pre] frame={frame_idx:3d}  class={cn:12s}  "
                    f"tmb64=None (stuff bypass)"
                )

    outputs = project_and_fill(outputs, (H, W))

    # Q4 post
    for tid, obj in outputs.items():
        cn = obj.get("stable_class_name", "")
        if cn in DEBUG_CLASSES:
            ffm = obj.get("full_frame_mask")
            if ffm is not None:
                LOG(
                    f"[Q4-post] frame={frame_idx:3d}  class={cn:12s}  "
                    f"full_frame_mask mean={ffm.mean():.4f}"
                )
            else:
                LOG(
                    f"[Q4-post] frame={frame_idx:3d}  class={cn:12s}  "
                    f"full_frame_mask=None"
                )

    # Q5
    for tid, obj in outputs.items():
        cn = obj.get("stable_class_name", "")
        if cn in DEBUG_CLASSES:
            ffm = obj.get("full_frame_mask")
            if ffm is not None:
                LOG(
                    f"[Q5] frame={frame_idx:3d}  class={cn:12s}  "
                    f"ffm_mean={ffm.mean():.4f} (before overlay)"
                )

    seg_frame, frame_data = seg_model.render_fusion_outputs(
        proc_frame, outputs, frame_idx, frame_idx / 30.0
    )

    frame_idx += 1
    if frame_idx % 5 == 0:
        LOG(f"--- processed {frame_idx} frames ---")

cap.release()

LOG(f"\n{'='*80}")
LOG(f"DEBUG RUN COMPLETE -- processed {frame_idx} frames")
LOG(f"{'='*80}")

# Write all lines to a results file (UTF-8)
with open("debug_results.txt", "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"\nResults also saved to debug_results.txt", flush=True)
