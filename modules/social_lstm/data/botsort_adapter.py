"""
BotSORT → shared trajectory format adapter.

Converts the per-frame detection JSON produced by your BotSORT pipeline
into the same scene / window representation used by ArgoverseLoader,
so the trained Social LSTM can run inference directly on your data.

Key challenges handled here
────────────────────────────
1. Pixel coordinates → metric world coordinates
   BotSORT gives bbox_xyxy / bbox_xywh in pixel space.
   We project via a homography (if provided) or use a simple scale
   (pixels-per-metre) fallback.

2. Fragmented tracks
   Tracks may disappear and reappear. We interpolate short gaps (≤ gap_fill
   frames) and split tracks at longer gaps.

3. Same windowing as training
   We expose get_inference_windows() which returns (obs, mask) pairs
   at the same OBS_LEN / PRED_LEN cadence.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple


OBS_LEN  = 16
PRED_LEN = 12
SEQ_LEN  = OBS_LEN + PRED_LEN


# ── Coordinate conversion ────────────────────────────────────────────────────

class CoordConverter:
    """
    Convert pixel bbox centres to metric (x, y) world coordinates.

    Priority:
      1. If a 3×3 homography matrix H is given, use  xw = H @ [px, py, 1]ᵀ
      2. Otherwise fall back to  xw = px / ppm,  yw = py / ppm
         where ppm = pixels_per_metre.

    Parameters
    ----------
    homography      : (3,3) numpy array mapping image → ground plane (metric)
    pixels_per_metre: fallback scale (pixels per metre)
    frame_h, frame_w: frame resolution (for optional sanity checks)
    """

    def __init__(
        self,
        homography: Optional[np.ndarray] = None,
        pixels_per_metre: float = 10.0,
        frame_h: int = 1080,
        frame_w: int = 1920,
    ):
        self.H   = homography        # (3,3) or None
        self.ppm = pixels_per_metre
        self.frame_h = frame_h
        self.frame_w = frame_w

    def convert(self, px: float, py: float) -> Tuple[float, float]:
        """Convert a single pixel point to metric world coordinates."""
        if self.H is not None:
            pt = self.H @ np.array([px, py, 1.0])
            return float(pt[0] / pt[2]), float(pt[1] / pt[2])
        return px / self.ppm, py / self.ppm


# ── BotSORT JSON loader ──────────────────────────────────────────────────────

class BotSORTAdapter:
    """
    Load a BotSORT JSON file and expose trajectory windows for inference.

    Parameters
    ----------
    json_path        : path to the BotSORT output JSON
    coord_converter  : CoordConverter instance (pixel → metric)
    gap_fill         : max consecutive missing frames to interpolate per track
    min_obs_frames   : minimum frames a track must be observed to be included
    max_agents       : pad/clip agent dimension to this number
    normalise        : zero-centre coords using mean of obs window
    classes_keep     : if set, only keep detections of these class names
                       e.g. {"car", "truck", "pedestrian", "bicycle"}
    """

    def __init__(
        self,
        json_path: str,
        coord_converter: Optional[CoordConverter] = None,
        gap_fill: int = 5,
        min_obs_frames: int = OBS_LEN // 2,
        max_agents: int = 32,
        normalise: bool = True,
        classes_keep: Optional[set] = None,
    ):
        self.conv          = coord_converter or CoordConverter()
        self.gap_fill      = gap_fill
        self.min_obs_frames = min_obs_frames
        self.max_agents    = max_agents
        self.normalise     = normalise
        self.classes_keep  = classes_keep

        with open(json_path, "r") as f:
            raw = json.load(f)

        meta = raw.get("metadata", {})
        self.fps         = float(meta.get("fps", 30.0))
        self.frame_count = int(meta.get("frame_count", 0))
        self.resolution  = meta.get("resolution", [1920, 1080])

        # Update converter resolution from metadata
        if hasattr(self.conv, "frame_w"):
            self.conv.frame_w = self.resolution[0]
            self.conv.frame_h = self.resolution[1]

        # Parse all frames into per-track dicts
        self.tracks: Dict[int, dict] = {}          # track_id → {frames, xy, class}
        self._parse_frames(raw.get("frames", []))
        self._fill_gaps()

    # ── parsing ─────────────────────────────────────────────────────────────

    def _parse_frames(self, frames_list: List[dict]):
        for frame_dict in frames_list:
            fid = int(frame_dict["frame_id"])
            for det in frame_dict.get("detections", []):
                tid   = int(det["track_id"])
                cname = str(det.get("class_name", "unknown")).lower()

                if self.classes_keep and cname not in self.classes_keep:
                    continue

                # Bbox centre in pixels
                bx, by, bw, bh = det["bbox_xywh"]
                px, py = float(bx), float(by)  # centre already

                wx, wy = self.conv.convert(px, py)

                if tid not in self.tracks:
                    self.tracks[tid] = {
                        "class_name": cname,
                        "frames": [],
                        "xy":     [],
                    }

                self.tracks[tid]["frames"].append(fid)
                self.tracks[tid]["xy"].append([wx, wy])

        # Convert to numpy arrays
        for tid in self.tracks:
            t = self.tracks[tid]
            order = np.argsort(t["frames"])
            t["frames"] = np.array(t["frames"], dtype=np.int64)[order]
            t["xy"]     = np.array(t["xy"],     dtype=np.float32)[order]

    def _fill_gaps(self):
        """Linear interpolation over short missing-frame gaps."""
        for tid, t in self.tracks.items():
            frames = t["frames"]
            xy     = t["xy"]
            if len(frames) < 2:
                continue

            new_frames, new_xy = [frames[0]], [xy[0]]
            for i in range(1, len(frames)):
                gap = int(frames[i] - frames[i - 1]) - 1
                if 0 < gap <= self.gap_fill:
                    for g in range(1, gap + 1):
                        alpha = g / (gap + 1)
                        interp_f = int(frames[i - 1]) + g
                        interp_xy = (1 - alpha) * xy[i - 1] + alpha * xy[i]
                        new_frames.append(interp_f)
                        new_xy.append(interp_xy)
                new_frames.append(frames[i])
                new_xy.append(xy[i])

            t["frames"] = np.array(new_frames, dtype=np.int64)
            t["xy"]     = np.array(new_xy,     dtype=np.float32)

    # ── public API ───────────────────────────────────────────────────────────

    def get_scene(self) -> dict:
        """
        Return the entire video as a single scene dict
        (compatible with ArgoverseLoader scene format).
        """
        agents = []
        for tid, t in self.tracks.items():
            if len(t["frames"]) < self.min_obs_frames:
                continue
            agents.append({
                "agent_id":  str(tid),
                "class_name": t["class_name"],
                "frames":    t["frames"],
                "xy":        t["xy"],
            })
        return {"scene_id": "botsort_video", "agents": agents}

    def get_inference_windows(
        self,
        start_frame: Optional[int] = None,
        end_frame:   Optional[int] = None,
        stride: int = 1,
    ) -> List[dict]:
        """
        Produce overlapping OBS_LEN windows for inference (no ground-truth pred).

        Each window dict:
          {
            "start_frame" : int,
            "obs"         : (OBS_LEN, max_agents, 2)  float32 – metric coords
            "mask"        : (max_agents,)              bool
            "track_ids"   : [str, ...]  – track IDs for each agent slot
            "class_names" : [str, ...]
            "mean_xy"     : (2,) float32  – subtracted if normalise=True
          }
        """
        # Determine global frame range
        all_frames = sorted(set(
            int(f)
            for t in self.tracks.values()
            for f in t["frames"]
        ))
        if not all_frames:
            return []

        f_min = start_frame if start_frame is not None else all_frames[0]
        f_max = end_frame   if end_frame   is not None else all_frames[-1]

        # Build per-track lookup
        lookup: Dict[int, Dict[int, np.ndarray]] = {}
        for tid, t in self.tracks.items():
            lookup[tid] = {int(f): xy for f, xy in zip(t["frames"], t["xy"])}

        windows = []
        frame_list = [f for f in all_frames if f_min <= f <= f_max]

        for i in range(0, max(1, len(frame_list) - OBS_LEN + 1), stride):
            obs_frames = frame_list[i: i + OBS_LEN]
            if len(obs_frames) < OBS_LEN:
                break

            # Agents present in ALL obs frames
            present_ids = [
                tid for tid in self.tracks
                if all(f in lookup[tid] for f in obs_frames)
                and len(self.tracks[tid]["frames"]) >= self.min_obs_frames
            ]
            if not present_ids:
                continue

            present_ids = present_ids[: self.max_agents]
            n = len(present_ids)

            obs       = np.zeros((OBS_LEN,  self.max_agents, 2), dtype=np.float32)
            mask      = np.zeros((self.max_agents,),              dtype=bool)
            track_ids = [""] * self.max_agents
            cnames    = [""] * self.max_agents

            for slot, tid in enumerate(present_ids):
                obs[:, slot, :] = np.stack([lookup[tid][f] for f in obs_frames])
                mask[slot]       = True
                track_ids[slot]  = str(tid)
                cnames[slot]     = self.tracks[tid]["class_name"]

            mean_xy = np.array([0.0, 0.0], dtype=np.float32)
            if self.normalise:
                mean_xy = obs[:, mask, :].mean(axis=(0, 1))
                obs    -= mean_xy[np.newaxis, np.newaxis, :]

            windows.append({
                "start_frame": obs_frames[0],
                "obs":         obs,
                "mask":        mask,
                "track_ids":   track_ids,
                "class_names": cnames,
                "mean_xy":     mean_xy,
            })

        return windows

    def summary(self) -> str:
        lines = [
            f"BotSORT video  : fps={self.fps}, frames={self.frame_count}",
            f"Total tracks   : {len(self.tracks)}",
        ]
        for tid, t in sorted(self.tracks.items())[:8]:
            lines.append(
                f"  track {tid:4d} | class={t['class_name']:12s} | "
                f"frames={len(t['frames'])}"
            )
        if len(self.tracks) > 8:
            lines.append(f"  ... and {len(self.tracks)-8} more")
        return "\n".join(lines)
