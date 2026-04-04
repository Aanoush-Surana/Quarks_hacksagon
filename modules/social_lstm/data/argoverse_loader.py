"""
Argoverse dataset loader for Social LSTM training.

Argoverse 1 motion forecasting format:
  Each CSV has columns: TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME
  Sequences are 5-second windows at 10 Hz (50 frames).
  We use the first 2 s (20 frames) as observation and predict the next 3 s (30 frames).
  For our use-case we re-window to obs=16 frames (1.6 s) + pred=12 frames (1.2 s) at 10 Hz.
"""

import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


# ── Shared trajectory dict schema ──────────────────────────────────────────────
# Each "scene" is a dict:
# {
#   "scene_id": str,
#   "agents": [
#       {
#           "agent_id": str,
#           "class_name": str,          # "VEHICLE", "PEDESTRIAN", etc.
#           "frames": np.ndarray,       # (T,) int  – frame indices (absolute)
#           "xy":    np.ndarray,        # (T, 2) float – world coords in metres
#       }, ...
#   ]
# }
# ──────────────────────────────────────────────────────────────────────────────


OBS_LEN  = 16   # observation frames  (1.6 s @ 10 Hz)
PRED_LEN = 12   # prediction frames   (1.2 s @ 10 Hz)
SEQ_LEN  = OBS_LEN + PRED_LEN


class ArgoverseLoader(Dataset):
    """
    Loads Argoverse 1 motion-forecasting CSVs and returns
    (obs_seq, pred_seq, agent_mask) tuples ready for Social LSTM.

    Parameters
    ----------
    data_dir   : path to the split folder, e.g.  .../train/data/
    skip       : stride between sampled windows (default 1 = every frame)
    min_agents : minimum co-present agents to keep a window
    max_agents : pad / clip agent dimension to this number
    normalise  : subtract scene-mean so coords are zero-centred
    """

    def __init__(
        self,
        data_dir: str,
        skip: int = 1,
        min_agents: int = 1,
        max_agents: int = 32,
        normalise: bool = True,
        max_files=2000,
    ):
        super().__init__()
        self.data_dir   = data_dir
        self.skip       = skip
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.normalise  = normalise
        self.max_files = max_files

        self.sequences: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._load_all()

    # ── internal ────────────────────────────────────────────────────────────

    def _load_all(self):
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

    # ── ADD THIS: limit files for quick testing ──────────────────────────
        if hasattr(self, 'max_files') and self.max_files:
            csv_files = csv_files[:self.max_files]
    # ─────────────────────────────────────────────────────────────────────

        for i, csv_path in enumerate(csv_files):
        # Progress every 500 files
            if i % 500 == 0:
             print(f"  Loading files... {i}/{len(csv_files)}", flush=True)
            scenes = self._parse_csv(csv_path)
            for scene in scenes:
                windows = self._extract_windows(scene)
                self.sequences.extend(windows)

        print(f"[ArgoverseLoader] Loaded {len(self.sequences)} windows "
            f"from {len(csv_files)} CSV files in '{self.data_dir}'")
    def _parse_csv(self, csv_path: str) -> List[dict]:
        """Return a list of scene dicts (one per CSV for Argoverse 1)."""
        df = pd.read_csv(csv_path)
        df = df.sort_values("TIMESTAMP").reset_index(drop=True)

        # Normalise timestamps to 0-indexed frame numbers
        ts_unique = sorted(df["TIMESTAMP"].unique())
        ts_to_frame = {t: i for i, t in enumerate(ts_unique)}
        df["frame"] = df["TIMESTAMP"].map(ts_to_frame)

        agents = []
        for track_id, grp in df.groupby("TRACK_ID"):
            grp = grp.sort_values("frame")
            agents.append({
                "agent_id":  str(track_id),
                "class_name": grp["OBJECT_TYPE"].iloc[0],
                "frames":    grp["frame"].to_numpy(dtype=np.int64),
                "xy":        grp[["X", "Y"]].to_numpy(dtype=np.float32),
            })

        scene_id = os.path.splitext(os.path.basename(csv_path))[0]
        return [{"scene_id": scene_id, "agents": agents}]

    def _extract_windows(
        self, scene: dict
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Slide a SEQ_LEN window over the scene.

        Returns list of (obs, pred, mask):
          obs   : (OBS_LEN,  max_agents, 2)
          pred  : (PRED_LEN, max_agents, 2)
          mask  : (max_agents,) bool  – which agent slots are real
        """
        agents = scene["agents"]
        if not agents:
            return []

        # Global frame range
        all_frames = sorted(set(
            f for a in agents for f in a["frames"]
        ))
        if len(all_frames) < SEQ_LEN:
            return []

        # Build dense lookup: agent_id → (frame → xy)
        lookup = {}
        for a in agents:
            lookup[a["agent_id"]] = dict(zip(a["frames"], a["xy"]))

        windows = []
        for start in range(0, len(all_frames) - SEQ_LEN + 1, self.skip):
            window_frames = all_frames[start: start + SEQ_LEN]
            obs_frames    = window_frames[:OBS_LEN]
            pred_frames   = window_frames[OBS_LEN:]

            # Agents present for ALL obs frames
            present = [
                a for a in agents
                if all(f in lookup[a["agent_id"]] for f in obs_frames)
            ]
            if len(present) < self.min_agents:
                continue

            # Clip to max_agents (focal agent first if available)
            present = present[: self.max_agents]
            n = len(present)

            obs  = np.zeros((OBS_LEN,  self.max_agents, 2), dtype=np.float32)
            pred = np.zeros((PRED_LEN, self.max_agents, 2), dtype=np.float32)
            mask = np.zeros((self.max_agents,),             dtype=bool)

            for i, a in enumerate(present):
                aid = a["agent_id"]
                obs[:, i, :]  = np.stack([lookup[aid][f] for f in obs_frames])
                mask[i]       = True
                # Pred frames may be missing for some agents → zero-pad
                pred_xy = np.array([
                    lookup[aid].get(f, [np.nan, np.nan]) for f in pred_frames
                ], dtype=np.float32)
                pred[:, i, :] = pred_xy

            if self.normalise:
                # Zero-centre using mean of obs positions of real agents
                mean = obs[:, mask, :].mean(axis=(0, 1), keepdims=False)  # (2,)
                obs  -= mean[np.newaxis, np.newaxis, :]
                pred -= mean[np.newaxis, np.newaxis, :]

            windows.append((obs, pred, mask))

        return windows

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        obs, pred, mask = self.sequences[idx]
        import torch
        return (
            torch.from_numpy(obs),
            torch.from_numpy(pred),
            torch.from_numpy(mask),
        )


# ── Convenience helper ───────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack variable-count batches; mask already pads to max_agents."""
    import torch
    obs   = torch.stack([b[0] for b in batch])   # (B, T_obs,  A, 2)
    pred  = torch.stack([b[1] for b in batch])   # (B, T_pred, A, 2)
    mask  = torch.stack([b[2] for b in batch])   # (B, A)
    return obs, pred, mask
