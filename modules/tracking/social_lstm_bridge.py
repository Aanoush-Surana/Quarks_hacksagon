import numpy as np
import torch
from collections import deque
from modules.social_lstm.data.botsort_adapter import CoordConverter

class SocialLSTMBridge:
    def __init__(self, obs_len=16, max_agents=8, pixels_per_metre=10.0):
        self.obs_len = obs_len
        self.max_agents = max_agents
        
        # We reuse CoordConverter to handle pixel->world conversions.
        self.converter = CoordConverter(homography=None, pixels_per_metre=pixels_per_metre)
        
        # History lookup: track_id -> deque of (frame_idx, px_x, px_y, metric_x, metric_y)
        self.history = {}

    def update_and_get_window(self, frame_idx, detections):
        """
        Takes current frame detections, updates the sliding window, and if sufficient data
        exists, returns the formatted tensors for Social LSTM inference.
        
        Args:
            frame_idx (int): Current frame number.
            detections (list): List of dicts representing observed tracks from temporal fusion.
        
        Returns:
            dict: If ready, returns dict with keys 'obs' and 'mask' representing tensors, and 
                  context dict containing the mean metric and track_ids. Otherwise returns None.
        """
        current_tids = []

        for det in detections:
            tid = det.get("track_id")
            if tid is None or det.get("state") == "stuff":
                continue
            
            bbox = det.get("bbox")
            if bbox is None:
                continue

            # Pixel Center
            px = (bbox[0] + bbox[2]) / 2.0
            py = (bbox[1] + bbox[3]) / 2.0

            # World Metric
            mx, my = self.converter.convert(px, py)

            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.obs_len)

            self.history[tid].append({
                "frame_idx": frame_idx,
                "px": (px, py),
                "metric": (mx, my)
            })
            current_tids.append(tid)

        # Cleanup stale tracks
        for tid in list(self.history.keys()):
            if tid not in current_tids:
                if len(self.history[tid]) == 0 or frame_idx - self.history[tid][-1]["frame_idx"] > 5:
                    del self.history[tid]

        # Filter tracks that have exactly `obs_len` continuous elements in their buffer terminating AT `frame_idx`.
        # (This enables the sliding window behavior at every single frame).
        ready_tids = []
        for tid in current_tids:
            track = self.history[tid]
            if len(track) == self.obs_len:
                ready_tids.append(tid)

        if len(ready_tids) == 0:
            return None

        # Gather 'obs' metrics
        # Sort ready_tids for deterministic slots
        ready_tids = ready_tids[:self.max_agents] 
        num_agents = len(ready_tids)

        obs = np.zeros((self.obs_len, self.max_agents, 2), dtype=np.float32)
        mask = np.zeros((self.max_agents,), dtype=bool)

        for slot_idx, tid in enumerate(ready_tids):
            track_seq = list(self.history[tid])
            for t_idx, state in enumerate(track_seq):
                obs[t_idx, slot_idx, 0] = state["metric"][0]
                obs[t_idx, slot_idx, 1] = state["metric"][1]
            mask[slot_idx] = True

        # Normalization (zero-center using scene mean of observed track window)
        scene_mean = obs[:, mask, :].mean(axis=(0, 1), keepdims=False)  # Shape (2,)
        obs -= scene_mean[np.newaxis, np.newaxis, :]  # Broadcasting to (obs_len, max_agents, 2)

        return {
            "obs": torch.from_numpy(obs),    # (16, 8, 2)
            "mask": torch.from_numpy(mask),  # (8)
            "context": {
                "track_ids": ready_tids,
                "scene_mean": scene_mean,
            }
        }

    def convert_predictions_to_pixel(self, pred_mu, context):
        """
        Takes prediction mean back to pixel space.
        
        Args:
            pred_mu (tensor/numpy): Shape (pred_len, max_agents, 2)
            context (dict): Dictionary with 'scene_mean' metric and 'track_ids'.
            
        Returns:
            dict: Mapping of track_id to a list of (px_x, px_y) trajectory points. 
        """
        if isinstance(pred_mu, torch.Tensor):
            pred_mu = pred_mu.cpu().numpy()

        results = {}
        for slot_idx, tid in enumerate(context["track_ids"]):
            # Denormalize metric coordinates
            pred_metric = pred_mu[:, slot_idx, :] + context["scene_mean"]
            
            # Revert to pixel coordinates
            pixel_traj = []
            for step in range(pred_metric.shape[0]):
                mx, my = pred_metric[step]
                # Inverse of px/ppm = mx => px = mx * ppm
                px = mx * self.converter.ppm
                py = my * self.converter.ppm
                pixel_traj.append((int(px), int(py)))
                
            results[tid] = pixel_traj

        return results
