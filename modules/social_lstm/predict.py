"""
Inference: run a trained Social LSTM on BotSORT output.

Usage
─────
python predict.py \
    --checkpoint  checkpoints/best.pt         \
    --botsort_json data/videoplayback_tracks.json \
    --output_json  predictions.json           \
    [--pixels_per_metre 10.0]                 \
    [--n_samples 20]                          \
    [--stride 5]

Output format
─────────────
{
  "metadata": { ... copied from BotSORT ... },
  "predictions": [
    {
      "start_frame": 0,
      "agents": [
        {
          "track_id":   "42",
          "class_name": "car",
          "obs_xy":   [[x0,y0], ...],          # OBS_LEN points (metric)
          "pred_mean":[[x0,y0], ...],          # PRED_LEN mean predictions
          "pred_std": [[sx0,sy0], ...],        # per-step std deviations
          "samples":  [[[x,y],...], ...],      # n_samples stochastic trajectories
        }, ...
      ]
    }, ...
  ]
}
"""

import os
import json
import argparse
import numpy as np
import torch

from data.botsort_adapter import BotSORTAdapter, CoordConverter
from models.social_lstm    import SocialLSTM, sample_trajectories


OBS_LEN  = 16
PRED_LEN = 12


def load_model(checkpoint_path: str, device: torch.device) -> SocialLSTM:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {})

    model = SocialLSTM(
        embedding_dim      = cfg.get("embedding_dim",  64),
        hidden_dim         = cfg.get("hidden_dim",    128),
        pred_len           = cfg.get("pred_len",       12),
        dropout            = 0.0,   # disabled at inference
        grid_size          = cfg.get("grid_size",       8),
        neighbourhood_size = cfg.get("nb_size",       32.0),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[predict] Loaded model from '{checkpoint_path}' "
          f"(epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def run_inference(
    model:       SocialLSTM,
    adapter:     BotSORTAdapter,
    device:      torch.device,
    n_samples:   int = 1,
    stride:      int = 1,
) -> list:
    """
    Returns a list of prediction dicts, one per inference window.
    """
    windows = adapter.get_inference_windows(stride=stride)
    if not windows:
        print("[predict] No inference windows found — check track lengths.")
        return []

    results = []

    for win in windows:
        obs_np  = win["obs"]     # (OBS_LEN, max_agents, 2)
        mask_np = win["mask"]    # (max_agents,)
        mean_xy = win["mean_xy"] # (2,) subtracted if normalise=True

        obs  = torch.from_numpy(obs_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)

        mu, sigma, rho = model(obs, mask)   # each (PRED_LEN, A, ...)

        # Sample stochastic trajectories
        if n_samples > 1:
            traj_samples = sample_trajectories(mu, sigma, rho, n_samples)
            # (n_samples, PRED_LEN, A, 2) → de-normalise
            traj_samples = traj_samples.cpu().numpy()
        else:
            traj_samples = None

        mu_np    = mu.cpu().numpy()     # (PRED_LEN, A, 2)
        sigma_np = sigma.cpu().numpy()  # (PRED_LEN, A, 2)

        agents_out = []
        for slot in range(obs_np.shape[1]):
            if not mask_np[slot]:
                continue

            # De-normalise (add back scene mean)
            obs_xy   = obs_np[:, slot, :]  + mean_xy        # (OBS_LEN, 2)
            pred_mean = mu_np[:, slot, :] + mean_xy         # (PRED_LEN, 2)
            pred_std  = sigma_np[:, slot, :]                 # (PRED_LEN, 2)

            agent_dict = {
                "track_id":   win["track_ids"][slot],
                "class_name": win["class_names"][slot],
                "obs_xy":     obs_xy.tolist(),
                "pred_mean":  pred_mean.tolist(),
                "pred_std":   pred_std.tolist(),
            }

            if traj_samples is not None:
                # shape: (n_samples, PRED_LEN, 2)
                s = traj_samples[:, :, slot, :] + mean_xy[np.newaxis, np.newaxis, :]
                agent_dict["samples"] = s.tolist()

            agents_out.append(agent_dict)

        results.append({
            "start_frame": win["start_frame"],
            "agents":      agents_out,
        })

    print(f"[predict] Generated predictions for {len(results)} windows.")
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device = {device}")

    # ── Coordinate converter ────────────────────────────────────────────────
    homography = None
    if args.homography:
        homography = np.array(json.load(open(args.homography))).reshape(3, 3)

    conv = CoordConverter(
        homography       = homography,
        pixels_per_metre = args.pixels_per_metre,
    )

    # ── Load BotSORT JSON ───────────────────────────────────────────────────
    adapter = BotSORTAdapter(
        json_path       = args.botsort_json,
        coord_converter = conv,
        gap_fill        = args.gap_fill,
        max_agents      = args.max_agents,
        normalise       = True,
    )
    print(adapter.summary())

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Inference ───────────────────────────────────────────────────────────
    predictions = run_inference(
        model     = model,
        adapter   = adapter,
        device    = device,
        n_samples = args.n_samples,
        stride    = args.stride,
    )

    # ── Save output ─────────────────────────────────────────────────────────
    # Read original metadata from BotSORT JSON
    with open(args.botsort_json) as f:
        raw = json.load(f)

    output = {
        "metadata":    raw.get("metadata", {}),
        "model":       args.checkpoint,
        "obs_len":     OBS_LEN,
        "pred_len":    PRED_LEN,
        "n_samples":   args.n_samples,
        "predictions": predictions,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[predict] Saved → '{args.output_json}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Social LSTM inference on BotSORT output")
    p.add_argument("--checkpoint",        required=True,  help="Path to trained .pt checkpoint")
    p.add_argument("--botsort_json",      required=True,  help="Path to BotSORT output JSON")
    p.add_argument("--output_json",       default="predictions.json")
    p.add_argument("--pixels_per_metre",  type=float, default=10.0,
                   help="Fallback scale if no homography given")
    p.add_argument("--homography",        default=None,
                   help="Path to JSON file with a 3×3 homography matrix (flattened)")
    p.add_argument("--n_samples",         type=int, default=1,
                   help="Stochastic trajectory samples (1 = mean only)")
    p.add_argument("--stride",            type=int, default=1,
                   help="Frame stride between inference windows")
    p.add_argument("--gap_fill",          type=int, default=5,
                   help="Max missing frames to interpolate per track")
    p.add_argument("--max_agents",        type=int, default=32)
    main(p.parse_args())
