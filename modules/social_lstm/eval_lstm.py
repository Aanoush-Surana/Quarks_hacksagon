"""
Evaluate a saved Social LSTM checkpoint on the Argoverse validation set.

Usage
─────
python evaluate.py --model_path ./checkpoints/best.pt \
                   [--val_dir  ./data/argoverse/val/data] \
                   [--max_files 100]

All model architecture flags (grid_size, hidden_dim, etc.) are automatically
recovered from the checkpoint's saved cfg, so you never get a size mismatch.
CLI values are only used as fallbacks when the checkpoint has no saved cfg.
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.argoverse_loader import ArgoverseLoader, collate_fn
from models.social_lstm    import SocialLSTM

# Re-use canonical defaults from training so nothing drifts.
from train import DEFAULTS


# ── Metrics (mirrors train.py exactly) ───────────────────────────────────────

def ade(pred_mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Average Displacement Error across all valid agents and timesteps."""
    T, A, _ = target.shape
    valid    = mask.unsqueeze(0).expand(T, A) & ~torch.isnan(target).any(-1)
    if valid.sum() == 0:
        return float("nan")
    err = torch.norm(pred_mu - target, dim=-1)
    return err[valid].mean().item()


def fde(pred_mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Final Displacement Error at the last predicted timestep."""
    last_mu  = pred_mu[-1]
    last_tgt = target[-1]
    valid    = mask & ~torch.isnan(last_tgt).any(-1)
    if valid.sum() == 0:
        return float("nan")
    err = torch.norm(last_mu - last_tgt, dim=-1)
    return err[valid].mean().item()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60, flush=True)
    print(f"[eval] Device     : {device}",            flush=True)
    print(f"[eval] Checkpoint : {cfg['model_path']}",  flush=True)
    print(f"[eval] Val dir    : {cfg['val_dir']}",     flush=True)
    print(f"[eval] Max files  : {cfg['max_files']}",   flush=True)
    print("=" * 60, flush=True)

    # ── load checkpoint first so we can recover the saved architecture ─────
    model_path = cfg["model_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"\n[ckpt] Loading checkpoint from {model_path}...", flush=True)
    # weights_only=False is explicit — we trust our own checkpoint file.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        # Patch cfg with whatever architecture was used at training time.
        # This prevents any grid_size / hidden_dim mismatch errors.
        saved_cfg = ckpt.get("cfg", {})
        arch_keys = ("embedding_dim", "hidden_dim", "pred_len",
                     "dropout", "grid_size", "nb_size")
        print("[ckpt] Recovering architecture from saved cfg:", flush=True)
        for key in arch_keys:
            if key in saved_cfg:
                if cfg.get(key) != saved_cfg[key]:
                    print(
                        f"  [ckpt] {key}: CLI/default={cfg.get(key)} "
                        f"-> overridden by checkpoint value={saved_cfg[key]}",
                        flush=True,
                    )
                cfg[key] = saved_cfg[key]
            else:
                print(
                    f"  [ckpt] {key} not found in checkpoint cfg, "
                    f"using CLI/default={cfg.get(key)}",
                    flush=True,
                )
        trained_epoch = ckpt.get("epoch", "?")
        saved_ade     = ckpt.get("best_ade", None)
        print(f"[ckpt] Checkpoint epoch : {trained_epoch}", flush=True)
        if saved_ade is not None:
            print(f"[ckpt] Best ADE at save : {saved_ade:.4f}", flush=True)
        is_full_ckpt = True
    else:
        # Raw state-dict — no cfg available, fall back to CLI/DEFAULTS entirely.
        print(
            "[ckpt] Raw state-dict detected (no 'cfg' key). "
            "Using CLI / DEFAULTS for architecture — make sure they match "
            "what the model was trained with.",
            flush=True,
        )
        is_full_ckpt = False

    # ── dataset ───────────────────────────────────────────────────────────────
    print(f"\n[data] Loading validation dataset from {cfg['val_dir']}...", flush=True)
    t0 = time.time()
    val_ds = ArgoverseLoader(
        data_dir   = cfg["val_dir"],
        skip       = cfg["skip"],
        max_agents = cfg["max_agents"],
        max_files  = cfg["max_files"],
    )
    print(f"[data] {len(val_ds)} windows loaded ({time.time()-t0:.1f}s)", flush=True)

    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = device.type == "cuda",
    )
    print(f"[data] {len(val_loader)} batches (batch_size={cfg['batch_size']})", flush=True)

    # ── build model with the (now corrected) cfg ──────────────────────────────
    print("\n[model] Building Social LSTM...", flush=True)
    print(f"  embedding_dim      = {cfg['embedding_dim']}", flush=True)
    print(f"  hidden_dim         = {cfg['hidden_dim']}",    flush=True)
    print(f"  pred_len           = {cfg['pred_len']}",      flush=True)
    print(f"  dropout            = {cfg['dropout']}",       flush=True)
    print(f"  grid_size          = {cfg['grid_size']}",     flush=True)
    print(f"  neighbourhood_size = {cfg['nb_size']}",       flush=True)

    model = SocialLSTM(
        embedding_dim      = cfg["embedding_dim"],
        hidden_dim         = cfg["hidden_dim"],
        pred_len           = cfg["pred_len"],
        dropout            = cfg["dropout"],
        grid_size          = cfg["grid_size"],
        neighbourhood_size = cfg["nb_size"],
    ).to(device)

    # ── load weights into the correctly shaped model ──────────────────────────
    if is_full_ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parameters : {n_params:,}", flush=True)
    print(f"[model] On device  : {next(model.parameters()).device}", flush=True)

    # ── sanity-check forward pass ─────────────────────────────────────────────
    print("\n[sanity] Running one forward pass before evaluation...", flush=True)
    sample_obs, _, sample_mask = val_ds[0]
    mu, sigma, rho = model(sample_obs.to(device), sample_mask.to(device))
    print(
        f"[sanity] OK — mu={tuple(mu.shape)}  "
        f"sigma={tuple(sigma.shape)}  rho={tuple(rho.shape)}",
        flush=True,
    )

    # ── evaluation loop ───────────────────────────────────────────────────────
    # The model processes one scene at a time — mirrors validate() in train.py.
    print(f"\n[eval] Evaluating {len(val_loader)} batches...\n", flush=True)
    total_ade   = 0.0
    total_fde   = 0.0
    n_scenes    = 0
    nan_skipped = 0
    eval_start  = time.time()

    for batch_idx, (obs, pred, mask) in enumerate(val_loader):
        obs  = obs.to(device)
        pred = pred.to(device)
        mask = mask.to(device)

        B = obs.size(0)
        for b in range(B):
            # Single-scene forward pass — identical to train.py's validate()
            mu, _, _ = model(obs[b], mask[b])

            a = ade(mu, pred[b], mask[b])
            f = fde(mu, pred[b], mask[b])

            if np.isnan(a) or np.isnan(f):
                nan_skipped += 1
                continue

            total_ade += a
            total_fde += f
            n_scenes  += 1

        if batch_idx % 50 == 0:
            elapsed = time.time() - eval_start
            pct     = 100.0 * batch_idx / max(len(val_loader), 1)
            print(
                f"  [eval] batch {batch_idx:5d}/{len(val_loader)} "
                f"({pct:5.1f}%) | "
                f"running ADE={total_ade/max(n_scenes,1):.4f}  "
                f"FDE={total_fde/max(n_scenes,1):.4f} | "
                f"{elapsed:.1f}s elapsed",
                flush=True,
            )

    # ── final results ─────────────────────────────────────────────────────────
    final_ade  = total_ade / max(n_scenes, 1)
    final_fde  = total_fde / max(n_scenes, 1)
    total_time = time.time() - eval_start

    print(f"\n{'='*60}", flush=True)
    print("FINAL VALIDATION RESULTS",                         flush=True)
    print(f"  Scenes evaluated : {n_scenes}",                 flush=True)
    print(f"  Scenes skipped   : {nan_skipped}  (NaN mask)",  flush=True)
    print(f"  ADE              : {final_ade:.4f} m",          flush=True)
    print(f"  FDE              : {final_fde:.4f} m",          flush=True)
    print(f"  Total time       : {total_time:.1f}s",          flush=True)
    print(f"{'='*60}", flush=True)

    return final_ade, final_fde


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Social LSTM on Argoverse val set"
    )

    # Checkpoint
    parser.add_argument(
        "--model_path", type=str, default="./checkpoints/best.pt",
        help="Path to checkpoint (.pt) produced by train.py",
    )

    # Data
    parser.add_argument("--val_dir",    type=str, default=DEFAULTS["val_dir"])
    parser.add_argument("--max_files",  type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--skip",       type=int, default=DEFAULTS["skip"])
    parser.add_argument("--max_agents", type=int, default=DEFAULTS["max_agents"])

    # Architecture fallbacks — only used when checkpoint has no saved cfg
    parser.add_argument("--embedding_dim", type=int,   default=DEFAULTS["embedding_dim"])
    parser.add_argument("--hidden_dim",    type=int,   default=DEFAULTS["hidden_dim"])
    parser.add_argument("--pred_len",      type=int,   default=DEFAULTS["pred_len"])
    parser.add_argument("--dropout",       type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--grid_size",     type=int,   default=DEFAULTS["grid_size"])
    parser.add_argument("--nb_size",       type=float, default=DEFAULTS["nb_size"])

    args = parser.parse_args()
    run_evaluation(vars(args))