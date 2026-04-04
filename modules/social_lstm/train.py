"""
Train / fine-tune Social LSTM on Argoverse data.

Usage
─────
python train.py --data_dir /path/to/argoverse/train/data \
                --val_dir  /path/to/argoverse/val/data   \
                --output_dir ./checkpoints               \
                [--resume ./checkpoints/best.pt]
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.argoverse_loader import ArgoverseLoader, collate_fn
from models.social_lstm    import SocialLSTM, bivariate_nll_loss


# ── Config defaults (override via CLI or config file) ────────────────────────

DEFAULTS = dict(
    # Data
    data_dir      = "./data/argoverse/train/data",
    val_dir       = "./data/argoverse/val/data",
    output_dir    = "./checkpoints",
    # Model
    embedding_dim = 64,
    hidden_dim    = 128,
    pred_len      = 12,
    dropout       = 0.1,
    grid_size     = 8,
    nb_size       = 32.0,
    max_agents    = 32,
    # Training
    epochs        = 20,
    batch_size    = 2,
    lr            = 1e-3,
    lr_decay      = 0.5,
    lr_decay_every = 5,
    grad_clip     = 1.0,
    skip          = 1,
    resume        = None,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
    max_files     = 2000,
    # How often to print batch progress (every N batches)
    print_every   = 200,
)


# ── Metrics ──────────────────────────────────────────────────────────────────

def ade(pred_mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    T, A, _ = target.shape
    valid    = mask.unsqueeze(0).expand(T, A) & ~torch.isnan(target).any(-1)
    if valid.sum() == 0:
        return float("nan")
    err = torch.norm(pred_mu - target, dim=-1)
    return err[valid].mean().item()


def fde(pred_mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    A = target.shape[1]
    last_mu  = pred_mu[-1]
    last_tgt = target[-1]
    valid = mask & ~torch.isnan(last_tgt).any(-1)
    if valid.sum() == 0:
        return float("nan")
    err = torch.norm(last_mu - last_tgt, dim=-1)
    return err[valid].mean().item()


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, grad_clip, epoch, print_every):
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    n_total     = len(loader)
    epoch_start = time.time()

    print(f"  [train] Starting training pass — {n_total} batches total", flush=True)

    for batch_idx, (obs, pred, mask) in enumerate(loader):

        if batch_idx % print_every == 0:
            elapsed  = time.time() - epoch_start
            pct      = 100.0 * batch_idx / max(n_total, 1)
            avg_loss = total_loss / max(n_batches, 1)
            print(
                f"  [train] Epoch {epoch+1} | batch {batch_idx:5d}/{n_total} "
                f"({pct:5.1f}%) | avg_loss={avg_loss:.4f} | {elapsed:.1f}s elapsed",
                flush=True,
            )

        obs  = obs.to(device)
        pred = pred.to(device)
        mask = mask.to(device)

        B = obs.size(0)
        batch_loss = 0.0

        for b in range(B):
            obs_b  = obs[b]
            pred_b = pred[b]
            mask_b = mask[b]

            mu, sigma, rho = model(obs_b, mask_b)
            loss = bivariate_nll_loss(mu, sigma, rho, pred_b, mask_b)
            batch_loss += loss

        batch_loss = batch_loss / B
        optimiser.zero_grad()
        batch_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()

        total_loss += batch_loss.item()
        n_batches  += 1

    epoch_time = time.time() - epoch_start
    avg = total_loss / max(n_batches, 1)
    print(
        f"  [train] Training pass complete — avg_loss={avg:.4f} | "
        f"total time={epoch_time:.1f}s",
        flush=True,
    )
    return avg


@torch.no_grad()
def validate(model, loader, device, epoch, print_every):
    model.eval()
    total_ade, total_fde, n = 0.0, 0.0, 0
    n_total   = len(loader)
    val_start = time.time()

    print(f"  [val]   Starting validation pass — {n_total} batches total", flush=True)

    for batch_idx, (obs, pred, mask) in enumerate(loader):

        if batch_idx % print_every == 0:
            elapsed = time.time() - val_start
            pct     = 100.0 * batch_idx / max(n_total, 1)
            print(
                f"  [val]   Epoch {epoch+1} | batch {batch_idx:5d}/{n_total} "
                f"({pct:5.1f}%) | {elapsed:.1f}s elapsed",
                flush=True,
            )

        obs  = obs.to(device)
        pred = pred.to(device)
        mask = mask.to(device)

        B = obs.size(0)
        for b in range(B):
            mu, _, _ = model(obs[b], mask[b])
            total_ade += ade(mu, pred[b], mask[b])
            total_fde += fde(mu, pred[b], mask[b])
            n += 1

    val_time = time.time() - val_start
    a = total_ade / max(n, 1)
    f = total_fde / max(n, 1)
    print(
        f"  [val]   Validation complete — ADE={a:.4f}  FDE={f:.4f} | "
        f"total time={val_time:.1f}s",
        flush=True,
    )
    return a, f


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    print("=" * 60, flush=True)
    print(f"[init] Device      : {device}", flush=True)
    print(f"[init] Epochs      : {cfg['epochs']}", flush=True)
    print(f"[init] Batch size  : {cfg['batch_size']}", flush=True)
    print(f"[init] LR          : {cfg['lr']}", flush=True)
    print(f"[init] Max files   : {cfg.get('max_files', 'all')}", flush=True)
    print(f"[init] Hidden dim  : {cfg['hidden_dim']}", flush=True)
    print(f"[init] Output dir  : {cfg['output_dir']}", flush=True)
    print("=" * 60, flush=True)

    # ── data ────────────────────────────────────────────────────────────────
    print("\n[data] Loading training dataset...", flush=True)
    t0 = time.time()
    train_ds = ArgoverseLoader(
        data_dir   = cfg["data_dir"],
        skip       = cfg["skip"],
        max_agents = cfg["max_agents"],
        max_files  = cfg.get("max_files"),
    )
    print(f"[data] Train dataset ready: {len(train_ds)} windows ({time.time()-t0:.1f}s)", flush=True)

    print("\n[data] Loading validation dataset...", flush=True)
    t0 = time.time()
    val_ds = ArgoverseLoader(
        data_dir   = cfg["val_dir"],
        skip       = cfg["skip"],
        max_agents = cfg["max_agents"],
        max_files  = cfg.get("max_files"),
    )
    print(f"[data] Val dataset ready  : {len(val_ds)} windows ({time.time()-t0:.1f}s)", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = False,
    )
    print(f"[data] Train batches: {len(train_loader)}  Val batches: {len(val_loader)}", flush=True)

    # ── model ────────────────────────────────────────────────────────────────
    print("\n[model] Building Social LSTM...", flush=True)
    model = SocialLSTM(
        embedding_dim      = cfg["embedding_dim"],
        hidden_dim         = cfg["hidden_dim"],
        pred_len           = cfg["pred_len"],
        dropout            = cfg["dropout"],
        grid_size          = cfg["grid_size"],
        neighbourhood_size = cfg["nb_size"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parameters : {n_params:,}", flush=True)
    print(f"[model] On device  : {next(model.parameters()).device}", flush=True)

    optimiser = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimiser,
        step_size = cfg["lr_decay_every"],
        gamma     = cfg["lr_decay"],
    )

    start_epoch = 0
    best_ade    = float("inf")

    # ── optional resume ──────────────────────────────────────────────────────
    if cfg["resume"] and os.path.isfile(cfg["resume"]):
        print(f"\n[resume] Loading checkpoint: {cfg['resume']}", flush=True)
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ade    = ckpt.get("best_ade", best_ade)
        print(f"[resume] Resuming from epoch {start_epoch}, best ADE so far = {best_ade:.4f}", flush=True)

    # ── quick sanity-check forward pass before committing to training ────────
    print("\n[sanity] Running one forward pass before training...", flush=True)
    model.eval()
    with torch.no_grad():
        sample_obs, sample_pred, sample_mask = train_ds[0]
        sample_obs  = sample_obs.to(device)
        sample_mask = sample_mask.to(device)
        mu, sigma, rho = model(sample_obs, sample_mask)
    print(f"[sanity] OK — mu={tuple(mu.shape)}  sigma={tuple(sigma.shape)}  rho={tuple(rho.shape)}", flush=True)

    # ── training loop ────────────────────────────────────────────────────────
    print(f"\n[train] Starting training for {cfg['epochs'] - start_epoch} epochs\n", flush=True)
    print_every = cfg.get("print_every", 200)

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_wall = time.time()
        print(f"\n{'─'*60}", flush=True)
        print(
            f"[epoch {epoch+1:3d}/{cfg['epochs']}] "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        train_loss = train_one_epoch(
            model, train_loader, optimiser, device, cfg["grad_clip"],
            epoch, print_every,
        )

        val_ade, val_fde = validate(
            model, val_loader, device, epoch, print_every,
        )

        scheduler.step()
        elapsed = time.time() - epoch_wall

        print(
            f"\n[epoch {epoch+1:3d}/{cfg['epochs']}] SUMMARY | "
            f"loss={train_loss:.4f} | "
            f"ADE={val_ade:.4f}  FDE={val_fde:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        print(f"[epoch {epoch+1:3d}] Saving last.pt...", flush=True)
        ckpt_path = os.path.join(cfg["output_dir"], "last.pt")
        torch.save(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_ade":  best_ade,
                "cfg":       cfg,
            },
            ckpt_path,
        )
        print(f"[epoch {epoch+1:3d}] last.pt saved.", flush=True)

        if val_ade < best_ade:
            best_ade = val_ade
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                os.path.join(cfg["output_dir"], "best.pt"),
            )
            print(f"[epoch {epoch+1:3d}] ★ New best ADE={best_ade:.4f} — best.pt saved.", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[done] Training complete. Best val ADE = {best_ade:.4f}", flush=True)
    print(f"[done] Checkpoints in: {cfg['output_dir']}", flush=True)
    print(f"{'='*60}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Social LSTM on Argoverse")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()
    main(vars(args))