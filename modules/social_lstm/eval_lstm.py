import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader

# --- IMPORT YOUR EXISTING CLASSES ---
from data.argoverse_loader import ArgoverseLoader, collate_fn
from models.social_lstm    import SocialLSTM

# This ensures we use the exact same logic as your training script
from train import DEFAULTS, ade_fde_batch 

@torch.no_grad()
def run_evaluation(model_path):
    # 1. Setup Configuration
    cfg = DEFAULTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on: {device} ---")

    # max_files_to_load= cfg.get("max_files", 500)
    max_files_to_load=100
    # 2. Load Validation Dataset
    # We remove 'val_batches' cap here to get the most accurate stats
    print(f"Loading data from: {cfg['val_dir']}...")
    val_ds = ArgoverseLoader(
        data_dir   = cfg["val_dir"],
        skip       = cfg["skip"],
        max_agents = cfg["max_agents"],
        max_files=max_files_to_load
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size  = cfg["batch_size"], 
        shuffle     = False, 
        collate_fn  = collate_fn,
        pin_memory  = True
    )

    # 3. Initialize Model & Load Weights
    model = SocialLSTM(
        embedding_dim = cfg["embedding_dim"],
        hidden_dim    = cfg["hidden_dim"],
        pred_len      = cfg["pred_len"],
        dropout       = cfg["dropout"],
        grid_size     = cfg["grid_size"]
    ).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Weights file not found at {model_path}")
        return

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        print("  Found 'model' key in checkpoint, loading state_dict...")
        model.load_state_dict(checkpoint["model"])
    else:
        # Fallback for if it's a direct state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 4. Metrics Loop
    total_ade, total_fde, n = 0.0, 0.0, 0
    t0 = time.time()

    for i, (obs, pred, mask) in enumerate(val_loader):
        obs  = obs.to(device)
        pred = pred.to(device)
        mask = mask.to(device)

        # Forward pass (only need mu for ADE/FDE)
        mu, _, _ = model(obs, mask)

        # Calculate metrics using your existing function
        a, f = ade_fde_batch(mu, pred, mask)
        
        if not np.isnan(a):
            total_ade += a
            total_fde += f
            n += 1

        if i % 50 == 0:
            print(f"Batch {i}/{len(val_loader)} | Running ADE: {total_ade/max(n,1):.4f}")

    # 5. Final Results
    final_ade = total_ade / max(n, 1)
    final_fde = total_fde / max(n, 1)
    
    print("\n" + "="*30)
    print("FINAL VALIDATION STATISTICS")
    print(f"Total Scenarios : {n * cfg['batch_size']}")
    print(f"Average ADE     : {final_ade:.4f}")
    print(f"Average FDE     : {final_fde:.4f}")
    print(f"Total Time      : {time.time()-t0:.1f}s")
    print("="*30)

if __name__ == "__main__":
    # Update this path to your actual best model filename
    BEST_MODEL_PATH = "./checkpoints/best.pt" 
    run_evaluation(BEST_MODEL_PATH)
