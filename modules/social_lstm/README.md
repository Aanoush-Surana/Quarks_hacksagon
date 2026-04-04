# Social LSTM – Road-Scene Trajectory Prediction

Predict the next ~1.2 s of motion for all tracked objects, trained on
**Argoverse 1** and deployed directly on **BotSORT** detection output.

```
social_lstm/
├── data/
│   ├── argoverse_loader.py   ← Argoverse CSV → training windows
│   └── botsort_adapter.py    ← BotSORT JSON → inference windows
├── models/
│   └── social_lstm.py        ← Social LSTM + bivariate NLL loss
├── utils/
│   └── visualise.py          ← Overlay predictions on video
├── train.py                  ← Training loop
├── predict.py                ← Inference on BotSORT JSON
└── requirements.txt
```

---

## 1  Install

```bash
pip install -r requirements.txt
```

---

## 2  Download Argoverse 1

```bash
# Sign up at https://www.argoverse.org/av1.html and download
# the Motion Forecasting dataset. Unzip to a local directory, e.g.:
#   data/argoverse/train/data/   (many .csv files)
#   data/argoverse/val/data/
```

---

## 3  Train

```bash
python train.py \
    --data_dir data/argoverse/train/data \
    --val_dir  data/argoverse/val/data   \
    --output_dir checkpoints             \
    --epochs 200                         \
    --batch_size 64                      \
    --hidden_dim 128                     \
    --embedding_dim 64
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--lr_decay` | 0.5 | LR decay factor |
| `--lr_decay_every` | 10 | Decay every N epochs |
| `--hidden_dim` | 128 | LSTM hidden state size |
| `--embedding_dim` | 64 | Input embedding size |
| `--grid_size` | 8 | Social pooling grid N×N |
| `--nb_size` | 32.0 | Social pooling radius (metres) |
| `--pred_len` | 12 | Future frames to predict (12 × 0.1s = 1.2 s) |
| `--dropout` | 0.1 | Dropout probability |
| `--resume` | None | Path to checkpoint to resume from |

Checkpoints are saved to `--output_dir/last.pt` (every epoch) and
`best.pt` (best validation ADE).

---

## 4  Run inference on your BotSORT output

### 4a  Pixel → metric coordinate conversion

Social LSTM was trained in **metric space (metres)**. BotSORT outputs
**pixel bbox centres**. You need to tell the adapter how to convert.

**Option A – simple pixels-per-metre scale (quick start)**
```bash
python predict.py \
    --checkpoint    checkpoints/best.pt           \
    --botsort_json  data/videoplayback_tracks.json \
    --output_json   predictions.json              \
    --pixels_per_metre 10.0
```
Adjust `pixels_per_metre` to match your camera setup.
A value of 10 means 1 m ≈ 10 pixels (bird's-eye view).

**Option B – homography matrix (accurate)**

If you have a ground-plane homography H (e.g. from a calibration
checkerboard or using OpenCV's `findHomography`), save it as a JSON
array of 9 floats (row-major) and pass it in:

```bash
python predict.py \
    --checkpoint    checkpoints/best.pt           \
    --botsort_json  data/videoplayback_tracks.json \
    --output_json   predictions.json              \
    --homography    data/homography.json
```

`homography.json` format (3×3 flattened, row-major):
```json
[h00, h01, h02, h10, h11, h12, h20, h21, h22]
```

### 4b  Stochastic sampling

Add `--n_samples 20` to get 20 sampled trajectories per agent per window
(in addition to the mean prediction). Useful for uncertainty estimation.

```bash
python predict.py \
    --checkpoint checkpoints/best.pt        \
    --botsort_json my_video_tracks.json     \
    --n_samples 20                          \
    --stride 5
```

`--stride N` skips N−1 frames between windows (reduces output size).

---

## 5  Visualise

```bash
python utils/visualise.py \
    --video        data/videoplayback.mp4 \
    --predictions  predictions.json       \
    --output       annotated.mp4          \
    --pixels_per_metre 10.0               \
    --show_samples
```

Colours:
- **White** dots/line = observed history
- **Yellow** dots/line = predicted mean trajectory
- **Blue** (faint) = stochastic samples

---

## 6  Integrating into your pipeline

```python
from data.botsort_adapter import BotSORTAdapter, CoordConverter
from models.social_lstm    import SocialLSTM, sample_trajectories
import torch, json

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt    = torch.load("checkpoints/best.pt", map_location=device)
model   = SocialLSTM(**{k: ckpt["cfg"][k] for k in
              ["embedding_dim","hidden_dim","pred_len","grid_size"]},
              neighbourhood_size=ckpt["cfg"]["nb_size"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

adapter = BotSORTAdapter("my_video_tracks.json",
                          CoordConverter(pixels_per_metre=10.0))

for win in adapter.get_inference_windows(stride=5):
    obs  = torch.from_numpy(win["obs"]).to(device)
    mask = torch.from_numpy(win["mask"]).to(device)
    with torch.no_grad():
        mu, sigma, rho = model(obs, mask)
    # mu: (PRED_LEN, max_agents, 2) in normalised metric space
    # add win["mean_xy"] to de-normalise
```

---

## 7  Notes on coordinate systems

- Argoverse uses **metric** (x, y) in a city-level coordinate frame.
- Your BotSORT detections are in **pixel** space.
- The `CoordConverter` bridges the gap. If your camera is not calibrated,
  use `pixels_per_metre` as an approximation:
  - Drone overhead at 30 m height, wide lens → ~5–15 px/m
  - Dashcam, close foreground → 20–50 px/m
- The social pooling radius (`--nb_size`) is in the same units as the
  training data (metres). Keep it at 32 m or adjust to the density of
  your road scene.

---

## 8  Extending the model

To get better performance you can:

1. **Fine-tune on your own data** — annotate a small set of your BotSORT
   sequences with ground-truth future positions and run `train.py` with
   `--resume checkpoints/best.pt` pointing to the Argoverse checkpoint.
2. **Add class conditioning** — embed `class_id` and concatenate to the
   input embedding so the model learns different motion priors per class.
3. **Replace social pooling** — the occupancy-grid pooling is the original
   Social LSTM mechanism; you can swap it for a Graph Attention layer
   (GATConv from PyG) to get Social-STGCNN-style interactions.
