"""
=============================================================
  IDD Segmentation II — YOLOv8 Evaluation Script
  Dataset: India Driving Dataset (IIIT Hyderabad)
  Model  : YOLOv8 (Ultralytics) — Semantic Segmentation
=============================================================

Folder structure expected:
  leftImg8bit/val/<seq_id>/frame####_leftImg8bit.jpg    
  gtFine/val/<seq_id>/frame####_gtFine_polygons.json

Usage:
    python evaluate_idd.py \
        --model   path/to/best.pt \
        --images  path/to/idd20kII/leftImg8bit/val \
        --gt_json path/to/idd20kII/gtFine/val \
        --output  eval_yolo_results/

Dependencies:
    pip install ultralytics opencv-python numpy matplotlib seaborn tqdm Pillow
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw

# ── IDD Segmentation II — label → class-id mapping ───────────────────────────
IDD_LABEL2ID = {
    "road": 0, "drivable fallback": 1, "sidewalk": 2,
    "non-drivable fallback": 3, "person": 4, "animal": 5,
    "rider": 6, "motorcycle": 7, "bicycle": 8, "autorickshaw": 9,
    "car": 10, "truck": 11, "bus": 12, "vehicle fallback": 13,
    "curb": 14, "wall": 15, "fence": 16, "guard rail": 17,
    "billboard": 18, "traffic sign": 19, "traffic light": 20,
    "pole": 21, "obs-str-bar-fallback": 22, "building": 23,
    "bridge": 24, "vegetation": 25, "sky": 26, "misc": 27,
    "unlabeled": 28,
}
IDD_CLASSES  = list(IDD_LABEL2ID.keys())
NUM_CLASSES  = len(IDD_CLASSES)
IGNORE_INDEX = 255

np.random.seed(42)
PALETTE = np.random.randint(50, 230, (NUM_CLASSES, 3)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  GT: rasterise polygon JSON → dense class-id mask
# ─────────────────────────────────────────────────────────────────────────────

def json_to_mask(json_path: Path) -> np.ndarray:
    """
    Read an IDD *_gtFine_polygons.json and render all polygons into
    a uint8 semantic mask of shape (H, W).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    h = data.get("imgHeight", 1080)
    w = data.get("imgWidth",  1920)

    mask = Image.fromarray(np.full((h, w), IGNORE_INDEX, dtype=np.uint8))
    draw = ImageDraw.Draw(mask)

    for obj in data.get("objects", []):
        label  = obj.get("label", "unlabeled").strip().lower()
        cls_id = IDD_LABEL2ID.get(label, IDD_LABEL2ID["unlabeled"])
        polygon = obj.get("polygon", [])
        if len(polygon) < 3:
            continue
        flat = [coord for pt in polygon for coord in pt]
        draw.polygon(flat, fill=int(cls_id))

    return np.array(mask, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Pair images ↔ JSON GT across nested subdirectories
# ─────────────────────────────────────────────────────────────────────────────

def collect_pairs(images_root: Path, gt_root: Path, ext_img=".jpg"):
    """
    Walk images_root/<seq_id>/frame####_leftImg8bit.jpg and find the
    matching gt_root/<seq_id>/frame####_gtFine_polygons.json
    """
    pairs = []
    img_suffix = f"_leftImg8bit{ext_img}"

    for seq_dir in sorted(images_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_id = seq_dir.name
        gt_dir = gt_root / seq_id
        if not gt_dir.is_dir():
            continue

        for img_path in sorted(seq_dir.glob(f"*{img_suffix}")):
            frame_id  = img_path.name.replace(img_suffix, "")
            json_path = gt_dir / f"{frame_id}_gtFine_polygons.json"
            if json_path.exists():
                pairs.append((img_path, json_path))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  YOLOv8 inference → dense semantic mask
# ─────────────────────────────────────────────────────────────────────────────

def predict_mask(model, image_path: Path, mask_hw: tuple) -> np.ndarray:
    results = model(str(image_path), verbose=False)[0]
    h, w    = mask_hw
    pred    = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)

    if results.masks is not None and results.masks.data is not None:
        masks_tensor = results.masks.data
        classes      = results.boxes.cls.cpu().numpy().astype(int)

        for inst_mask, cls_id in zip(masks_tensor, classes):
            m = inst_mask.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            pred[m > 0.5] = cls_id % NUM_CLASSES

    return pred


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds, gts):
        valid = gt != IGNORE_INDEX
        p = np.clip(pred[valid].astype(np.int64), 0, num_classes - 1)
        g = np.clip(gt[valid].astype(np.int64),   0, num_classes - 1)
        np.add.at(cm, (g, p), 1)
    return cm


def compute_metrics(cm):
    tp  = np.diag(cm)
    fn  = cm.sum(axis=1) - tp
    fp  = cm.sum(axis=0) - tp

    iou    = np.where((tp+fp+fn) > 0, tp / (tp+fp+fn+1e-10), np.nan)
    prec   = np.where((tp+fp)    > 0, tp / (tp+fp+1e-10),    np.nan)
    recall = np.where((tp+fn)    > 0, tp / (tp+fn+1e-10),    np.nan)

    return {
        "miou"          : float(np.nanmean(iou)),
        "pixel_acc"     : float(tp.sum() / (cm.sum() + 1e-10)),
        "mean_cls_acc"  : float(np.nanmean(recall)),
        "per_class_iou" : iou,
        "per_class_prec": prec,
        "per_class_rec" : recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE):
        rgb[mask == cls_id] = color
    return rgb


def save_sample_overlays(image_paths, pred_masks, gt_masks, out_dir: Path, n=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(image_paths)-1, min(n, len(image_paths)), dtype=int)
    for idx in indices:
        img  = cv2.cvtColor(cv2.imread(str(image_paths[idx])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        pred = cv2.resize(colorize(pred_masks[idx]), (w, h), interpolation=cv2.INTER_NEAREST)
        gt   = cv2.resize(colorize(gt_masks[idx]),   (w, h), interpolation=cv2.INTER_NEAREST)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, data, title in zip(
            axes,
            [img,
             cv2.addWeighted(img, 0.55, gt,   0.45, 0),
             cv2.addWeighted(img, 0.55, pred, 0.45, 0)],
            ["Input Image", "Ground Truth", "Prediction"],
        ):
            ax.imshow(data)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  ✔ {len(indices)} overlay images saved → {out_dir}")


def plot_confusion_matrix(cm, classes, out_path: Path, top_n=15):
    freq    = cm.sum(axis=1)
    top_idx = np.argsort(freq)[::-1][:top_n]
    sub_cm  = cm[np.ix_(top_idx, top_idx)]
    sub_cls = [classes[i] for i in top_idx]
    norm_cm = sub_cm.astype(float) / (sub_cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sub_cls, yticklabels=sub_cls,
                linewidths=0.4, linecolor="grey", ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix (top-{top_n} classes by frequency)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Confusion matrix saved → {out_path}")


def plot_per_class_iou(iou, classes, out_path: Path):
    valid = ~np.isnan(iou)
    vals  = iou[valid]
    lbls  = [classes[i] for i in range(len(iou)) if valid[i]]
    order = np.argsort(vals)[::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(vals)), vals[order], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([lbls[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(np.nanmean(iou), color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {np.nanmean(iou):.3f}")
    ax.legend(fontsize=11)
    ax.set_title("Per-Class IoU — IDD Segmentation II", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals[order]):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Per-class IoU chart saved → {out_path}")


def print_metrics_table(metrics, classes):
    iou  = metrics["per_class_iou"]
    prec = metrics["per_class_prec"]
    rec  = metrics["per_class_rec"]
    print("\n" + "="*70)
    print("  IDD SEGMENTATION II — EVALUATION RESULTS")
    print("="*70)
    print(f"  {'mIoU':<30} {metrics['miou']:>10.4f}")
    print(f"  {'Pixel Accuracy':<30} {metrics['pixel_acc']:>10.4f}")
    print(f"  {'Mean Class Accuracy':<30} {metrics['mean_cls_acc']:>10.4f}")
    print("="*70)
    print(f"\n  {'Class':<28} {'IoU':>8}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*58}")
    for i, cls in enumerate(classes):
        iv = f"{iou[i]:.4f}"  if not np.isnan(iou[i])  else "   N/A"
        pv = f"{prec[i]:.4f}" if not np.isnan(prec[i]) else "   N/A"
        rv = f"{rec[i]:.4f}"  if not np.isnan(rec[i])  else "   N/A"
        print(f"  {cls:<28} {iv:>8}  {pv:>10}  {rv:>8}")
    print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 IDD-Seg-II Evaluation")
    p.add_argument("--model",   required=True,
                   help="Path to best.pt / last.pt")
    p.add_argument("--images",  required=True,
                   help="Root val dir: leftImg8bit/val/  (contains numbered sub-folders)")
    p.add_argument("--gt_json", required=True,
                   help="Root GT dir:  gtFine/val/       (contains numbered sub-folders)")
    p.add_argument("--output",  default="eval_results",
                   help="Output directory for plots & JSON")
    p.add_argument("--ext_img", default=".jpg",
                   help="Image extension (default: .jpg)")
    p.add_argument("--samples", type=int, default=6,
                   help="Number of overlay samples to save (default: 6)")
    p.add_argument("--limit",   type=int, default=None,
                   help="Evaluate only first N pairs (quick test)")
    return p.parse_args()


def main():
    args        = parse_args()
    model_path  = Path(args.model)
    images_root = Path(args.images)
    gt_root     = Path(args.gt_json)
    out_dir     = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert model_path.exists(),  f"Model not found: {model_path}"
    assert images_root.is_dir(), f"Images root not found: {images_root}"
    assert gt_root.is_dir(),     f"GT root not found: {gt_root}"

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run:  pip install ultralytics")
    model = YOLO(str(model_path))

    # ── Collect pairs ────────────────────────────────────────────────────────
    print(f"[2/4] Scanning dataset …")
    pairs = collect_pairs(images_root, gt_root, ext_img=args.ext_img)

    if not pairs:
        print("\n  ERROR: No image-JSON pairs found.")
        print(f"  Images searched: {images_root}/<seq_id>/*_leftImg8bit{args.ext_img}")
        print(f"  GT searched:     {gt_root}/<seq_id>/*_gtFine_polygons.json")
        sys.exit(1)

    if args.limit:
        pairs = pairs[: args.limit]

    num_seqs = len(set(p[0].parent.name for p in pairs))
    print(f"   Found {len(pairs)} image-JSON pairs across {num_seqs} sequences.")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"[3/4] Running inference …")
    all_preds, all_gts, all_img_paths = [], [], []

    for img_path, json_path in tqdm(pairs, unit="img"):
        gt   = json_to_mask(json_path)
        pred = predict_mask(model, img_path, gt.shape[:2])
        all_preds.append(pred)
        all_gts.append(gt)
        all_img_paths.append(img_path)

    # ── Metrics ──────────────────────────────────────────────────────────────
    print(f"[4/4] Computing metrics …")
    cm      = build_confusion_matrix(all_preds, all_gts, NUM_CLASSES)
    metrics = compute_metrics(cm)
    print_metrics_table(metrics, IDD_CLASSES)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Saving visualisations …")
    plot_confusion_matrix(cm, IDD_CLASSES, out_dir / "confusion_matrix.png")
    plot_per_class_iou(metrics["per_class_iou"], IDD_CLASSES, out_dir / "per_class_iou.png")
    save_sample_overlays(all_img_paths, all_preds, all_gts,
                         out_dir / "overlays", n=args.samples)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    import json as _json
    summary = {
        "mIoU"          : round(metrics["miou"],         4),
        "pixel_accuracy": round(metrics["pixel_acc"],    4),
        "mean_class_acc": round(metrics["mean_cls_acc"], 4),
        "per_class_iou" : {
            IDD_CLASSES[i]: (round(float(v), 4) if not np.isnan(v) else None)
            for i, v in enumerate(metrics["per_class_iou"])
        },
    }
    json_out = out_dir / "metrics_summary.json"
    with open(json_out, "w") as f:
        _json.dump(summary, f, indent=2)

    print(f"\n  ✔ Summary JSON saved → {json_out}")
    print(f"  All results written to: {out_dir.resolve()}")
    print("\n  ┌─────────────────────────────────────┐")
    print(f"  │  mIoU          : {metrics['miou']:.4f}             │")
    print(f"  │  Pixel Accuracy: {metrics['pixel_acc']:.4f}             │")
    print(f"  │  Mean Cls Acc  : {metrics['mean_cls_acc']:.4f}             │")
    print("  └─────────────────────────────────────┘\n")


if __name__ == "__main__":
    main()