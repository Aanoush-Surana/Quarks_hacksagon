# 📊 YOLOv8 Segmentation Evaluation Results

This document summarizes the performance metrics of the YOLOv8m-seg model trained on the IDD20k II dataset.

## 📈 Global Metrics

| Metric | Value |
|--------|-------|
| **mIoU** (Mean Intersection over Union) | **0.2103** |
| **Pixel Accuracy** | **0.2865** |
| **Mean Class Accuracy** | **0.2763** |

---

## 🚗 Per-Class IoU Breakdown (Top Performers)

The model performs exceptionally well on primary road objects and traffic participants, which are critical for autonomous navigation in Indian environments.

| Class | IoU |
|-------|-----|
| 🛣️ **Road** | **0.6979** |
| 🚗 **Car** | **0.6879** |
| 🚕 **Autorickshaw** | **0.6796** |
| 🚌 **Bus** | **0.6441** |
| 🏍️ **Motorcycle** | **0.6305** |
| 🚶 **Person** | **0.5949** |
| 🚛 **Truck** | **0.5690** |
| 🚴 **Rider** | **0.5251** |
| 🐕 **Animal** | **0.5047** |

---

## 🖼️ Visualizations

The following analysis assets are available in this directory:

- `confusion_matrix.png`: Normalized heatmap showing classification performance across the top 15 classes.
- `per_class_iou.png`: Bar chart visualization of Intersection over Union for all classes compared to the mean IoU.
- `overlays/`: Directory containing side-by-side comparisons of Input, Ground Truth, and Model Predictions.

> [!NOTE]
> Performance is strongest on high-frequency traffic objects. Some rare or extremely small classes are captured in the visualization outputs but contribute less to the overall mIoU due to their sparsity in the evaluation set.
