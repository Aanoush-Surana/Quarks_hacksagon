# YOLOv8 Segmentation & Tracking Pipeline

This repository contains a modular Python pipeline for real-time video segmentation and tracking, primarily designed for dashcam footage. It utilizes a trained YOLOv8 model to perform inference on video frames sequentially, processes these frames to enhance contrast/sharpness, outputs a tracking JSON file, and renders the segmented video.

## Project Structure
```
.
├── config.yaml               # Central configuration file for paths & pipeline settings
├── main.py                   # Main entry point to run the pipeline
├── requirements.txt          # Python dependencies
├── modules/                  # Extracted and structured modules
│   ├── preprocess/
│   │   └── cleaner.py        # Frame preprocessor (enhances contrast, sharpen)
│   ├── segmentation/
│   │   └── inference.py      # YOLOv8 model handler and renderer
│   └── tracking/
│       └── tracker.py        # Object tracking placeholder
└── README.md                 # Project documentation
```

## Setup & Installation

1. **Clone or Navigate to the directory:**
    Ensure you are in the project root containing `main.py`.

2. **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3. **Get YOLO weights:**
    Make sure you have your trained YOLOv8 PyTorch weights file (e.g. `best.pt` or `yolo26n.pt`) and update the `default_weights` path in `config.yaml`.

## Usage

You can run the pipeline with the default settings configured in `config.yaml`:

```bash
python main.py
```

Or you can override the paths using CLI arguments:

```bash
python main.py --video_path /path/to/your/input.mp4 --weights_path /path/to/your/model_weights.pt
```

## Pipeline Flow

1. **Preprocessing:** Frames are read incrementally. The preprocessor enhances contrast and sharpens them before resizing to the target resolution (default 640x640).
2. **Segmentation:** The `SegmentationModel` loads a YOLO model, runs prediction on the processed frame, and draws overlays (masks, bounding boxes, labels) using specified colour palettes.
3. **Tracking:** (To be fully implemented) Currently passes frames forward, along with JSON metadata for bounding boxes and mask coordinates per frame.
4. **Outputs:** The resultant frames are written to an MP4 video under `data/outputs/tracking/`, and the structured data corresponds to a `_results.json` dumped alongside it.

## Limitations & Edge Cases Handled

- **Memory Leak Proof Pipeline**: Replaced bulk pre-processing methods that write videos on disk multiple times. This pipeline now processes streams natively and saves final combined output directly.
- **Robust Model Handling**: Automatically scales detection masks to match frame resolution overlay.

