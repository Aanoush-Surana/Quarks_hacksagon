# Refactoring YOLOv8 Pipeline Architecture

The goal of this refactor is to transition the research code from Jupyter notebooks (`IDD_YOLOv8_Segmentation.py`, `new_testbed.py`) into a clean, modular Python pipeline structure that uses OOP principles (e.g., a `SegmentationModel` class). 

> [!NOTE]
> All existing `.ipynb` files will remain completely untouched.

## User Review Required

> [!IMPORTANT]
> - Is the root directory `D:\Hacksagon\notebbooks` the correct place to generate these new `modules/` and `data/` directories, or would you prefer a new parent project folder entirely? (My plan currently structures everything in `D:\Hacksagon\notebbooks`).
> - For the segmentation JSON results, what specific format does your tracking team expect for mask coordinates? By default, I will extract contour coordinates as lists of `(x, y)` points. Let me know if you prefer run-length encoding (RLE) or a specific bounding-box format (e.g., `[x_min, y_min, x_max, y_max]` vs `[x, y, w, h]`).

## Proposed Changes

We will establish a modular structure following the provided diagram, leaving notebooks unmodified.

---

### Core Pipeline Script

#### [NEW] `main.py`
- Acts as the main entry point utilizing `argparse`.
- Accepts arguments like `--video_path` and `--weights_path` (with sensible defaults hardcoded to your existing weights).
- Orchestrates the sequence: `Preprocess -> Segmentation -> Tracking`.
- Ensures intermediate files are handed off to the next step cleanly.
- Implements basic logging via the standard Python `logging` library.

#### [NEW] `config.yaml`
- Centralizes default paths for IO operations and weights, ensuring `main.py` can load them easily.

---

### Modules

#### [NEW] `modules/preprocess/cleaner.py`
- Will adapt `preprocessor.py` (which applies CLAHE and Laplacian sharpening) into a well-defined `Preprocessor` class or functional module.
- Saves processed outputs to `data/outputs/preprocess/`.

#### [NEW] `modules/segmentation/inference.py`
- Contains the `SegmentationModel` class.
- Will load YOLOv8 weights (no training).
- Applies inference to input frames.
- Renders the image with standard overlays (derived from the existing palette and mask logic).
- Saves the processed video to `data/outputs/segmentation/`.
- Extracts and saves `results.json` containing: `frame_id`, `class_id`, `class_name`, `confidence`, `bbox`, and `mask_coordinates`.

#### [NEW] `modules/tracking/tracker.py`
- A placeholder tracking module.
- Receives the final video and `.json` files from the segmentation step.
- Implements a dummy `run_tracking()` method outlining where the downstream teammates will add their logic.

---

### File Storage / Data Persistence

#### [NEW] `data/inputs/`
- Directory for holding original videos.

#### [NEW] `data/outputs/...`
- Dirs: `preprocess/`, `segmentation/`, `tracking/`.

#### [NEW] `data/weights/`
- Directory for models. I will leave this empty or point `main.py` to your current `yolo26n.pt` / `best.pt` path.

## Open Questions
- You provided `D:\Hacksagon\notebbooks\yolo26n.pt` and also the training checkpoint `best.pt`. Which model do you want to use as the default weight in the code file? I will default to `yolo26n.pt` unless you tell me otherwise.
- In `preprocessor.py`, there is streaming logic `preprocessing_stream()`. Do you want the whole video to be preprocessed and saved as an intermediate MP4 first, or do you want the preprocessor to stream directly to the `SegmentationModel` frame by frame? (I will implement a stream directly for efficiency unless you prefer saving the preprocessed video physically to disk as an intermediate).

## Verification Plan

### Automated/Manual Verification
- Execute `D:\Obj_detection\env\python.exe main.py` with no arguments, ensuring it correctly picks up the default paths.
- Execute it with a `--video_path` argument to see if it processes end-to-end.
- Check the console logs for "Preprocess started...", "Segmentation started...", etc.
- Verify `data/outputs/segmentation/` to ensure it contains both `.mp4` video output and the structured `results.json` file.
