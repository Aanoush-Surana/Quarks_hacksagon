import cv2
import os
import logging

class Preprocessor:
    def __init__(self, target_resolution=(640, 640), high_perf=True, enabled=True):
        self.target_resolution = target_resolution
        self.high_perf = high_perf
        self.enabled=enabled
        self.logger = logging.getLogger(self.__class__.__name__)

    def enhance_contrast(self, frame):
        """LAB color space CLAHE enhancement."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_filters(self, frame):
        """Fast sharpening filter."""
        if self.high_perf:
            # Much faster than bilateralFilter for real-time
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        else:
            blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            
        # Laplacian sharpening
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(blurred, 1.2, laplacian_3ch, -0.3, 0)
        return sharpened

    def process_frame(self, frame):
        """Processes a single frame: contrast, sharp, resize."""

        if not self.enabled:
        # 🚀 Completely bypass preprocessing
            return frame
        # For maximum performance, we can skip contrast enhancement if already good
        if not self.high_perf:
            frame = self.enhance_contrast(frame)
            
        frame_proc = self.apply_filters(frame)
        frame_proc = cv2.resize(frame_proc, self.target_resolution)
        return frame_proc

    def process_video(self, input_path, output_path):
        """
        Reads a video, processes all frames, saves to output_path,
        and returns the output_path for downstream tasks.
        """
        self.logger.info(f"Preprocessing started: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {input_path}")
            raise ValueError(f"Cannot open video source: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # For writing mp4, mp4v is commonly supported cross-platform
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.target_resolution)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            proc_frame = self.process_frame(frame)
            out.write(proc_frame)
            frame_count += 1

        cap.release()
        out.release()
        self.logger.info(f"Preprocessing completed. Saved {frame_count} frames to {output_path}")
        return output_path

    def stream(self, input_path, target_fps=30.0):
        """
        Stream frames sequentially with processing applied.
        Useful if intermediate writing to disk is disabled.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
             raise ValueError("Cannot open video source")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 1000:
            source_fps = target_fps

        source_frame_time = 1.0 / source_fps
        target_frame_time = 1.0 / target_fps
        time_accumulator = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_accumulator += source_frame_time
            while time_accumulator >= target_frame_time:
                yield self.process_frame(frame)
                time_accumulator -= target_frame_time

        cap.release()
