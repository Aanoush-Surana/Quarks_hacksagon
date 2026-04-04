"""
modules.temporal_fusion — Bayesian temporal mask fusion for dashcam pipelines.

Public API:
    TemporalMaskFusion           — core temporal fusion engine
    DetectionPrefilter           — pre-BoT-SORT flicker suppression
    extract_detections_from_result — extract detection dicts from YOLO result
"""

from .temporal_fusion_core import TemporalMaskFusion
from .detection_prefilter import DetectionPrefilter
from .helpers import extract_detections_from_result

__all__ = [
    "TemporalMaskFusion",
    "DetectionPrefilter",
    "extract_detections_from_result",
]
