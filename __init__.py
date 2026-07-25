"""Offline AI photo culling."""

__version__ = "2.0.0"

from .batch import BatchCuller
from .config import Config
from .decision import CullDecider
from .extractor import RawThumbnailExtractor
from .grouping import annotate_results, group_photos
from .models import CullResult, ImageMetrics
from .vision import ModelCannotSee, OllamaVisionAnalyzer, VisionUnavailable, detect_vision_model

__all__ = [
    "BatchCuller",
    "Config",
    "CullDecider",
    "CullResult",
    "ImageMetrics",
    "ModelCannotSee",
    "OllamaVisionAnalyzer",
    "RawThumbnailExtractor",
    "VisionUnavailable",
    "annotate_results",
    "detect_vision_model",
    "group_photos",
]
