"""Objective exposure measurement from the pixel histogram.

Unlike sharpness, exposure *is* reliably measurable without a model: clipped highlights
and crushed shadows are a counting exercise. So here the numbers lead and the vision
model is the cross-check, which is the opposite of the arrangement in blur_detector.py.

Judgement still needs care. A high-key beach scene is legitimately full of near-white
pixels and a night shot is legitimately mostly black, so we look at how much of the
frame is *irrecoverably* clipped rather than at average brightness.
"""

from typing import Dict
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ANALYSIS_EDGE = 1024

# A pixel is unrecoverable once it hits the top/bottom of the range. Small amounts are
# normal (specular highlights, deep shadow) so thresholds are generous.
HIGHLIGHT_LEVEL = 250
SHADOW_LEVEL = 5

# Fraction of the frame that must be clipped before it counts as a real problem.
SEVERE_CLIP_FRACTION = 0.15
MILD_CLIP_FRACTION = 0.05

# Clipping alone misses the frame that is uniformly murky without ever touching the ends
# of the range -- an underexposed low-contrast scan does exactly that. These bounds are
# deliberately extreme so that legitimate low-key and high-key work stays untouched.
VERY_DARK_MEAN = 0.10
VERY_BRIGHT_MEAN = 0.92


class ExposureMeter:
    """Measures clipping and overall tone from the luminance histogram."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def measure(self, image: Image.Image) -> Dict[str, float]:
        try:
            working = image.copy()
            if working.mode != "L":
                working = working.convert("L")
            working.thumbnail((ANALYSIS_EDGE, ANALYSIS_EDGE), Image.Resampling.LANCZOS)
            luma = np.asarray(working, dtype=np.uint8)

            total = luma.size
            blown = float(np.count_nonzero(luma >= HIGHLIGHT_LEVEL)) / total
            crushed = float(np.count_nonzero(luma <= SHADOW_LEVEL)) / total
            mean_level = float(luma.mean()) / 255.0
            contrast = float(luma.std()) / 255.0

            if blown >= SEVERE_CLIP_FRACTION or mean_level >= VERY_BRIGHT_MEAN:
                category = "badly_over"
            elif crushed >= SEVERE_CLIP_FRACTION or mean_level <= VERY_DARK_MEAN:
                category = "badly_under"
            elif blown >= MILD_CLIP_FRACTION or crushed >= MILD_CLIP_FRACTION:
                category = "slightly_off"
            else:
                category = "good"

            return {
                "category": category,
                "blown_fraction": blown,
                "crushed_fraction": crushed,
                "mean_level": mean_level,
                "contrast": contrast,
                "available": True,
            }

        except Exception as e:
            self.logger.warning(f"Exposure measurement failed: {e}")
            return {
                "category": "good",
                "blown_fraction": 0.0,
                "crushed_fraction": 0.0,
                "mean_level": 0.5,
                "contrast": 0.0,
                "available": False,
            }

    @staticmethod
    def is_severe(stats: Dict[str, float]) -> bool:
        """True when the histogram independently confirms unrecoverable exposure."""
        return stats.get("available", False) and stats.get("category") in {
            "badly_over",
            "badly_under",
        }
