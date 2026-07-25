"""Technical sharpness measurement, used as a cross-check on the vision model.

Design note -- why this module is deliberately one-sided:

The obvious approach (average edge energy over the whole frame) punishes exactly the
photographs a culler must not throw away: shallow depth of field, fog, snow, minimal
compositions, soft film scans. Those images are mostly smooth by intent, so a global
sharpness average reads "blurry" even when the subject is critically sharp.

So instead of asking "is this image sharp on average", we ask the question that has a
reliable answer: **is any part of this frame critically sharp?** We tile the frame and
take the sharpest tile. A photo with one crisp eye and a wash of bokeh scores high, as
it should. A genuinely out-of-focus frame has no sharp tile anywhere and scores low.

That gives an asymmetric signal, and the caller must treat it that way:

  high score -> trustworthy positive. Real detail exists; the frame is not soft.
  low score  -> NOT trustworthy as evidence of blur. A flat grey sky scores low too.

Consequently this module can veto a deletion but must never cause one. See
batch.py::CullDecider.
"""

from typing import Dict, Optional
import logging
import math

import numpy as np
from PIL import Image

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# All measurements happen at this long edge so the calibration constants below mean
# the same thing for a 12MP phone shot and a 61MP RAW.
ANALYSIS_EDGE = 1536

# Frame is split into TILES x TILES cells; the sharpest cells decide the score.
TILES = 8
TOP_TILES = 3  # average of the N sharpest tiles, so one speck of noise cannot win

# Laplacian variance of the sharpest tiles, mapped through log10 onto 0-1.
# Calibrated on real photographs at ANALYSIS_EDGE: a heavily defocused frame sits
# near 10, an ordinary in-focus frame near 300, a crisp detailed one above 2000.
PEAK_VAR_FLOOR = 10.0
PEAK_VAR_CEILING = 2000.0

# Above this score the frame demonstrably contains critically sharp detail, which is
# enough to block a blur-based deletion.
SHARP_EVIDENCE_THRESHOLD = 0.55


class BlurDetector:
    """Measures whether any region of a frame is critically sharp."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available = CV2_AVAILABLE
        if not CV2_AVAILABLE:
            self.logger.warning(
                "OpenCV not installed - technical sharpness cross-check disabled. "
                "Install with: pip install opencv-python"
            )

    def _to_gray_array(self, image: Image.Image) -> np.ndarray:
        working = image.copy()
        if working.mode != "L":
            working = working.convert("L")
        working.thumbnail((ANALYSIS_EDGE, ANALYSIS_EDGE), Image.Resampling.LANCZOS)
        return np.asarray(working, dtype=np.uint8)

    def _tile_variances(self, laplacian: np.ndarray) -> np.ndarray:
        height, width = laplacian.shape
        tile_h = max(1, height // TILES)
        tile_w = max(1, width // TILES)

        variances = []
        for row in range(TILES):
            for col in range(TILES):
                y0, x0 = row * tile_h, col * tile_w
                y1 = height if row == TILES - 1 else y0 + tile_h
                x1 = width if col == TILES - 1 else x0 + tile_w
                tile = laplacian[y0:y1, x0:x1]
                if tile.size:
                    variances.append(tile.var())

        return np.array(variances, dtype=np.float64)

    def measure(self, image: Image.Image) -> Dict[str, float]:
        """Return sharpness evidence for one frame.

        Keys: sharpness_score (0-1), peak_tile_var, global_var, has_sharp_evidence.
        """
        if not CV2_AVAILABLE:
            return {
                "sharpness_score": None,
                "peak_tile_var": 0.0,
                "global_var": 0.0,
                "has_sharp_evidence": False,
                "available": False,
            }

        try:
            gray = self._to_gray_array(image)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            tile_vars = self._tile_variances(laplacian)
            if tile_vars.size == 0:
                raise ValueError("image too small to tile")

            top = np.sort(tile_vars)[-min(TOP_TILES, tile_vars.size) :]
            peak_var = float(top.mean())
            global_var = float(laplacian.var())

            span = math.log10(PEAK_VAR_CEILING) - math.log10(PEAK_VAR_FLOOR)
            normalized = (math.log10(max(peak_var, 1e-6)) - math.log10(PEAK_VAR_FLOOR)) / span
            score = float(np.clip(normalized, 0.0, 1.0))

            return {
                "sharpness_score": score,
                "peak_tile_var": peak_var,
                "global_var": global_var,
                "has_sharp_evidence": score >= SHARP_EVIDENCE_THRESHOLD,
                "available": True,
            }

        except Exception as e:
            self.logger.warning(f"Technical sharpness measurement failed: {e}")
            return {
                "sharpness_score": None,
                "peak_tile_var": 0.0,
                "global_var": 0.0,
                "has_sharp_evidence": False,
                "available": False,
            }

    def score(self, image: Image.Image) -> Optional[float]:
        """Convenience wrapper returning just the 0-1 score (None if unavailable)."""
        return self.measure(image)["sharpness_score"]


# Backwards-compatible alias; the old class name is referenced by older scripts.
HybridBlurDetector = BlurDetector
