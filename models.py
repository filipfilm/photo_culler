from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Categorical vocabularies the vision model is constrained to (see vision.py schemas).
SHARPNESS_LEVELS = ("sharp", "acceptable", "soft", "unusable")
EXPOSURE_LEVELS = ("good", "slightly_off", "badly_under", "badly_over")
FRAMING_LEVELS = ("strong", "fine", "weak", "broken")
VERDICTS = ("keep", "review", "delete")

# Categories are what the model can actually judge reliably; these numbers exist only
# so the sidecar writers, CSV and star-rating suggestions have something continuous to
# work with. Never threshold on them to make a decision -- decide on the category.
SHARPNESS_SCORES = {"sharp": 0.90, "acceptable": 0.70, "soft": 0.40, "unusable": 0.10}
EXPOSURE_SCORES = {"good": 0.90, "slightly_off": 0.65, "badly_under": 0.20, "badly_over": 0.20}
FRAMING_SCORES = {"strong": 0.90, "fine": 0.65, "weak": 0.40, "broken": 0.15}


@dataclass
class ImageMetrics:
    """What we learned about one photo.

    The float scores are derived from the categorical judgements and kept for
    backwards compatibility with the sidecar/CSV writers.
    """

    blur_score: float  # 0-1, higher is sharper
    exposure_score: float  # 0-1, higher is better exposed
    composition_score: float  # 0-1, higher is more interesting
    overall_quality: float  # 0-1, weighted combination
    keywords: Optional[List[str]] = None
    description: Optional[str] = None

    # Categorical judgements from the vision model.
    subject: str = ""
    subject_sharpness: str = ""
    exposure: str = ""
    framing: str = ""
    technical_issues: List[str] = field(default_factory=list)
    verdict: str = ""
    verdict_reason: str = ""

    # Independent measurements, used to cross-check the model (see blur_detector.py
    # and exposure.py for why each one is trusted in only one direction).
    cv_sharpness: Optional[float] = None
    blown_fraction: Optional[float] = None
    crushed_fraction: Optional[float] = None

    @classmethod
    def from_triage(
        cls,
        triage: Dict,
        cv_sharpness: Optional[float] = None,
        exposure_stats: Optional[Dict] = None,
    ) -> "ImageMetrics":
        sharpness = triage.get("subject_sharpness", "acceptable")
        exposure = triage.get("exposure", "good")
        framing = triage.get("framing", "fine")

        blur_score = SHARPNESS_SCORES.get(sharpness, 0.5)
        exposure_score = EXPOSURE_SCORES.get(exposure, 0.5)
        composition_score = FRAMING_SCORES.get(framing, 0.5)
        overall = 0.55 * blur_score + 0.30 * exposure_score + 0.15 * composition_score

        stats = exposure_stats or {}
        return cls(
            blur_score=blur_score,
            exposure_score=exposure_score,
            composition_score=composition_score,
            overall_quality=round(overall, 3),
            keywords=[],
            description="",
            subject=triage.get("subject", ""),
            subject_sharpness=sharpness,
            exposure=exposure,
            framing=framing,
            technical_issues=list(triage.get("technical_issues", []) or []),
            verdict=triage.get("verdict", "review"),
            verdict_reason=triage.get("verdict_reason", ""),
            cv_sharpness=cv_sharpness,
            blown_fraction=stats.get("blown_fraction"),
            crushed_fraction=stats.get("crushed_fraction"),
        )


@dataclass
class CullResult:
    filepath: Path
    decision: str  # "Keep", "Delete", "Review", "Failed"
    confidence: float  # 0-1
    metrics: ImageMetrics
    issues: List[str] = field(default_factory=list)
    processing_ms: float = 0.0

    # Captured during analysis, consumed by the burst grouping pass.
    capture_time: Optional[datetime] = None
    phash: Optional[str] = None

    # Set by the burst grouping pass (grouping.py), not by per-image analysis.
    group_id: Optional[int] = None
    group_size: int = 1
    is_best_of_group: bool = True
    duplicate_of: Optional[str] = None
