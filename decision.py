"""Turning analysis into a Keep / Review / Delete call.

The governing asymmetry: a photograph wrongly sent to Review costs a few seconds of the
photographer's attention, while one wrongly sent to Delete may be gone for good. So
Delete has to clear a high bar and Review is the honest answer for everything uncertain.
A large Review pile is not a failure of the tool.

Every delete needs two independent witnesses -- the vision model's judgement and a
measurement that corroborates it. Neither one alone can reject a frame.
"""

from typing import Dict, List, Optional, Tuple

try:
    from .models import CullResult, ImageMetrics
except ImportError:
    from models import CullResult, ImageMetrics

KEEP = "Keep"
REVIEW = "Review"
DELETE = "Delete"
FAILED = "Failed"

# Phrases that describe a frame no amount of editing rescues. Matched as substrings
# against the model's technical_issues, which is why they are short and generic.
# Technical sharpness below this means the frame contains no resolved detail at all.
# Well under blur_detector.SHARP_EVIDENCE_THRESHOLD, because a merely low score is not
# evidence of anything (see that module's docstring) -- this is the floor.
NO_DETAIL_THRESHOLD = 0.20

FATAL_ISSUE_HINTS = (
    "lens cap",
    "finger over lens",
    "accidental",
    "pocket shot",
    "completely black",
    "completely white",
)


class CullDecider:
    """Applies the two-witness rule to one analysed photograph."""

    def __init__(self, sharp_evidence_vetoes_delete: bool = True):
        self.sharp_evidence_vetoes_delete = sharp_evidence_vetoes_delete

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _has_fatal_issue(metrics: ImageMetrics) -> bool:
        text = " ".join(metrics.technical_issues).lower()
        return any(hint in text for hint in FATAL_ISSUE_HINTS)

    @staticmethod
    def _measurement_confirms_blur(metrics: ImageMetrics) -> bool:
        """True when OpenCV found no critically sharp region anywhere in the frame.

        Low technical sharpness is weak evidence on its own (see blur_detector.py), so
        this only ever runs alongside a vision verdict that already said delete.
        """
        if metrics.cv_sharpness is None:
            return False
        return not _has_sharp_evidence(metrics.cv_sharpness)

    @staticmethod
    def _measurement_confirms_exposure(metrics: ImageMetrics) -> bool:
        """True when the histogram independently shows unrecoverable exposure."""
        blown = metrics.blown_fraction
        crushed = metrics.crushed_fraction
        if blown is None or crushed is None:
            return False
        return blown >= 0.15 or crushed >= 0.15

    def _sharp_evidence_blocks_delete(self, metrics: ImageMetrics) -> bool:
        if not self.sharp_evidence_vetoes_delete or metrics.cv_sharpness is None:
            return False
        return _has_sharp_evidence(metrics.cv_sharpness)

    @staticmethod
    def _measurement_contradicts_good_exposure(metrics: ImageMetrics) -> bool:
        """Histogram found unrecoverable clipping while the model reported no problem."""
        if metrics.exposure not in ("good", "slightly_off", ""):
            return False
        return CullDecider._measurement_confirms_exposure(metrics)

    @staticmethod
    def _measurement_finds_no_detail(metrics: ImageMetrics) -> bool:
        """Nothing anywhere in the frame resolves, yet the model was happy with it.

        The bar sits far below the sharp-evidence threshold because a low technical
        score has innocent explanations -- fog, snow, deliberate softness. Only a frame
        with essentially no fine detail at all trips this.
        """
        return metrics.cv_sharpness is not None and metrics.cv_sharpness < NO_DETAIL_THRESHOLD

    # ------------------------------------------------------------------ decision

    def decide(self, metrics: ImageMetrics) -> Tuple[str, float, List[str]]:
        issues = self.describe_issues(metrics)
        verdict = (metrics.verdict or "review").lower()
        sharpness = metrics.subject_sharpness
        exposure = metrics.exposure

        if verdict == "delete":
            # Witness 1 is the model. Find a second one before acting on it.
            if self._has_fatal_issue(metrics):
                return DELETE, 0.90, issues

            blur_case = sharpness == "unusable" and self._measurement_confirms_blur(metrics)
            exposure_case = (
                exposure in ("badly_under", "badly_over")
                and self._measurement_confirms_exposure(metrics)
            )

            if blur_case and self._sharp_evidence_blocks_delete(metrics):
                # The model called it unusable but the frame provably contains crisp
                # detail. Shallow depth of field reads as "blurry" to a model looking at
                # a downsample; trust the pixels and let a human settle it.
                issues.append("model and measurement disagree on sharpness")
                return REVIEW, 0.50, issues

            if blur_case or exposure_case:
                return DELETE, 0.85, issues

            # Model wants it gone but nothing corroborates. Not enough to delete.
            return REVIEW, 0.55, issues

        if verdict == "keep":
            # Vision models read exposure poorly -- in testing they called frames three
            # stops under "good". The histogram does not have that problem, so where the
            # two disagree on exposure the measurement wins and the frame gets a second
            # look. It is still only one witness, so this can raise a concern but never
            # delete.
            if self._measurement_contradicts_good_exposure(metrics):
                issues.append("histogram shows clipping the model did not report")
                return REVIEW, 0.55, issues

            if self._measurement_finds_no_detail(metrics):
                issues.append("no critically sharp region found in frame")
                return REVIEW, 0.55, issues

            if sharpness in ("sharp", "acceptable") and exposure == "good":
                confidence = 0.90 if sharpness == "sharp" else 0.75
                if metrics.framing == "broken":
                    return REVIEW, 0.55, issues
                return KEEP, confidence, issues

            if sharpness == "unusable" or exposure in ("badly_under", "badly_over"):
                # Model said keep while flagging something serious. Contradictory.
                return REVIEW, 0.50, issues

            return KEEP, 0.65, issues

        return REVIEW, 0.50, issues

    # ------------------------------------------------------------------ reporting

    @staticmethod
    def describe_issues(metrics: ImageMetrics) -> List[str]:
        issues: List[str] = []

        if metrics.subject_sharpness == "unusable":
            issues.append("subject out of focus")
        elif metrics.subject_sharpness == "soft":
            issues.append("soft focus")

        if metrics.exposure == "badly_under":
            issues.append("severely underexposed")
        elif metrics.exposure == "badly_over":
            issues.append("severely overexposed")
        elif metrics.exposure == "slightly_off":
            issues.append("exposure slightly off")

        if metrics.framing == "broken":
            issues.append("framing broken")
        elif metrics.framing == "weak":
            issues.append("weak framing")

        for issue in metrics.technical_issues:
            cleaned = str(issue).strip().lower()
            if cleaned and cleaned not in issues:
                issues.append(cleaned)

        return issues


def _has_sharp_evidence(cv_score: float) -> bool:
    try:
        from .blur_detector import SHARP_EVIDENCE_THRESHOLD
    except ImportError:
        from blur_detector import SHARP_EVIDENCE_THRESHOLD
    return cv_score >= SHARP_EVIDENCE_THRESHOLD


def suggested_rating(metrics: ImageMetrics) -> int:
    """1-5 star suggestion for photo apps, from the overall quality figure."""
    quality = metrics.overall_quality
    if quality >= 0.85:
        return 5
    if quality >= 0.70:
        return 4
    if quality >= 0.50:
        return 3
    if quality >= 0.30:
        return 2
    return 1
