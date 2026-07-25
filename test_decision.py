#!/usr/bin/env python3
"""Tests for the two-witness delete rule.

These encode the invariants that the previous version violated. Run with pytest, or
directly: python test_decision.py
"""

import sys

try:
    from .blur_detector import SHARP_EVIDENCE_THRESHOLD
    from .decision import CullDecider, DELETE, KEEP, REVIEW
    from .models import ImageMetrics
except ImportError:
    from blur_detector import SHARP_EVIDENCE_THRESHOLD
    from decision import CullDecider, DELETE, KEEP, REVIEW
    from models import ImageMetrics

SHARP = SHARP_EVIDENCE_THRESHOLD + 0.2
NOT_SHARP = SHARP_EVIDENCE_THRESHOLD - 0.3


def metrics(**overrides) -> ImageMetrics:
    """A photograph the model considers fine, unless overridden."""
    triage = {
        "subject": "a subject",
        "subject_sharpness": "sharp",
        "exposure": "good",
        "framing": "fine",
        "technical_issues": [],
        "verdict": "keep",
        "verdict_reason": "",
    }
    exposure_stats = {"blown_fraction": 0.0, "crushed_fraction": 0.0}

    for key in ("cv_sharpness",):
        overrides.setdefault(key, SHARP)
    cv = overrides.pop("cv_sharpness")
    exposure_stats.update(overrides.pop("exposure_stats", {}))
    triage.update(overrides)

    return ImageMetrics.from_triage(triage, cv_sharpness=cv, exposure_stats=exposure_stats)


decider = CullDecider()


def decide(**overrides) -> str:
    return decider.decide(metrics(**overrides))[0]


# --------------------------------------------------------------- the expensive error


def test_shallow_dof_is_never_deleted():
    """The original bug: a sharp subject against a soft background, measured low.

    A global sharpness average reads this as blurry. Because the frame still contains a
    critically sharp region, it must survive even when the model votes to delete.
    """
    assert decide(
        verdict="delete", subject_sharpness="unusable", cv_sharpness=SHARP
    ) == REVIEW


def test_model_alone_cannot_delete():
    """No corroborating measurement means no deletion, whatever the model says."""
    assert decide(verdict="delete", subject_sharpness="soft", cv_sharpness=SHARP) == REVIEW
    assert decide(verdict="delete", subject_sharpness="acceptable") == REVIEW


def test_measurement_alone_cannot_delete():
    """A frame measuring as featureless is not deleted while the model is happy."""
    assert decide(verdict="keep", cv_sharpness=0.0) == REVIEW


def test_low_contrast_sharp_photo_survives():
    """Film scans and foggy scenes: soft-looking, genuinely sharp, must not be deleted."""
    for verdict in ("keep", "review", "delete"):
        assert decide(verdict=verdict, subject_sharpness="soft", cv_sharpness=SHARP) != DELETE


# --------------------------------------------------------------- deletion that should happen


def test_two_witnesses_delete():
    """Model says unusable and nothing in the frame resolves: that is a delete."""
    assert decide(
        verdict="delete", subject_sharpness="unusable", cv_sharpness=NOT_SHARP
    ) == DELETE


def test_confirmed_exposure_deletes():
    """Model and histogram agree the exposure is beyond recovery."""
    assert decide(
        verdict="delete",
        subject_sharpness="acceptable",
        exposure="badly_over",
        exposure_stats={"blown_fraction": 0.4},
    ) == DELETE


def test_fatal_issue_deletes():
    """Some frames are accidents and need no second witness."""
    assert decide(
        verdict="delete", technical_issues=["lens cap on"], cv_sharpness=SHARP
    ) == DELETE


# --------------------------------------------------------------- measurement raising concern


def test_histogram_overrides_model_on_exposure():
    """Models read exposure badly, so clipping the model missed still gets a look."""
    assert decide(
        verdict="keep", exposure="good", exposure_stats={"crushed_fraction": 0.3}
    ) == REVIEW


def test_clean_photo_is_kept():
    assert decide() == KEEP
    assert decide(subject_sharpness="acceptable") == KEEP


def test_broken_framing_goes_to_review():
    assert decide(framing="broken") == REVIEW


def test_contradictory_keep_goes_to_review():
    """Model votes keep while calling the subject unusable. Let a human settle it."""
    assert decide(verdict="keep", subject_sharpness="unusable", cv_sharpness=NOT_SHARP) == REVIEW


# --------------------------------------------------------------- degraded environments


def test_no_opencv_means_no_blur_deletions():
    """Without a measurement there is only one witness, so blur cannot delete."""
    assert decide(
        verdict="delete", subject_sharpness="unusable", cv_sharpness=None
    ) == REVIEW


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []

    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError:
            print(f"  FAIL  {test.__name__}")
            failures.append(test.__name__)
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failures.append(test.__name__)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
