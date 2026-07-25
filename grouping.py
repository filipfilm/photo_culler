"""Burst and near-duplicate grouping.

Culling a shoot is mostly a relative judgement. Fourteen frames of the same moment are
not fourteen independent decisions -- thirteen of them are there so that one can be
good. Scoring each frame in isolation, as the tool previously did, cannot express that.

Two frames belong to the same burst when they were taken close together in time *and*
look alike. Time alone would merge a fast-moving handheld sequence of different
subjects; appearance alone would merge every frame of a repeated setup across a whole
day. Requiring both keeps the groups honest.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence
import logging

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    IMAGEHASH_AVAILABLE = False

try:
    from .decision import DELETE, FAILED, KEEP, REVIEW
    from .models import CullResult
except ImportError:
    from decision import DELETE, FAILED, KEEP, REVIEW
    from models import CullResult

logger = logging.getLogger(__name__)

# Frames more than this far apart in time start a new burst.
DEFAULT_BURST_GAP_SECONDS = 3.0

# Perceptual-hash Hamming distance. Below NEAR_IDENTICAL the frames are effectively the
# same picture; below SIMILAR they are the same scene from the same position.
NEAR_IDENTICAL_DISTANCE = 4
SIMILAR_DISTANCE = 12

# Ranking within a burst: better verdict first, then sharper, then better exposed.
_VERDICT_RANK = {KEEP: 0, REVIEW: 1, DELETE: 2, FAILED: 3}
_SHARPNESS_RANK = {"sharp": 0, "acceptable": 1, "soft": 2, "unusable": 3, "": 2}


@dataclass
class PhotoGroup:
    group_id: int
    members: List[CullResult]
    best: CullResult

    @property
    def size(self) -> int:
        return len(self.members)


def compute_phash(image) -> Optional[str]:
    """Perceptual hash of one frame, or None when ImageHash is not installed."""
    if not IMAGEHASH_AVAILABLE:
        return None
    try:
        return str(imagehash.phash(image))
    except Exception as e:
        logger.warning(f"Perceptual hash failed: {e}")
        return None


def _hash_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b or not IMAGEHASH_AVAILABLE:
        return None
    try:
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    except Exception:
        return None


def _quality_key(result: CullResult):
    """Sort key picking the strongest frame in a burst. Lower is better."""
    metrics = result.metrics
    return (
        _VERDICT_RANK.get(result.decision, 3),
        _SHARPNESS_RANK.get(metrics.subject_sharpness, 2),
        # Within a burst the frames show the same scene, so comparing the technical
        # sharpness numbers directly is meaningful here even though comparing them
        # across different photographs would not be.
        -(metrics.cv_sharpness if metrics.cv_sharpness is not None else 0.0),
        0 if metrics.exposure == "good" else 1,
        -metrics.overall_quality,
        result.filepath.name,
    )


def group_photos(
    results: Sequence[CullResult],
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    similar_distance: int = SIMILAR_DISTANCE,
) -> List[PhotoGroup]:
    """Partition results into bursts of visually similar, near-simultaneous frames."""
    usable = [r for r in results if r.decision != FAILED]
    if not usable:
        return []

    # Order by capture time where known, falling back to filename, which for camera
    # output is chronological anyway.
    def order_key(result: CullResult):
        capture = getattr(result, "capture_time", None)
        return (capture is None, capture, result.filepath.name)

    ordered = sorted(usable, key=order_key)

    groups: List[PhotoGroup] = []
    current: List[CullResult] = [ordered[0]]

    for previous, result in zip(ordered, ordered[1:]):
        prev_time = getattr(previous, "capture_time", None)
        this_time = getattr(result, "capture_time", None)

        if prev_time and this_time:
            close_in_time = abs(this_time - prev_time) <= timedelta(seconds=burst_gap_seconds)
        else:
            close_in_time = False

        distance = _hash_distance(
            getattr(previous, "phash", None), getattr(result, "phash", None)
        )
        looks_alike = distance is not None and distance <= similar_distance

        # Without hashes available, time proximity alone has to carry the grouping.
        if distance is None:
            same_burst = close_in_time
        else:
            same_burst = close_in_time and looks_alike

        if same_burst:
            current.append(result)
        else:
            groups.append(_finalize_group(len(groups), current))
            current = [result]

    groups.append(_finalize_group(len(groups), current))
    return groups


def _finalize_group(group_id: int, members: List[CullResult]) -> PhotoGroup:
    best = min(members, key=_quality_key)
    return PhotoGroup(group_id=group_id, members=members, best=best)


def annotate_results(
    results: Sequence[CullResult],
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    demote_duplicates: bool = True,
) -> Dict[str, int]:
    """Tag each result with its burst, and optionally demote the also-rans.

    Demotion only ever moves a frame from Keep to Review. A redundant frame is not a
    bad frame -- it is one the photographer probably does not need, which is a
    suggestion, not grounds for deletion.
    """
    groups = group_photos(results, burst_gap_seconds=burst_gap_seconds)

    bursts = 0
    demoted = 0
    near_identical = 0

    for group in groups:
        for member in group.members:
            member.group_id = group.group_id
            member.group_size = group.size
            member.is_best_of_group = member is group.best

        if group.size == 1:
            continue
        bursts += 1

        for member in group.members:
            if member is group.best:
                continue

            distance = _hash_distance(
                getattr(group.best, "phash", None), getattr(member, "phash", None)
            )
            if distance is not None and distance <= NEAR_IDENTICAL_DISTANCE:
                member.duplicate_of = group.best.filepath.name
                near_identical += 1

            if demote_duplicates and member.decision == KEEP:
                member.decision = REVIEW
                member.confidence = min(member.confidence, 0.6)
                label = (
                    f"near-duplicate of {group.best.filepath.name}"
                    if member.duplicate_of
                    else f"weaker frame in burst of {group.size}"
                )
                if label not in member.issues:
                    member.issues.append(label)
                demoted += 1

    return {
        "groups": len(groups),
        "bursts": bursts,
        "demoted_to_review": demoted,
        "near_identical": near_identical,
        "hashing_available": IMAGEHASH_AVAILABLE,
    }
