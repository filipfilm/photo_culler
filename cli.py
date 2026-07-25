"""Shared command-line implementation for the ON1 and universal entry points.

The two front ends previously duplicated about ninety percent of their logic, so a fix
applied to one silently missed the other. They now differ only in which sidecar they
write.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
import csv
import logging

import click
from tqdm import tqdm

try:
    from .batch import BatchCuller
    from .config import Config
    from .decision import DELETE, FAILED, KEEP, REVIEW
    from .grouping import IMAGEHASH_AVAILABLE
    from .models import CullResult
    from .sidecars import write_sidecar
    from .vision import ModelCannotSee, VisionUnavailable, detect_vision_model
except ImportError:
    from batch import BatchCuller
    from config import Config
    from decision import DELETE, FAILED, KEEP, REVIEW
    from grouping import IMAGEHASH_AVAILABLE
    from models import CullResult
    from sidecars import write_sidecar
    from vision import ModelCannotSee, VisionUnavailable, detect_vision_model

CSV_FIELDS = [
    "timestamp", "filepath", "filename", "decision", "confidence", "issues",
    "subject", "subject_sharpness", "exposure", "framing",
    "blur_score", "exposure_score", "composition_score", "overall_score",
    "measured_sharpness", "burst_size", "best_of_burst", "duplicate_of",
    "description", "keywords", "processing_ms",
]


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s - %(message)s",
    )


def common_options(func):
    """CLI flags shared by every entry point."""
    options = [
        click.argument("folder", type=click.Path(exists=True, file_okay=False)),
        click.option("--fast", is_flag=True,
                     help="Measurement-only triage, no model. Sorts into Keep/Review; never deletes."),
        click.option("--model", "ollama_model", default=None,
                     help="Ollama vision model. Default: auto-detect the best installed one."),
        click.option("--host", default=None, help="Ollama host URL."),
        click.option("--cache-dir", type=click.Path(), default=None,
                     help="Reuse analysis for unchanged files across runs."),
        click.option("--extensions", default=None,
                     help="Comma-separated list, e.g. nef,jpg. Default comes from config.yaml."),
        click.option("--workers", type=int, default=None,
                     help="Parallel requests to Ollama."),
        click.option("--no-tags", is_flag=True,
                     help="Skip the description/keyword pass. Roughly twice as fast."),
        click.option("--no-grouping", is_flag=True,
                     help="Do not group bursts or demote near-duplicates."),
        click.option("--no-recursive", is_flag=True, help="Do not descend into subfolders."),
        click.option("--override", is_flag=True,
                     help="Replace existing keywords and descriptions. Ratings are kept."),
        click.option("--dry-run", is_flag=True,
                     help="Analyse and report without writing sidecars or moving anything."),
        click.option("--move-deletes", is_flag=True,
                     help="Move confident deletions into _culled_deletes/ (files are moved, not erased)."),
        click.option("--csv-file", type=click.Path(), default=None,
                     help="Write results here instead of a timestamped file in cull_runs/."),
        click.option("--detail", is_flag=True, help="Print every photo as it is decided."),
        click.option("--verbose", is_flag=True, help="Debug logging."),
        click.option("--skip-vision-check", is_flag=True,
                     help="Skip the startup proof that the model can see. Not recommended."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _decision_icon(decision: str) -> str:
    return {KEEP: "🟢", REVIEW: "🟡", DELETE: "🔴", FAILED: "⚫"}.get(decision, "  ")


def print_result(result: CullResult):
    metrics = result.metrics
    print(f"{_decision_icon(result.decision)} {result.decision:<7} {result.filepath.name}"
          f"   ({result.confidence:.2f})")
    if metrics.subject:
        print(f"     subject: {metrics.subject}")
    print(f"     sharpness={metrics.subject_sharpness or '-'} "
          f"exposure={metrics.exposure or '-'} framing={metrics.framing or '-'}")
    if result.issues:
        print(f"     issues: {', '.join(result.issues)}")
    if metrics.description:
        print(f"     {metrics.description}")
    if metrics.keywords:
        print(f"     keywords: {', '.join(metrics.keywords)}")


def write_csv(path: Path, results: Sequence[CullResult]):
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            metrics = result.metrics
            writer.writerow({
                "timestamp": now,
                "filepath": str(result.filepath),
                "filename": result.filepath.name,
                "decision": result.decision,
                "confidence": f"{result.confidence:.2f}",
                "issues": "; ".join(result.issues),
                "subject": metrics.subject,
                "subject_sharpness": metrics.subject_sharpness,
                "exposure": metrics.exposure,
                "framing": metrics.framing,
                "blur_score": f"{metrics.blur_score:.2f}",
                "exposure_score": f"{metrics.exposure_score:.2f}",
                "composition_score": f"{metrics.composition_score:.2f}",
                "overall_score": f"{metrics.overall_quality:.2f}",
                "measured_sharpness": (
                    f"{metrics.cv_sharpness:.2f}" if metrics.cv_sharpness is not None else ""
                ),
                "burst_size": result.group_size,
                "best_of_burst": "yes" if result.is_best_of_group else "no",
                "duplicate_of": result.duplicate_of or "",
                "description": metrics.description or "",
                "keywords": ", ".join(metrics.keywords or []),
                "processing_ms": f"{result.processing_ms:.0f}",
            })


def move_deletions(results: Sequence[CullResult], folder: Path, min_confidence: float = 0.8) -> int:
    """Move confident deletions aside. Nothing is ever erased."""
    candidates = [
        r for r in results if r.decision == DELETE and r.confidence >= min_confidence
    ]
    if not candidates:
        return 0

    trash = folder / "_culled_deletes"
    trash.mkdir(exist_ok=True)

    moved = 0
    for result in candidates:
        try:
            destination = trash / result.filepath.name
            if destination.exists():
                print(f"  skipping {result.filepath.name}: already in _culled_deletes")
                continue
            result.filepath.rename(destination)

            # Keep sidecars with their photo, otherwise the catalogue is left with
            # orphans pointing at a file that moved.
            for companion in (
                result.filepath.with_suffix(".on1"),
                result.filepath.with_suffix(".xmp"),
                Path(str(result.filepath) + ".xmp"),
            ):
                if companion.exists():
                    companion.rename(trash / companion.name)
            moved += 1
        except OSError as e:
            print(f"  could not move {result.filepath.name}: {e}")

    return moved


def summarise(results: Sequence[CullResult], culler: BatchCuller, elapsed: float):
    total = len(results)
    counts = {d: 0 for d in (KEEP, REVIEW, DELETE, FAILED)}
    for result in results:
        counts[result.decision] = counts.get(result.decision, 0) + 1

    print("\n" + "=" * 62)
    print("CULLING COMPLETE")
    print("=" * 62)
    for decision in (KEEP, REVIEW, DELETE, FAILED):
        count = counts.get(decision, 0)
        if count:
            print(f"  {_decision_icon(decision)} {decision:<7} {count:>5}  ({count / total * 100:4.1f}%)")

    grouping = getattr(culler, "grouping_summary", None)
    if grouping and grouping.get("bursts"):
        print(f"\n  {grouping['bursts']} bursts found; "
              f"{grouping['demoted_to_review']} weaker frames moved to Review"
              + (f", {grouping['near_identical']} near-identical"
                 if grouping.get("near_identical") else ""))

    deletions = [r for r in results if r.decision == DELETE]
    if deletions:
        print(f"\n  Deletion candidates ({len(deletions)}) - check these before removing anything:")
        for result in sorted(deletions, key=lambda r: -r.confidence)[:10]:
            print(f"    {result.filepath.name}  {', '.join(result.issues) or 'no reason given'}")
        if len(deletions) > 10:
            print(f"    ... and {len(deletions) - 10} more")

    per_image = elapsed / total if total else 0
    print(f"\n  {total} photos in {elapsed / 60:.1f} min ({per_image:.1f}s each)")


def run_cull(sidecar_style: Optional[str], **kwargs) -> int:
    """Shared entry point. sidecar_style is 'on1', 'xmp', or None for CSV only."""
    setup_logging(kwargs.get("verbose", False))
    config = Config.load()

    folder = Path(kwargs["folder"])
    fast = kwargs.get("fast", False)
    dry_run = kwargs.get("dry_run", False)
    detail = kwargs.get("detail", False)

    host = kwargs.get("host") or config.host
    # Fast mode is CPU-only and parallelises freely; the model path does not (see the
    # workers comment in config.yaml).
    workers = kwargs.get("workers") or (config.fast_workers if fast else config.workers)
    extensions = config.normalized_extensions(kwargs.get("extensions"))
    recursive = config.recursive and not kwargs.get("no_recursive", False)
    with_tags = config.tagging and not kwargs.get("no_tags", False)
    grouping = config.grouping_enabled and not kwargs.get("no_grouping", False)

    model = kwargs.get("ollama_model") or config.model_name
    if not fast and not model:
        try:
            model = detect_vision_model(host)
        except VisionUnavailable as e:
            print(f"\n{e}\n")
            return 1

    print(f"  folder      {folder}")
    print(f"  mode        {'fast (measurement only, no deletions)' if fast else 'accurate'}")
    if not fast:
        print(f"  model       {model or 'auto-detect'}")
        print(f"  tagging     {'on' if with_tags else 'off'}")
    print(f"  sidecars    {sidecar_style or 'none'}{' (dry run)' if dry_run else ''}")
    print(f"  grouping    {'on' if grouping else 'off'}"
          + ("" if IMAGEHASH_AVAILABLE else "  [ImageHash missing: time-only grouping]"))
    print("=" * 62)

    try:
        culler = BatchCuller(
            cache_dir=Path(kwargs["cache_dir"]) if kwargs.get("cache_dir") else None,
            mode="fast" if fast else "accurate",
            max_workers=workers,
            use_ollama=not fast,
            ollama_model=model,
            ollama_host=host,
            timeout=config.timeout_seconds,
            context_tokens=config.context_tokens,
            with_tags=with_tags,
            verify_vision=config.verify_vision and not kwargs.get("skip_vision_check", False),
        )
    except (VisionUnavailable, ModelCannotSee) as e:
        print(f"\n{e}\n")
        return 1
    except Exception as e:
        print(f"\nCould not start the culler: {e}\n")
        return 1

    files = culler.find_image_files(folder, extensions, recursive)
    if not files:
        print(f"No images with extensions {', '.join(extensions)} found in {folder}")
        return 1
    print(f"  {len(files)} photos to analyse\n")

    started = datetime.now()
    results = culler.cull_folder(
        folder, extensions, recursive=recursive, group_bursts=grouping
    )
    elapsed = (datetime.now() - started).total_seconds()

    if detail:
        print()
        for result in results:
            print_result(result)

    sidecars_written = 0
    if sidecar_style and not dry_run:
        override = kwargs.get("override", False)
        for result in tqdm(results, desc="Writing metadata", unit="file"):
            if result.decision == FAILED:
                continue
            if write_sidecar(result, sidecar_style, override):
                sidecars_written += 1

    csv_path = (
        Path(kwargs["csv_file"]) if kwargs.get("csv_file")
        else folder / config.csv_dir / f"cull_{started:%Y%m%d_%H%M%S}.csv"
    )
    if not dry_run:
        write_csv(csv_path, results)

    summarise(results, culler, elapsed)

    if sidecar_style and not dry_run:
        label = "ON1 sidecars updated" if sidecar_style == "on1" else "XMP sidecars written"
        print(f"\n  {label}: {sidecars_written}/{len(results)}")
        if sidecar_style == "on1" and sidecars_written < len(results):
            print("    (ON1 must have created a .on1 file before the culler can update it)")
    if not dry_run:
        print(f"  results: {csv_path}")

    if kwargs.get("move_deletes") and not dry_run:
        moved = move_deletions(results, folder)
        print(f"  moved {moved} confident deletions into {folder / '_culled_deletes'}")

    print(f"\n  Search your photo app for PhotoCuller:Review to work through the "
          f"{sum(1 for r in results if r.decision == REVIEW)} uncertain frames.")
    return 0
