#!/usr/bin/env python3
"""Regression test for the culling pipeline.

Builds a ground-truth set from photographs you supply -- each one kept as-is, and also
heavily defocused, three stops under and three stops over -- then checks two things:

  1. No original is ever sent to Delete.        (the expensive error)
  2. No ruined frame is ever sent to Keep.      (the useless error)

Criterion 1 is the one that matters. The original tool failed it on sharp low-contrast
photographs, which is what prompted this harness existing at all.

    python eval_harness.py ~/Photos/some-folder
    python eval_harness.py ~/Photos/some-folder --model qwen3-vl:8b-instruct
    python eval_harness.py ~/Photos/some-folder --measurement-only
"""

from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile
import time

import click
from PIL import Image, ImageEnhance, ImageFilter

try:
    from .batch import BatchCuller
    from .decision import DELETE, FAILED, KEEP
    from .extractor import RawThumbnailExtractor
except ImportError:
    from batch import BatchCuller
    from decision import DELETE, FAILED, KEEP
    from extractor import RawThumbnailExtractor

# Enough blur that no reasonable viewer would call the frame usable.
DEFOCUS_RADIUS = 14
UNDEREXPOSE = 0.13
OVEREXPOSE = 3.2

VARIANTS = {
    "original": lambda img: img,
    "defocus": lambda img: img.filter(ImageFilter.GaussianBlur(DEFOCUS_RADIUS)),
    "underexposed": lambda img: ImageEnhance.Brightness(img).enhance(UNDEREXPOSE),
    "overexposed": lambda img: ImageEnhance.Brightness(img).enhance(OVEREXPOSE),
}


def build_set(sources, workdir: Path) -> dict:
    """Write the variants and return {filename: variant_kind}."""
    extractor = RawThumbnailExtractor()
    truth = {}

    for source in sources:
        image = extractor.extract(source)
        if image is None:
            click.echo(f"  skipping unreadable {source.name}")
            continue
        image = image.convert("RGB")

        for kind, transform in VARIANTS.items():
            name = f"{source.stem}__{kind}.jpg"
            transform(image).save(workdir / name, quality=92)
            truth[name] = kind

    return truth


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--model", default=None, help="Ollama model. Default: auto-detect.")
@click.option("--count", default=6, help="How many source photographs to use.")
@click.option("--measurement-only", is_flag=True,
              help="Test fast mode (no vision model). Only criterion 1 applies.")
@click.option("--keep-files", is_flag=True, help="Leave the generated images in place.")
def main(folder, model, count, measurement_only, keep_files):
    """Check the pipeline against photographs whose correct handling is known."""
    folder = Path(folder)
    extensions = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".jpg", ".jpeg"}
    sources = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )[:count]

    if not sources:
        click.echo(f"No photographs found in {folder}")
        sys.exit(1)

    workdir = Path(tempfile.mkdtemp(prefix="photo_culler_eval_"))
    click.echo(f"Building ground-truth set from {len(sources)} photographs...")
    truth = build_set(sources, workdir)
    click.echo(f"  {len(truth)} images in {workdir}\n")

    try:
        culler = BatchCuller(
            mode="fast" if measurement_only else "accurate",
            use_ollama=not measurement_only,
            ollama_model=model,
            with_tags=False,
        )
    except Exception as e:
        click.echo(f"Could not start the culler:\n{e}")
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(1)

    click.echo(f"mode: {'fast (measurement only)' if measurement_only else culler.ollama_model}\n")

    started = time.time()
    # Grouping off: the variants of one photograph are near-identical by construction,
    # so burst demotion would move them to Review and mask what we are testing.
    results = culler.cull_folder(workdir, [".jpg"], group_bursts=False)
    elapsed = time.time() - started

    by_kind = defaultdict(lambda: defaultdict(int))
    false_deletes = []
    missed = []

    for result in results:
        kind = truth.get(result.filepath.name, "?")
        by_kind[kind][result.decision] += 1

        if kind == "original" and result.decision == DELETE:
            false_deletes.append(result)
        if kind != "original" and result.decision == KEEP:
            missed.append(result)

    click.echo("Decisions by kind:")
    for kind in VARIANTS:
        counts = dict(by_kind.get(kind, {}))
        click.echo(f"  {kind:14s} {counts}")

    originals = sum(by_kind["original"].values())
    ruined = sum(sum(by_kind[k].values()) for k in VARIANTS if k != "original")

    click.echo()
    ok = not false_deletes
    click.echo(f"[{'PASS' if ok else 'FAIL'}] criterion 1: originals sent to Delete: "
               f"{len(false_deletes)}/{originals}")
    for result in false_deletes:
        click.echo(f"         {result.filepath.name}: {', '.join(result.issues)}")

    if measurement_only:
        click.echo("       criterion 2 not applicable in fast mode (it never deletes)")
    else:
        second_ok = not missed
        ok = ok and second_ok
        click.echo(f"[{'PASS' if second_ok else 'FAIL'}] criterion 2: ruined frames left in Keep: "
                   f"{len(missed)}/{ruined}")
        for result in missed:
            click.echo(f"         {result.filepath.name}")

    failed = [r for r in results if r.decision == FAILED]
    if failed:
        click.echo(f"\n  {len(failed)} images failed to process")

    click.echo(f"\n  {len(results)} images in {elapsed:.0f}s ({elapsed / len(results):.1f}s each)")

    if keep_files:
        click.echo(f"  generated images left in {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
