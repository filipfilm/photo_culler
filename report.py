#!/usr/bin/env python3
"""Visual contact-sheet report for a culling run.

A CSV answers "what did it decide"; it cannot answer "was it right". At the scale this
tool is meant for, checking the verdicts means looking at the photographs, so this turns
a run's CSV into a single HTML page: every reject with a large thumbnail and its
reasons, the Review pile split into burst alternates versus genuinely flagged frames,
and the Keeps as a grid for spot-checking. Each thumbnail links to the original file.

    python report.py ~/Photos/Shoot                 # latest CSV in Shoot/cull_runs/
    python report.py ~/Photos/Shoot/cull_runs/cull_20260725_112928.csv

The report and its thumbnails are written next to the CSV. Thumbnails come from the
embedded RAW previews, so a few thousand photographs take minutes, not hours.
"""

from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
import csv
import hashlib
import sys

import click
from tqdm import tqdm

try:
    from .extractor import RawThumbnailExtractor
except ImportError:
    from extractor import RawThumbnailExtractor

THUMB_EDGE = 360
THUMB_QUALITY = 82

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
       background: #16181d; color: #d7dae0; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin: 28px 0 4px; }
h2 .count { color: #8a919e; font-weight: normal; }
p.note { color: #8a919e; margin: 2px 0 12px; font-size: 13px; }
.summary { display: flex; gap: 18px; margin: 10px 0 6px; flex-wrap: wrap; }
.summary div { padding: 8px 14px; border-radius: 8px; background: #1f232b; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
        gap: 10px; }
.grid.big { grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
.card { background: #1f232b; border-radius: 8px; overflow: hidden; }
.card img { width: 100%; height: 170px; object-fit: cover; display: block; }
.grid.big .card img { height: 250px; }
.card .body { padding: 7px 9px 9px; font-size: 12px; }
.card .name { font-weight: 600; }
.card .issues { color: #e0b060; margin-top: 3px; }
.card .subject { color: #8a919e; margin-top: 3px; }
.delete .name::before { content: "🔴 "; } .review .name::before { content: "🟡 "; }
.keep .name::before { content: "🟢 "; } .failed .name::before { content: "⚫ "; }
a { color: inherit; text-decoration: none; }
details { margin: 8px 0 20px; } summary { cursor: pointer; color: #8a919e; }
.missing { height: 170px; display: flex; align-items: center; justify-content: center;
           color: #555; background: #14161a; }
"""


def find_csv(target: Path) -> Path:
    if target.is_file():
        return target
    candidates = sorted((target / "cull_runs").glob("cull_*.csv"))
    if not candidates:
        raise click.ClickException(
            f"No cull_runs/cull_*.csv found under {target}. Run the culler first."
        )
    return candidates[-1]


def thumb_name(filepath: str) -> str:
    return hashlib.md5(filepath.encode()).hexdigest() + ".jpg"


def build_thumbs(rows, thumbs_dir: Path, workers: int = 8):
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    extractor = RawThumbnailExtractor()

    def one(row):
        source = Path(row["filepath"])
        destination = thumbs_dir / thumb_name(row["filepath"])
        if destination.exists() or not source.exists():
            return
        image = extractor.extract(source)
        if image is None:
            return
        image = image.convert("RGB")
        image.thumbnail((THUMB_EDGE, THUMB_EDGE))
        image.save(destination, quality=THUMB_QUALITY)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(one, rows), total=len(rows), desc="Thumbnails", unit="img"))


def card(row, thumbs_dir_name: str, big: bool = False) -> str:
    decision = row["decision"].lower()
    thumb = f"{thumbs_dir_name}/{thumb_name(row['filepath'])}"
    issues = escape(row.get("issues", ""))
    subject = escape(row.get("subject", "") or row.get("description", "")[:80])
    tooltip = escape(row.get("description", ""))

    burst = ""
    if row.get("burst_size", "1") not in ("", "1"):
        role = "best" if row.get("best_of_burst") == "yes" else "alt"
        burst = f" · burst {role} of {row['burst_size']}"

    return (
        f'<a class="card {decision}" href="file://{escape(row["filepath"])}" title="{tooltip}">'
        f'<img src="{thumb}" loading="lazy" alt="">'
        f'<div class="body"><div class="name">{escape(row["filename"])}'
        f' <span style="color:#8a919e">({row["confidence"]}{burst})</span></div>'
        + (f'<div class="issues">{issues}</div>' if issues else "")
        + (f'<div class="subject">{subject}</div>' if subject else "")
        + "</div></a>"
    )


def render(rows, thumbs_dir_name: str, csv_path: Path) -> str:
    deletes = [r for r in rows if r["decision"] == "Delete"]
    failed = [r for r in rows if r["decision"] == "Failed"]
    keeps = [r for r in rows if r["decision"] == "Keep"]
    review = [r for r in rows if r["decision"] == "Review"]
    burst_alts = [
        r for r in review
        if "burst of" in r.get("issues", "") or "near-duplicate" in r.get("issues", "")
    ]
    flagged = [r for r in review if r not in burst_alts]

    sections = [f"""
<h1>Culling report</h1>
<p class="note">{escape(str(csv_path))} — click any photo to open the original.</p>
<div class="summary">
  <div>🟢 Keep {len(keeps)}</div><div>🟡 Review {len(review)}
  <span style="color:#8a919e">({len(burst_alts)} burst alternates)</span></div>
  <div>🔴 Delete {len(deletes)}</div><div>⚫ Failed {len(failed)}</div>
</div>"""]

    sections.append(f'<h2>Rejects <span class="count">{len(deletes)}</span></h2>'
                    '<p class="note">Tagged PhotoCuller:Delete. Files untouched — every one '
                    'deserves a look before you act in your catalogue.</p>'
                    '<div class="grid big">'
                    + "".join(card(r, thumbs_dir_name, big=True) for r in deletes)
                    + "</div>")

    flagged.sort(key=lambda r: r.get("issues", ""))
    sections.append(f'<h2>Flagged for review <span class="count">{len(flagged)}</span></h2>'
                    '<p class="note">Sorted by issue so similar problems sit together.</p>'
                    '<div class="grid">'
                    + "".join(card(r, thumbs_dir_name) for r in flagged) + "</div>")

    sections.append(f'<h2>Burst alternates <span class="count">{len(burst_alts)}</span></h2>'
                    '<p class="note">A stronger frame of the same moment is already in Keep.</p>'
                    f'<details><summary>show {len(burst_alts)} frames</summary><div class="grid">'
                    + "".join(card(r, thumbs_dir_name) for r in burst_alts)
                    + "</div></details>")

    sections.append(f'<h2>Keeps <span class="count">{len(keeps)}</span></h2>'
                    f'<details open><summary>show {len(keeps)} frames</summary><div class="grid">'
                    + "".join(card(r, thumbs_dir_name) for r in keeps) + "</div></details>")

    if failed:
        sections.append(f'<h2>Failed to analyse <span class="count">{len(failed)}</span></h2>'
                        '<div class="grid">'
                        + "".join(card(r, thumbs_dir_name) for r in failed) + "</div>")

    return (f"<!doctype html><meta charset='utf-8'><title>Culling report</title>"
            f"<style>{CSS}</style>" + "".join(sections))


def generate_report(csv_path: Path, workers: int = 8) -> Path:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise click.ClickException(f"{csv_path} contains no results")

    report_path = csv_path.with_suffix(".html")
    thumbs_dir = csv_path.parent / (csv_path.stem + "_thumbs")
    build_thumbs(rows, thumbs_dir, workers)
    report_path.write_text(render(rows, thumbs_dir.name, csv_path), encoding="utf-8")
    return report_path


@click.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--workers", default=8, help="Parallel thumbnail extraction.")
def main(target, workers):
    """Build an HTML contact sheet from a culling run's CSV."""
    csv_path = find_csv(Path(target))
    report_path = generate_report(csv_path, workers)
    print(f"Report: {report_path}\nOpen it with: open '{report_path}'")


if __name__ == "__main__":
    main()
