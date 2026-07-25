# Photo Culler

Offline photo culling with a local vision model. Point it at a shoot and it sorts every
frame into **Keep**, **Review** or **Delete**, writes a description and keywords into
your ON1 or XMP sidecars, and groups burst sequences so only the best frame of a run
stays in Keep.

Everything runs on your own machine through [Ollama](https://ollama.com). No cloud, no
API keys, no photographs leaving the computer.

## It cannot delete your photographs

This tool has no code that moves, renames or removes a photo file, and
`test_no_destructive_operations.py` fails the build if any is ever added. The only
things it writes are sidecars, its own cache, and a CSV report.

**Delete is a keyword, not an action.** A rejected frame gets tagged
`PhotoCuller:Delete` and is otherwise left exactly where it was. Acting on that is your
job, in your own catalogue, where it can be undone.

## The rule that shapes the verdicts

**A photo wrongly sent to Review costs you a few seconds. A photo wrongly marked for
deletion wastes your attention and erodes your trust in the tool.** So the Delete verdict
requires two independent witnesses that agree: the vision model's judgement *and* a
measurement that corroborates it. Neither can reject a frame alone.

The practical consequence is that Review is often the largest pile. That is the tool
working, not failing. It is a triage assistant, not a decision maker.

## Install

```bash
pip install -r requirements.txt          # core, includes HEIC/HEIF for iPhone photos
pip install -r requirements_photo.txt    # adds RAW support (NEF, CR2, ARW, DNG...)
```

Handles RAW (NEF, CR2, CR3, ARW, DNG, RAF, RW2, ORF), JPEG, and HEIC/HEIF. Mixed
folders are fine — an iPhone shot and a Nikon RAW of the same moment group into the
same burst, since capture time and appearance are what matter, not file format.

Then install a vision model:

```bash
ollama pull qwen3-vl:8b-instruct         # 6 GB  - fast, good enough for triage
ollama pull qwen3-vl:30b-a3b-instruct    # 20 GB - noticeably better judgement
```

Check the setup before running a whole shoot:

```bash
python vision.py
```

This sends the model a generated test image and refuses to continue unless it describes
it correctly — see [Why the vision check exists](#why-the-vision-check-exists).

## Use

```bash
# ON1 Photo RAW workflow: updates the .on1 sidecars ON1 already created
python culler_on1.py ~/Photos/Shoot

# Everything else (Lightroom, Bridge, Capture One): writes .xmp sidecars
python culler_universal.py ~/Photos/Shoot

# See what it would do without touching a single file
python culler_universal.py ~/Photos/Shoot --dry-run --detail

# Fast triage of a huge folder: measurement only, no model, never deletes
python culler_universal.py ~/Photos/Shoot --fast
```

Results go to a timestamped CSV in `<folder>/cull_runs/`, plus an HTML contact sheet
next to it — every reject with a large thumbnail and its reasons, the Review pile split
into burst alternates versus genuinely flagged frames, and the Keeps as a grid for
spot-checking. Rebuild one any time with `python report.py <folder>`.

## The ON1 plugin

For the other half of the job — a handful of frames you want looked at *now*, with the
proposed metadata in front of you before any of it reaches the catalogue.

```bash
python3 on1_plugin/install.py
```

That builds `~/Applications/PhotoCuller.app`. In ON1 Photo RAW, select photographs in
Browse, right-click, **Send to › Send to Other Application…**, and pick it once; ON1
remembers it as **Send to › PhotoCuller** from then on. It also works by dragging photos
onto the app's Dock icon, or from Finder's **Open With**.

A window opens listing the selection, each frame analysed as it finishes. For every
photograph you get the verdict, the preview, and a tick box per field:

| Field | Default |
| --- | --- |
| Culling keywords (`PhotoCuller:Keep`, confidence, issues) | on |
| Descriptive keywords — editable | on |
| Description — editable | on |
| Full analysis block | on |
| Star rating | **off** |

Keywords and descriptions are editable text, ratings default to off for the same reason
they are opt-in on the command line, and nothing is written until you press **Write
metadata**. Untick a photograph and it is skipped entirely. You choose `.on1`, `.xmp` or
both; whichever the folder already uses is pre-selected.

**ON1 has no plugin API for this.** It hosts Photoshop-format pixel plugins and ships a
Lightroom `.lrplugin`, neither of which can be handed a selection of originals and asked
to write metadata. "Send to Other Application" is the one door, so the plugin is a small
application bundle — an AppleScript droplet that catches the open-documents event and
hands the paths to Python. Two consequences worth knowing:

- **Send To renders a copy first**, usually a TIFF beside the RAW. Metadata on that copy
  would be useless, so a render is traced back to its original by filename and the
  sidecar is written for the original. The window says `sent as DSC_0001.tif, writing to
  the original` when this happens.
- **macOS keeps unsigned applications out of `~/Documents`, `~/Desktop` and
  `~/Downloads`.** If the project lives in one of those, the installer also leaves a
  readable copy in `~/Library/Application Support/PhotoCuller/` and the bundle runs that
  — so re-run the installer after changing the code. Granting PhotoCuller.app Full Disk
  Access in System Settings › Privacy & Security makes it run your working tree directly
  instead. Photographs themselves are unaffected: they arrive as arguments from an
  application you chose, which is exactly the case macOS permits.

If Ollama is not running the window offers to fall back to measurement-only fast mode.
Anything that goes wrong before the window appears is logged to
`~/Library/Logs/PhotoCuller.log`.

For a whole shoot, use the command line instead — it caches, resumes, and writes a CSV
and a contact sheet.

## Long runs are safe to interrupt

Analysis is cached per photograph (default `~/.cache/photo_culler`), so a crash, a
Ctrl-C or a sleep-lid moment costs nothing: re-run the same command and it resumes where
it stopped, at cache speed. An interrupted run still writes its CSV for whatever
finished. Transient Ollama hiccups are retried once; if ten photographs fail in a row
the run stops and says so, rather than burning hours marking the rest of the folder
Failed because the backend died.

### Options

| Flag | Effect |
|---|---|
| `--fast` | Measurement only, no model. Sorts Keep/Review, never deletes. ~0.05 s/photo. |
| `--dry-run` | Analyse and report; write nothing. |
| `--model` | Pick an Ollama model. Default: auto-detect the best installed one. |
| `--no-tags` | Skip descriptions and keywords. Roughly twice as fast. |
| `--no-grouping` | Do not group bursts or demote near-duplicates. |
| `--override` | Replace existing keywords and descriptions. Ratings are always kept. |
| `--suggest-ratings` | Set stars on *unrated* photos. Off by default; never overwrites a rating you set. |
| `--cache-dir` | Where results are cached. Default `~/.cache/photo_culler`; `--no-cache` disables. |
| `--workers` | Parallel requests to Ollama (see [Speed](#speed)). |
| `--report/--no-report` | Build the HTML contact sheet next to the CSV (default on). |
| `--detail` | Print every photo as it is decided. |

Defaults live in `config.yaml` and every one of them is actually read; flags override it.

## What it judges

**Sharpness** — of the *main subject only*. Background blur is a creative choice and is
ignored. The model sees the full frame plus a 100% centre crop so it can assess focus at
the pixel level rather than guessing from a thumbnail.

**Exposure** — as a category (good / slightly off / badly under / badly over), backed by
an actual histogram measurement of clipped highlights and crushed shadows.

**Framing** — strong / fine / weak / broken, where "broken" means the subject is badly
cut off or the frame is an accident.

**Technical issues** — closed eyes, camera shake, tilted horizon, an obstruction, and so
on, in plain language.

**Bursts and duplicates** — frames taken close together in time *that also look alike*
are grouped; the strongest keeps its Keep, the rest drop to Review labelled as
alternates. Both conditions are required, so a fast handheld sequence of different
subjects is not merged, and a repeated setup hours apart is not either.

The model reports categories rather than numeric scores, because "is the subject sharp
or soft" is a question it answers reliably while "rate the sharpness 0.0–1.0" is not.
Scores still appear in the CSV and sidecars, derived from the categories, for sorting.

## Choosing a model

Measured on a Mac Studio M4 Max / 64 GB against a 32-image ground-truth set: 8 real
photographs, each also supplied heavily defocused, three stops under and three stops
over. These are the models' **raw verdicts**, before the decision layer's safety net, so
they show what each model actually perceives. "Caught" means a ruined frame was not
approved.

| Model | Size | False deletes | Caught | Speed |
|---|---|---|---|---|
| `qwen3-vl:4b-instruct` | 3.3 GB | 0/8 | 3/24 | 3.8 s |
| `qwen3-vl:8b-instruct` | 6.1 GB | 0/8 | 12/24 | 6.6 s |
| **`qwen3-vl:30b-a3b-instruct`** | 20 GB | **0/8** | **23/24** | **5.5 s** |
| `gemma4:31b` | 19 GB | 1/8 | 17/24 | 14.0 s |

Numbers are from a second run after the context-size fix; the first run agreed within
noise, except that the 30B model caught 24/24 rather than 23/24. Treat single-frame
differences as run-to-run variance, not signal.

**Recommendation for a 64 GB M4 Max: `qwen3-vl:30b-a3b-instruct`.** It never discarded a
good photograph, caught essentially everything, and ran two and a half times faster than
`gemma4:31b`. It is a mixture-of-experts model with roughly 3B parameters active per
token, so it runs at about the speed of an 8B model while judging like a much larger
one, and 20 GB of weights sits comfortably in 64 GB of unified memory.

Use `qwen3-vl:8b-instruct` if you want to get through a large folder faster and are
willing to work through a bigger Review pile. Do not use the 4B model for culling: it
approves of everything, so it never makes a mistake and never finds a problem either.

Run through the full pipeline rather than as raw verdicts, both `qwen3-vl:30b-a3b` and
`gemma4:31b` pass: no original reaches Delete and no ruined frame stays in Keep. The
photograph `gemma4:31b` wanted to throw away came back as Keep, because no measurement
corroborated it — which is the two-witness rule doing the exact job it exists for. Run
`eval_harness.py` to reproduce this on your own photographs.

Two findings worth knowing:

- **Model size is not the story.** `gemma4:31b` is the same weight class as the winner
  but was slower, less accurate, and the only model to throw away a good photograph.
- **All of these models read exposure poorly** — they will call a three-stop
  underexposure "good". That is why exposure is measured from the histogram and the
  measurement overrides the model. Sharpness is the reverse: models judge it well and
  the measurement is the cross-check.

## Speed

Measured on the M4 Max: about 5.5 s per photo for triage with
`qwen3-vl:30b-a3b-instruct`, 6.6 s with the 8B, 14 s with `gemma4:31b`. Tagging is a
second pass over frames worth keeping, so a full run with descriptions and keywords
lands near twice the triage figure; `--no-tags` skips it. `--fast` mode is around
0.05 s per photo.

**Do not reach for `--workers` first.** It defaults to 1 for the model path on purpose:
one request already keeps Apple Silicon busy, and every extra Ollama slot wants its own
context. If you do try it, give Ollama matching permission and watch memory:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
python culler_universal.py ~/Photos/Shoot --workers 2
```

The tool asks Ollama for an 8192-token context rather than letting it size from the
model's 262,144-token maximum. That is a memory setting, not a speed one — per-photo
time is the same either way — but it holds `qwen3-vl:30b-a3b-instruct` at 19.8 GB
instead of 46 GB. At 46 GB a 64 GB machine has nothing spare, and adding workers tipped
it into swap, where inference ran at disk speed with the GPU idling under 1%.

`--fast` mode loads no model at all — it is plain CPU work, defaults to 8 workers, and
got through 24 Nikon RAW files in 1.1 seconds.

Use `--cache-dir ~/.cache/photo_culler` so re-running a folder only analyses what
changed.

## Why the vision check exists

Ask Ollama to analyse an image with a model that has no vision encoder and, depending on
the version, it either rejects the request or quietly ignores the image and answers from
the prompt alone. The previous version of this tool caught every error and substituted
neutral 0.5 scores, so a run that never looked at a single pixel still produced a
complete, plausible-looking set of results.

Now the tool sends a randomly generated shape on startup and refuses to run unless the
model reports the right colour and shape, and analysis failures raise instead of
becoming average scores. `--skip-vision-check` exists but you should not use it.

## Why "blurry" is not measured the obvious way

The natural approach — average edge energy across the frame — throws away exactly the
photographs you care about most. Shallow depth of field, fog, snow, minimal
compositions and soft film scans are mostly smooth by intent, so a global sharpness
average reads "blurry" while the subject is critically sharp. In the previous version
this measurement could override the model, and sharp low-contrast frames were being sent
to Delete with high confidence.

So the frame is tiled and the *sharpest* tile decides: the question is "is any part of
this critically sharp?", not "is it sharp on average". A crisp eye in a wash of bokeh
scores high, as it should. On the ground-truth set that separates originals (0.78–1.00)
from defocused copies (0.00) with nothing in between.

The signal is still only trusted in one direction. A high score is real evidence of
detail; a low score might just be fog. So it can veto a deletion but never cause one.

## Checking it still works

Two levels, both worth running after changing anything:

```bash
# Proof that no code path can move, rename or remove a photograph.
python test_no_destructive_operations.py

# Decision-logic invariants. No model needed, runs instantly.
python test_decision.py

# Which formats are accepted, and that the two extension lists agree.
python test_formats.py

# The plugin: which file a Send To actually lands on, and that an unticked
# field never reaches the sidecar.
python -m pytest test_on1_plugin.py

# End to end against your own photographs. Builds a ground-truth set by taking
# real frames and also defocusing / under- / overexposing them, then checks that
# no original is ever sent to Delete.
python eval_harness.py ~/Photos/some-folder
python eval_harness.py ~/Photos/some-folder --model qwen3-vl:8b-instruct
```

The harness generates its images in a temporary directory and never writes to the
folder you point it at.

## Files

```
vision.py         Ollama client: model detection, vision check, prompts, JSON schemas
blur_detector.py  Tile-based sharpness measurement
exposure.py       Histogram clipping measurement
decision.py       Keep/Review/Delete, and the two-witness rule
grouping.py       Burst and near-duplicate grouping
batch.py          Folder orchestration, caching, threading
extractor.py      RAW and standard image loading, capture times
sidecars.py       ON1 .on1 and standard .xmp writers
report.py         HTML contact-sheet report for reviewing a run
config.py         Reads config.yaml
cli.py            Shared command line
culler_on1.py     Entry point, ON1 sidecars
culler_universal.py  Entry point, XMP sidecars

on1_plugin/install.py  Builds PhotoCuller.app, the bundle ON1 sends photos to
on1_plugin/app.py      Plugin entry point: analyse a selection, show the popup
on1_plugin/review.py   The metadata popup
on1_plugin/resolve.py  Traces ON1's rendered copies back to the originals

test_decision.py  Invariants for the two-witness rule
test_formats.py   Accepted formats, and the two extension lists agreeing
test_no_destructive_operations.py  Proof no code path can touch a photo file
test_on1_plugin.py  Which file gets culled, and which metadata reaches disk
eval_harness.py   End-to-end check against generated ground truth
```

## What gets written where

Three things land on every photo, in both sidecar formats:

**Keywords** — the searchable part. `PhotoCuller:Keep`, `CullerConfidence:0.90`,
`CullerIssues:soft focus, motion blur`, `CullerSuggestedRating:5`, plus
`CullerBurst:best-of-6` and `CullerDuplicate:<file>` where they apply, alongside the AI
keywords and everything that was already there.

**Description** — the plain sentence, in `metadata.Description` (ON1) or
`dc:description` (XMP).

**The full analysis** — verdict, confidence, sharpness/exposure/framing categories, all
four scores, the measured sharpness and the model's reasoning. In ON1 that is a JSON
block at `metadata.PhotoCullerAnalysis`. XMP has no equivalent free-form slot, so it
goes two places: a `photoculler:` namespace holding each field separately, and a
readable one-line summary in `photoshop:Instructions`, which is a field Lightroom,
Bridge and ON1 actually display.

**Star ratings are left alone** unless you pass `--suggest-ratings`, and even then only
empty ones are filled. The culler's opinion is always available as the
`CullerSuggestedRating` keyword regardless.

Everything above, plus `subject`, `burst_size`, `best_of_burst` and `duplicate_of`, is
also in the run's CSV — 21 columns, the complete record.

## Notes on your photo app

**ON1**: browse the folder in ON1 once before running the culler — it only updates `.on1`
files ON1 has already created, rather than inventing sidecars that could confuse the
catalogue. Restart ON1 afterwards to see the keywords.

**Everything else**: `.xmp` sidecars are written as `photo.NEF.xmp`. Existing ratings,
keywords and descriptions are preserved unless you pass `--override`.

Either way, search for `PhotoCuller:Review` to work through the uncertain frames, or
`PhotoCuller:Delete` to see the rejects. Those files are still sitting exactly where
they were — removing any of them is a decision you make in your own catalogue.

## Licence

MIT.
