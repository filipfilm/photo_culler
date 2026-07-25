# Photo Culler

Offline photo culling with a local vision model. Point it at a shoot and it sorts every
frame into **Keep**, **Review** or **Delete**, writes a description and keywords into
your ON1 or XMP sidecars, and groups burst sequences so only the best frame of a run
stays in Keep.

Everything runs on your own machine through [Ollama](https://ollama.com). No cloud, no
API keys, no photographs leaving the computer.

## The one rule that shapes everything

**A photo wrongly sent to Review costs you a few seconds. A photo wrongly deleted is
gone.** So deleting requires two independent witnesses that agree: the vision model's
judgement *and* a measurement that corroborates it. Neither can reject a frame alone.

The practical consequence is that Review is often the largest pile. That is the tool
working, not failing. It is a triage assistant, not a decision maker.

## Install

```bash
pip install -r requirements.txt          # core
pip install -r requirements_photo.txt    # adds RAW support (NEF, CR2, ARW, DNG...)
```

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

Results also go to a timestamped CSV in `<folder>/cull_runs/`.

### Options

| Flag | Effect |
|---|---|
| `--fast` | Measurement only, no model. Sorts Keep/Review, never deletes. ~0.2 s/photo. |
| `--dry-run` | Analyse and report; write nothing. |
| `--model` | Pick an Ollama model. Default: auto-detect the best installed one. |
| `--no-tags` | Skip descriptions and keywords. Roughly twice as fast. |
| `--no-grouping` | Do not group bursts or demote near-duplicates. |
| `--override` | Replace existing keywords and descriptions. Ratings are always kept. |
| `--move-deletes` | Move confident deletions into `_culled_deletes/`. Moves, never erases. |
| `--cache-dir` | Reuse analysis for unchanged files across runs. |
| `--workers` | Parallel requests to Ollama (see [Speed](#speed)). |
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
| `qwen3-vl:4b-instruct` | 3.3 GB | 0/8 | 3/24 | 3.9 s |
| `qwen3-vl:8b-instruct` | 6.1 GB | 0/8 | 12/24 | 6.3 s |
| **`qwen3-vl:30b-a3b-instruct`** | 20 GB | **0/8** | **24/24** | **5.0 s** |
| `gemma4:31b` | 19 GB | 1/8 | 17/24 | 14.5 s |

**Recommendation for a 64 GB M4 Max: `qwen3-vl:30b-a3b-instruct`.** It was perfect on
both criteria and three times faster than `gemma4:31b`. It is a mixture-of-experts
model with roughly 3B parameters active per token, so it runs at about the speed of an
8B model while judging like a much larger one, and 20 GB of weights sits comfortably in
64 GB of unified memory.

Use `qwen3-vl:8b-instruct` if you want to get through a large folder faster and are
willing to work through a bigger Review pile. Do not use the 4B model for culling: it
approves of everything, so it never makes a mistake and never finds a problem either.

Two findings worth knowing:

- **Model size is not the story.** `gemma4:31b` is the same weight class as the winner
  but was slower, less accurate, and the only model to throw away a good photograph.
- **All of these models read exposure poorly** — they will call a three-stop
  underexposure "good". That is why exposure is measured from the histogram and the
  measurement overrides the model. Sharpness is the reverse: models judge it well and
  the measurement is the cross-check.

## Speed

Roughly 5–7 s per photo with tagging on, half that with `--no-tags`, and about 0.2 s in
`--fast` mode.

**Do not reach for `--workers` first.** It defaults to 1 for the model path on purpose.
Ollama gives every parallel slot its own context, so asking `qwen3-vl:30b-a3b-instruct`
for four of them took it from 20 GB of weights to 46 GB resident on a 64 GB machine,
which pushed the system into swap and left the GPU idling at under 1% while it waited on
the disk. One request already keeps Apple Silicon busy. If you do want to try it, give
Ollama matching permission and watch memory:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
python culler_universal.py ~/Photos/Shoot --workers 2
```

`--fast` mode is different — it loads no model and is plain CPU work, so it defaults to
8 workers and chews through RAW files at about 0.2 s each.

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
# Decision-logic invariants. No model needed, runs instantly.
python test_decision.py

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
config.py         Reads config.yaml
cli.py            Shared command line
culler_on1.py     Entry point, ON1 sidecars
culler_universal.py  Entry point, XMP sidecars
test_decision.py  Invariants for the two-witness rule
eval_harness.py   End-to-end check against generated ground truth
```

## Notes on your photo app

**ON1**: browse the folder in ON1 once before running the culler — it only updates `.on1`
files ON1 has already created, rather than inventing sidecars that could confuse the
catalogue. Restart ON1 afterwards to see the keywords.

**Everything else**: `.xmp` sidecars are written as `photo.NEF.xmp`. Existing ratings,
keywords and descriptions are preserved unless you pass `--override`.

Either way, search for `PhotoCuller:Review` to work through the uncertain frames, or
`PhotoCuller:Delete` to check the rejects before removing anything.

## Licence

MIT.
