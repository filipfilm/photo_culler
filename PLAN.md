# Photo Culler — Diagnosis & Rework Plan

Plan for a coding agent to fix this project. Read the whole Diagnosis section before
touching code — the fixes only make sense in light of what was actually broken.

## Context

Offline photo culler: extracts thumbnails (JPEG + RAW via rawpy), scores them with an
Ollama vision model plus OpenCV blur heuristics, decides Keep/Review/Delete, and writes
metadata (ON1 `.on1` sidecars via `culler_on1.py`, universal `.xmp` via
`culler_universal.py`, CSV via `batch.py`). Owner reports it "didn't work well".

Verified environment (July 2026, this machine):

- Ollama running at `localhost:11434`. Installed models: `gemma4:31b`, `llama3`,
  `llama3.2:3b`, `qwen2.5:3b`, `qwen3:30b`, `qwen3-coder-30b`.
- `gemma4:31b` **does** accept images and describes them accurately (verified with a real
  photo), even though its `/api/tags` capabilities list omits "vision". ~5.5 s/image.
- The configured default model `gemma4:e4b` is **not installed** — first run tries to
  pull it and may pull nothing usable.
- Python 3.14 system install has none of the deps; use a venv.

## Diagnosis (verified by running the code)

### Bug 1 — CV blur score can veto the vision model and delete good photos (critical)

`blur_detector.py::combine_vision_and_cv` takes `min(vision, cv)` whenever the two
disagree. The CV score is a mean of four global edge metrics with arbitrary
normalization constants (`laplacian/1000`, `sobel/10000`, `brenner/1e6`,
`edge_density*10`). Global edge metrics punish any photo with large smooth areas:
shallow-DOF portraits (bokeh), fog, sky, minimalist compositions, soft film scans.

Measured on real photos (vision score fixed at 0.9 = "sharp"):

| photo | cv score (800px) | combined | resulting decision |
|---|---|---|---|
| sharp textured wall | 1.00 | 0.93 | Keep |
| same photo, gaussian-blurred | 0.46 | 0.46 | Delete (correct, by luck) |
| sharp low-contrast film photo (beach) | 0.35 | **0.35** | **Delete, conf 0.8** |
| ordinary iPhone photo | 0.45 | **0.45** | **Delete, conf 0.7** |

So sharp keepers are deleted no matter what the vision model says. This is the primary
"didn't work well".

### Bug 2 — CV runs on an already-mutated 800px image

`ollama_vision.py::_image_to_base64` calls `image.thumbnail((800, 800))` which mutates
the caller's image **in place**. `analyze_batch` then runs `detect_cv_blur(image)` on
that shrunken copy, so the CV constants (tuned for who-knows-what resolution) see a
different image than intended. Also 800px is too small for the vision model to judge
critical focus at all.

### Bug 3 — Delete thresholds are far too aggressive

`batch.py::_make_decision`: `blur < 0.5 → Delete`. Half the score range is a delete.
Vision models cluster scores around 0.4–0.7, so ordinary photos constantly land in
Delete. For a culler the costly error is a false delete; the logic must be conservative.

### Bug 4 — Prompt is blur-obsessed and bias-loaded

One mega-prompt tells the model "Be very strict… when in doubt, score lower", then asks
for keywords/description as an afterthought (`"keywords": ["subject"]`). Results:
skewed-low blur scores (amplifying Bug 3), and near-useless tags/descriptions — which
was half the point of the tool.

### Bug 5 — Absolute 0–1 scores from an LLM are not calibrated

LLMs emit coarse steps (0.3/0.5/0.7/0.8); thresholding them at 0.35/0.5 is noise.
Categorical judgments (sharp / acceptable / soft / unusable) are far more reliable.

### Rot / dead weight

- `config.yaml` is loaded by **nothing** (`config_loader.py` was deleted); thresholds
  are hardcoded in `batch.py`.
- README documents features that don't exist or are wrong: `ollama serve --port` is not
  a real flag; multi-port "concurrent" advice should be `OLLAMA_NUM_PARALLEL`; the
  benchmark/accuracy tables are invented.
- ~20 deleted modules sit uncommitted in git status; `my_results.json` and a 1.2 MB CSV
  (with confidences of 1.0 that current code cannot produce — stale runs) are committed.
- `culler_on1.py` and `culler_universal.py` duplicate ~90% of their logic.

## The Plan

### Phase 0 — housekeeping (do first)

1. Commit the current deletion state so there's a clean baseline.
2. Delete `my_results.json`, `photo_culler_results.csv`, `.DS_Store`; add to
   `.gitignore` (`*.csv`, `.DS_Store`, `__pycache__/`, `venv/`).
3. Merge `requirements_photo.txt` into `requirements.txt` with optional extras noted.

### Phase 1 — make analysis truthful

1. **Model selection & startup validation** (`ollama_vision.py`):
   - Default model: auto-detect. Query `/api/tags`; prefer, in order, any installed
     model from a known-good vision list (`gemma4:*`, `qwen2.5vl:*`, `gemma3:*`,
     `llava:*`, `moondream:*`); else instruct the user to `ollama pull` one. Never
     silently pull 19 GB.
   - **Canary test at startup**: send a generated image with known content (e.g. solid
     red square with a white circle) and ask what it shows. If the answer is wrong, the
     model can't see images — abort with a clear error instead of fabricating scores.
     (This is exactly the failure mode that silently produced garbage before: a
     text-only model happily returns invented JSON.)
2. **Fix the image pipeline**:
   - `_image_to_base64` must operate on a copy; never mutate the input.
   - Send the vision model a ~1280px long-edge JPEG (quality 90), not 800px.
   - Additionally send a centered 100% crop (e.g. 768px from the full-res image) in the
     same request (`images: [overview, crop]`) so the model can judge critical focus.
3. **Demote CV blur to advisory** (`blur_detector.py`, `ollama_vision.py`):
   - Delete `combine_vision_and_cv`'s `min()` veto. The vision verdict is primary.
   - CV disagreement (vision=sharp, cv=very low or vice versa) → force decision to
     **Review**, never Delete. CV alone must never cause a delete.
   - Compute CV metrics on a fixed-size input (resize long edge to 1536 first) so the
     normalization constants mean something consistent.

### Phase 2 — two-stage prompting with structured output

1. **Stage A — triage prompt** (every photo). Neutral wording, no "be strict" bias.
   Ask for categories, not floats:
   ```json
   {
     "subject": "one line: what is the main subject",
     "subject_sharpness": "sharp | acceptable | soft | unusable",
     "exposure": "good | slightly_off | badly_under | badly_over",
     "technical_issues": ["closed eyes", "motion blur", "..."],
     "verdict": "keep | review | delete",
     "verdict_reason": "one line"
   }
   ```
   Use Ollama structured outputs: pass a JSON schema in `format` (supported since
   Ollama 0.5) instead of regex-scraping `{...}` from text.
2. **Stage B — tagging prompt** (Keep/Review only, skip deletes to save time): 1–2
   sentence description written for photo metadata, 5–10 lowercase keywords, no
   quality commentary. This is what lands in `.on1` / `.xmp` sidecars.
3. **Decision logic** (`batch.py::_make_decision`), conservative by construction:
   - Delete only when: vision verdict is `delete` AND `subject_sharpness` is
     `unusable` (or a hard technical issue) AND CV agrees (cv score below its 20th
     percentile for the session). Otherwise Review.
   - Keep when verdict `keep` and sharpness `sharp|acceptable`.
   - Everything else Review. It is fine for Review to be large; it must never be wrong
     in the Delete pile.
   - Map categories to the existing 0–1 fields (sharp=0.9, acceptable=0.7, soft=0.4,
     unusable=0.1) so `models.py`, cache format, CSV and sidecar writers keep working.

### Phase 3 — culling is relative, not absolute

1. **Burst/series grouping**: group by EXIF capture time (≤2 s gaps) + perceptual hash
   distance (add `imagehash` dep). Within a group, rank by sharpness/verdict and mark
   only the best as Keep, rest as Review-duplicate. This is where real culling value is.
2. **Exact/near-duplicate detection** across the folder via phash; tag duplicates in
   metadata (`PhotoCuller:Duplicate`).
3. Optional (flagged): face/eye sharpness check with OpenCV cascades on detected face
   region for portrait shoots.

### Phase 4 — consolidation & honesty

1. Extract shared CLI logic from `culler_on1.py` / `culler_universal.py` into one
   module; the two entry points keep only their sidecar-writing differences.
2. Actually load `config.yaml` (thresholds, model, extensions) with CLI flags
   overriding; delete dead keys.
3. Concurrency: drop the multi-port advice; use one Ollama with
   `OLLAMA_NUM_PARALLEL=4` and a worker pool (the code's ThreadPoolExecutor is fine).
4. Rewrite README to match reality; delete invented benchmarks; document the canary
   check and the conservative-delete philosophy.
5. Per-run CSV (timestamped name) instead of appending to one global file forever.

### Validation (do not skip)

- Keep a tiny eval harness: `testphotos/` with a sharp photo, a gaussian-blurred copy
  (`GaussianBlur(12)`), a low-contrast-but-sharp photo, and an out-of-focus real shot.
  Assert: blurred → delete/review, sharp low-contrast → **never** delete.
- Run the full pipeline on a real folder (e.g. 30 mixed photos) before/after and eyeball
  the Delete pile — the acceptance test is "zero keepers in Delete".

## Priorities if time-boxed

Bug 1 + Bug 3 + the canary check (Phase 1) fix "deletes good photos" and "fabricates
results" — that alone makes the tool usable. Phase 2 makes tags/descriptions worth
writing to sidecars. Phase 3 is the biggest feature win. Phase 4 is cleanup.
