# Ollama setup

The culler talks to a vision model running locally through [Ollama](https://ollama.com).
Nothing leaves your machine.

## 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Windows and a macOS app are available from https://ollama.com. The macOS app starts the
server automatically; otherwise run `ollama serve`.

## 2. Pull a vision model

The model **must** have a vision encoder. A text-only model cannot see the photograph
and will invent its answers.

```bash
ollama pull qwen3-vl:30b-a3b-instruct   # 20 GB - best tested, needs ~32 GB RAM
ollama pull qwen3-vl:8b-instruct        #  6 GB - faster, larger Review pile
```

Leave `model.name` blank in `config.yaml` and the culler picks the best installed model
on its own.

## 3. Verify

```bash
python vision.py
```

This sends the model a randomly generated coloured shape and checks it comes back
correctly described:

```
INFO - Vision check passed: qwen3-vl:30b-a3b-instruct correctly saw a red triangle
OK - qwen3-vl:30b-a3b-instruct is reachable and can see images.
```

The same check runs at the start of every culling run. If it fails, the run stops rather
than producing scores for photographs nothing looked at.

To test a specific model:

```bash
python vision.py qwen3-vl:8b-instruct
```

## Which model

Measured on a Mac Studio M4 Max / 64 GB against 32 images with known ground truth. See
the README for the full table.

| Model | Size | False deletes | Caught | Speed |
|---|---|---|---|---|
| `qwen3-vl:4b-instruct` | 3.3 GB | 0/8 | 3/24 | 3.9 s |
| `qwen3-vl:8b-instruct` | 6.1 GB | 0/8 | 12/24 | 6.3 s |
| **`qwen3-vl:30b-a3b-instruct`** | 20 GB | **0/8** | **24/24** | **5.0 s** |
| `gemma4:31b` | 19 GB | 1/8 | 17/24 | 14.5 s |

`qwen3-vl:30b-a3b-instruct` is a mixture-of-experts model: 30B parameters stored, about
3B active per token. That is why it beats a dense 31B model on quality *and* runs three
times faster.

Rough memory guidance: allow the model's file size plus a few GB. A 20 GB model is
comfortable on 32 GB and easy on 64 GB.

## Going faster

One Ollama server handles concurrent requests; the multiple-servers-on-different-ports
approach an earlier version of this project recommended was never necessary and is not
supported by the `ollama serve` flags it suggested.

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
python culler_universal.py ~/Photos/Shoot --workers 4
```

Other levers:

```bash
--no-tags                        # skip descriptions/keywords, roughly twice as fast
--cache-dir ~/.cache/photo_culler  # re-runs only analyse what changed
--fast                           # no model at all, ~0.2 s/photo, never deletes
```

## Troubleshooting

**"Cannot reach Ollama"** — start it with `ollama serve`, then check
`curl http://localhost:11434/api/tags`.

**"No vision-capable model found"** — pull one of the models above. The culler will not
download a 20 GB model behind your back.

**"Model X cannot see images"** — the model has no vision encoder. The error names what
the test image was and what the model claimed to see. Pick a `qwen3-vl` or `gemma4`
build.

**"Multimodal data provided, but model does not support multimodal requests"** — same
cause, reported by Ollama itself. Some Ollama versions raise this; others quietly ignore
the image, which is exactly why the startup check exists.

**Everything lands in Review** — expected with a small model. Deleting needs two
independent witnesses, and a weak model rarely provides a confident first one. Move up
to `qwen3-vl:30b-a3b-instruct`.

## Custom host

```python
from vision import OllamaVisionAnalyzer

analyzer = OllamaVisionAnalyzer(
    model="qwen3-vl:30b-a3b-instruct",
    host="http://192.168.1.100:11434",
)
```

Or set `model.host` in `config.yaml`, or pass `--host`.

Prompts and JSON schemas live at the top of `vision.py`.
