"""Ollama vision analysis for photo culling.

Two things here matter more than the prompts.

**The canary.** A model without a vision encoder cannot look at the photograph, and
depending on the Ollama version it either rejects the request or answers from the prompt
alone. Either way the old code caught every exception and substituted neutral 0.5
scores, so a run that never saw a single pixel still produced a full, plausible-looking
set of results. On startup we now send a randomly generated test image and refuse to run
unless the model describes it correctly -- and analysis failures raise instead of
quietly becoming average scores.

**Two passes.** Judging quality and writing archive metadata are different jobs, and a
single prompt asking for both does neither well. Triage runs on everything; the tagging
pass runs only on frames worth keeping.
"""

from typing import Dict, List, Optional, Tuple
import base64
import io
import json
import logging
import random

import requests
from PIL import Image, ImageDraw

try:
    from .blur_detector import BlurDetector
    from .exposure import ExposureMeter
    from .models import ImageMetrics
except ImportError:
    from blur_detector import BlurDetector
    from exposure import ExposureMeter
    from models import ImageMetrics

DEFAULT_HOST = "http://localhost:11434"

# Ollama model families known to have a working vision encoder, best first. Matching is
# by prefix so any size/quant tag of a family counts.
KNOWN_VISION_FAMILIES = (
    "qwen3-vl",
    "qwen2.5vl",
    "gemma4",
    "gemma3",
    "minicpm-v",
    "llava",
    "llama3.2-vision",
    "moondream",
    "mistral-small3",
)

# Long edge sent to the model for the overview frame. 800px (the previous value) is not
# enough pixels to judge critical focus on.
OVERVIEW_EDGE = 1280
# Side length of the 100% centre crop taken from the full-resolution frame, so the model
# can see actual pixel-level detail rather than a downsample of it.
DETAIL_CROP = 768

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "subject_sharpness": {
            "type": "string",
            "enum": ["sharp", "acceptable", "soft", "unusable"],
        },
        "exposure": {
            "type": "string",
            "enum": ["good", "slightly_off", "badly_under", "badly_over"],
        },
        "framing": {"type": "string", "enum": ["strong", "fine", "weak", "broken"]},
        "technical_issues": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["keep", "review", "delete"]},
        "verdict_reason": {"type": "string"},
    },
    "required": [
        "subject",
        "subject_sharpness",
        "exposure",
        "framing",
        "technical_issues",
        "verdict",
        "verdict_reason",
    ],
}

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "keywords"],
}

CANARY_SCHEMA = {
    "type": "object",
    "properties": {
        "colour": {"type": "string"},
        "shape": {"type": "string"},
    },
    "required": ["colour", "shape"],
}

TRIAGE_PROMPT = """You are helping a photographer sort through a shoot.

You are given two views of ONE photograph: first the full frame, then a 100% centre crop
of the same photograph showing pixel-level detail.

Judge the photograph on its own terms. Shallow depth of field, deliberate motion blur,
film grain, fog, rain, night scenes and low-contrast or minimal compositions are
creative choices, not faults. Only the intended main subject needs to be sharp; a soft
background is normal and good.

subject_sharpness - how sharp is the MAIN SUBJECT only, ignoring the background:
  sharp       critically sharp, would hold up printed large
  acceptable  slightly soft but perfectly usable
  soft        noticeably soft; usable only small, or if the moment is special
  unusable    badly out of focus, or smeared by camera shake

exposure - is detail recoverable?
  good | slightly_off | badly_under | badly_over

framing - how the subject sits in the frame:
  strong   deliberate, well balanced
  fine     ordinary and correct, nothing wrong with it
  weak     awkward but salvageable by cropping
  broken   subject badly cut off, or the frame is an accident

technical_issues - short plain phrases for defects you can actually see, for example
"eyes closed", "camera shake", "blown highlights", "tilted horizon", "finger over lens",
"subject cut off". Return an empty list when there is nothing wrong.

verdict:
  keep    technically sound, no reason to reject it
  review  borderline, or you are not certain - a human should look
  delete  clearly unusable: badly out of focus, exposure beyond recovery, or an
          accidental frame (lens cap, floor, ceiling, pocket shot)

Be accurate rather than harsh. Only say delete when the photograph could not be used for
anything. If you are unsure, say review."""

TAG_PROMPT = """Write photo library metadata for this photograph.

description: one or two plain sentences saying what is actually in the photograph -
subject, setting, action, mood. Write it for someone searching an archive years from
now. Do not comment on sharpness, exposure or image quality.

keywords: 5 to 10 short lowercase search terms covering subject, objects, type of
location, activity, time of day or lighting, dominant colours, and mood. Single words or
short phrases, no punctuation, no quality judgements."""

CANARY_COLOURS = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (40, 70, 210),
    "yellow": (240, 210, 40),
    "purple": (140, 50, 180),
    "orange": (240, 130, 30),
}
CANARY_SHAPES = ("circle", "square", "triangle")


class VisionUnavailable(RuntimeError):
    """Raised when no usable vision model is reachable."""


class ModelCannotSee(RuntimeError):
    """Raised when the selected model returns text but demonstrably ignores images."""


def list_installed_models(host: str = DEFAULT_HOST, timeout: int = 5) -> List[str]:
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def detect_vision_model(host: str = DEFAULT_HOST) -> Optional[str]:
    """Pick the best installed model that plausibly has a vision encoder.

    Preference order follows KNOWN_VISION_FAMILIES. Nothing is ever pulled implicitly --
    downloading 20GB because a config default was stale is not a helpful surprise.
    """
    try:
        installed = list_installed_models(host)
    except Exception as e:
        raise VisionUnavailable(f"Cannot reach Ollama at {host}: {e}") from e

    for family in KNOWN_VISION_FAMILIES:
        matches = [m for m in installed if m.lower().startswith(family)]
        if matches:
            # Prefer an explicit 'instruct' build; thinking builds burn tokens
            # reasoning out loud before emitting the JSON we asked for.
            instruct = [m for m in matches if "instruct" in m.lower()]
            return sorted(instruct or matches)[0]

    return None


class OllamaVisionAnalyzer:
    """Vision-model analysis of photographs, with a startup proof that it can see."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = DEFAULT_HOST,
        timeout: int = 180,
        verify_vision: bool = True,
        use_detail_crop: bool = True,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.use_detail_crop = use_detail_crop
        self.logger = logging.getLogger(__name__)

        self.model = model or detect_vision_model(self.host)
        if not self.model:
            raise VisionUnavailable(
                "No vision-capable model found in Ollama.\n"
                "Install one, for example:\n"
                "  ollama pull qwen3-vl:8b-instruct      (6 GB, fast)\n"
                "  ollama pull qwen3-vl:30b-a3b-instruct (20 GB, sharper judgement)"
            )

        self._ensure_model_present()
        if verify_vision:
            self.verify_can_see()

        self.blur_detector = BlurDetector()
        self.exposure_meter = ExposureMeter()

    # ------------------------------------------------------------------ setup

    def _ensure_model_present(self):
        try:
            installed = list_installed_models(self.host)
        except Exception as e:
            raise VisionUnavailable(f"Cannot reach Ollama at {self.host}: {e}") from e

        if self.model not in installed:
            base = self.model.split(":")[0]
            near = [m for m in installed if m.split(":")[0] == base]
            hint = f" Installed builds of {base}: {', '.join(near)}." if near else ""
            raise VisionUnavailable(
                f"Model '{self.model}' is not installed.{hint}\n"
                f"Pull it with: ollama pull {self.model}"
            )

    def _make_canary(self, colour_name: str, shape: str) -> Image.Image:
        image = Image.new("RGB", (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        fill = CANARY_COLOURS[colour_name]

        if shape == "circle":
            draw.ellipse((110, 110, 400, 400), fill=fill)
        elif shape == "square":
            draw.rectangle((120, 120, 390, 390), fill=fill)
        else:
            draw.polygon([(256, 100), (400, 400), (112, 400)], fill=fill)

        return image

    def verify_can_see(self):
        """Prove the model actually receives images before trusting any of its output.

        A text-only model answers this from the prompt and gets the colour wrong. With
        six colours and three shapes a blind guess passes about 1 time in 18, and a
        wrong answer here is always fatal, so a single round is enough.
        """
        colour = random.choice(list(CANARY_COLOURS))
        shape = random.choice(CANARY_SHAPES)
        image = self._make_canary(colour, shape)

        prompt = (
            "This image contains exactly one solid shape on a white background. "
            "Report its colour and its shape. "
            "colour must be one of: red, green, blue, yellow, purple, orange. "
            "shape must be one of: circle, square, triangle."
        )

        try:
            raw = self._generate(prompt, [self._encode(image, 512)], CANARY_SCHEMA, timeout=90)
            answer = json.loads(raw)
        except Exception as e:
            raise ModelCannotSee(
                f"Vision check on '{self.model}' failed: {e}\n"
                "The model did not return a usable answer for a trivial test image."
            ) from e

        got_colour = str(answer.get("colour", "")).strip().lower()
        got_shape = str(answer.get("shape", "")).strip().lower()

        if colour not in got_colour or shape not in got_shape:
            raise ModelCannotSee(
                f"Model '{self.model}' cannot see images.\n"
                f"  Test image: a {colour} {shape} on white.\n"
                f"  Model said: a {got_colour or '?'} {got_shape or '?'}.\n"
                "Ollama accepts images for text-only models and silently ignores them, "
                "so this model would have invented every score.\n"
                "Choose a vision model, for example: ollama pull qwen3-vl:8b-instruct"
            )

        self.logger.info(f"Vision check passed: {self.model} correctly saw a {colour} {shape}")

    # ------------------------------------------------------------------ plumbing

    def _encode(self, image: Image.Image, max_edge: int) -> str:
        """JPEG-and-base64 a copy of the image. Never mutates the caller's image."""
        working = image.copy()
        if working.mode != "RGB":
            working = working.convert("RGB")
        if max_edge:
            working.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        working.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode()

    def _centre_crop(self, image: Image.Image, size: int = DETAIL_CROP) -> Optional[Image.Image]:
        """A 100% crop from the middle of the full-resolution frame."""
        width, height = image.size
        if width <= size or height <= size:
            return None

        left = (width - size) // 2
        top = (height - size) // 2
        return image.crop((left, top, left + size, top + size))

    def _generate(
        self,
        prompt: str,
        images: List[str],
        schema: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {"temperature": 0},
        }
        if schema:
            payload["format"] = schema

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=timeout or self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text[:300]}")

        return response.json().get("response", "").strip()

    def _generate_json(self, prompt: str, images: List[str], schema: Dict) -> Dict:
        raw = self._generate(prompt, images, schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Structured output should make this impossible, but a truncated response
            # can still arrive. Surfacing it beats silently substituting neutral scores.
            raise ValueError(f"Model returned unparseable JSON: {raw[:300]}") from e

    def _views(self, image: Image.Image) -> List[str]:
        views = [self._encode(image, OVERVIEW_EDGE)]
        if self.use_detail_crop:
            crop = self._centre_crop(image)
            if crop is not None:
                views.append(self._encode(crop, 0))
        return views

    # ------------------------------------------------------------------ analysis

    def triage(self, image: Image.Image) -> ImageMetrics:
        """Quality judgement for one photograph. Raises on failure -- never guesses."""
        triage_data = self._generate_json(TRIAGE_PROMPT, self._views(image), TRIAGE_SCHEMA)

        cv_stats = self.blur_detector.measure(image)
        exposure_stats = self.exposure_meter.measure(image)

        return ImageMetrics.from_triage(
            triage_data,
            cv_sharpness=cv_stats.get("sharpness_score"),
            exposure_stats=exposure_stats,
        )

    def describe(self, image: Image.Image) -> Tuple[str, List[str]]:
        """Archive description and keywords. Best-effort: metadata is not worth a crash."""
        try:
            data = self._generate_json(TAG_PROMPT, [self._encode(image, OVERVIEW_EDGE)], TAG_SCHEMA)
        except Exception as e:
            self.logger.warning(f"Tagging pass failed: {e}")
            return "", []

        keywords = [str(k).strip().lower() for k in data.get("keywords", []) if str(k).strip()]
        return str(data.get("description", "")).strip(), keywords[:10]

    def analyze(self, image: Image.Image, with_tags: bool = True) -> ImageMetrics:
        """Full analysis: triage, then tagging unless the frame is being rejected."""
        metrics = self.triage(image)

        if with_tags and metrics.verdict != "delete":
            description, keywords = self.describe(image)
            metrics.description = description
            metrics.keywords = keywords

        return metrics


def self_test(model: Optional[str] = None, host: str = DEFAULT_HOST) -> bool:
    """Check that Ollama is reachable and the chosen model can genuinely see."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        analyzer = OllamaVisionAnalyzer(model=model, host=host)
    except (VisionUnavailable, ModelCannotSee) as e:
        print(f"FAILED\n{e}")
        return False

    print(f"OK - {analyzer.model} is reachable and can see images.")
    return True


if __name__ == "__main__":
    import sys

    sys.exit(0 if self_test(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
