"""Working out which photograph ON1 actually meant.

ON1 has no plugin API that can hand a script the selected originals, so the plugin is
reached through "Send to Other Application", drag-to-Dock and Finder's Open With. Only
the last two give us the original file. Send To renders a copy first -- a TIFF or PSD
written next to the RAW with the same stem -- and hands us that.

Writing the culling metadata onto the render would be useless: it is a derived file the
catalogue treats as a separate photo, and the NEF the photographer is actually culling
would stay untouched. So a render is traced back to its original by stem, and the
sidecar is written for the original.

The mapping is deliberately conservative. A JPEG is only treated as a render when a RAW
sibling exists, because a JPEG on its own is a perfectly good original -- half this
tool's users shoot straight to JPEG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import re

try:
    from ..extractor import HEIF_EXTENSIONS, RAW_EXTENSIONS
except ImportError:
    from extractor import HEIF_EXTENSIONS, RAW_EXTENSIONS

# Formats ON1 can produce from a Send To. PSB is the large-document Photoshop format;
# ON1 offers it for files past Photoshop's 30,000 pixel limit.
RENDER_EXTENSIONS = {".tif", ".tiff", ".psd", ".psb", ".png"}

# Ambiguous: an original when it stands alone, a render when a RAW sits beside it.
AMBIGUOUS_EXTENSIONS = {".jpg", ".jpeg"}

ORIGINAL_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | AMBIGUOUS_EXTENSIONS

SIDECAR_EXTENSIONS = {".on1", ".xmp", ".aae"}

# What ON1 and friends bolt onto a stem when the plain name is taken. "IMG_1234-2.tif",
# "IMG_1234 copy.tif", "IMG_1234-Edit.tif" all trace back to IMG_1234.
_DERIVED_STEM_SUFFIX = re.compile(
    r"(?:[-_ ](?:copy|edit|edited|enhanced|final)|[-_ ]\d{1,3})+$", re.IGNORECASE
)


@dataclass
class ResolvedPhoto:
    """One photograph to analyse, and the file we were handed to reach it."""

    photo: Path  # what gets analysed, and whose sidecar gets written
    received: Path  # what ON1 or Finder actually passed us

    @property
    def was_redirected(self) -> bool:
        return self.photo != self.received

    @property
    def note(self) -> str:
        if not self.was_redirected:
            return ""
        return f"sent as {self.received.name}, writing to the original"


def _candidate_stems(stem: str) -> List[str]:
    """The stem itself, then the same stem with a derived-copy suffix peeled off."""
    stems = [stem]
    stripped = _DERIVED_STEM_SUFFIX.sub("", stem)
    if stripped and stripped != stem:
        stems.append(stripped)
    return stems


def _index_folder(folder: Path, cache: Dict[Path, Dict]) -> Dict:
    """Map lowercased stem -> the photographs in this folder, by kind.

    Built by listing the folder rather than by testing constructed filenames, because
    the Mac's filesystem is case-insensitive: "A.nef" reports itself as existing next to
    a file actually named "A.NEF", and the wrongly-cased path that comes back then fails
    to match the real one when duplicate selections are collapsed.
    """
    if folder in cache:
        return cache[folder]

    originals: Dict[str, Path] = {}
    renders: Dict[str, Path] = {}
    try:
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            stem = path.stem.lower()
            if suffix in RAW_EXTENSIONS or suffix in HEIF_EXTENSIONS:
                originals.setdefault(stem, path)
            elif suffix in AMBIGUOUS_EXTENSIONS or suffix in RENDER_EXTENSIONS:
                renders.setdefault(stem, path)
    except OSError:
        pass

    cache[folder] = {"originals": originals, "renders": renders}
    return cache[folder]


def find_original(render: Path, cache: Optional[Dict[Path, Dict]] = None) -> Optional[Path]:
    """The RAW or HEIF a rendered copy came from, if it is sitting in the same folder."""
    index = _index_folder(render.parent, cache if cache is not None else {})
    for stem in _candidate_stems(render.stem):
        match = index["originals"].get(stem.lower())
        if match is not None and match != render:
            return match
    return None


def _resolve_one(path: Path, cache: Dict[Path, Dict]) -> Optional[ResolvedPhoto]:
    suffix = path.suffix.lower()

    if suffix in SIDECAR_EXTENSIONS:
        # Somebody selected the sidecar instead of the photo. Follow it home; a JPEG
        # will do when there is no RAW, since ON1 writes .on1 files for those too.
        index = _index_folder(path.parent, cache)
        original = find_original(path, cache) or index["renders"].get(path.stem.lower())
        return ResolvedPhoto(photo=original, received=path) if original else None

    if suffix in RENDER_EXTENSIONS or suffix in AMBIGUOUS_EXTENSIONS:
        original = find_original(path, cache)
        if original:
            return ResolvedPhoto(photo=original, received=path)

    if suffix in ORIGINAL_EXTENSIONS or suffix in RENDER_EXTENSIONS:
        return ResolvedPhoto(photo=path, received=path)

    return None


def resolve_paths(
    paths: Sequence[str], extensions: Optional[Sequence[str]] = None
) -> List[ResolvedPhoto]:
    """Turn whatever ON1 or Finder handed us into a list of photographs to analyse.

    Folders are expanded, renders are traced back to their originals, and two renders of
    the same RAW collapse into one entry -- otherwise selecting a photo and its exported
    TIFF would analyse the same NEF twice.
    """
    allowed = None
    if extensions:
        allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    found: Dict[str, ResolvedPhoto] = {}
    folder_cache: Dict[Path, Dict] = {}

    def consider(path: Path):
        if path.name.startswith("."):
            return
        resolved = _resolve_one(path, folder_cache)
        if resolved is None:
            return
        if allowed and resolved.photo.suffix.lower() not in allowed:
            return
        # Lowercased, because the filesystem treats Shoot/A.NEF and shoot/a.nef as one
        # file and analysing it twice would be the visible symptom.
        key = str(resolved.photo.resolve()).lower()
        # An entry we reached directly beats one we inferred from a render, so the UI
        # reports "sent as ..." only when that is the only way we got there.
        if key not in found or (found[key].was_redirected and not resolved.was_redirected):
            found[key] = resolved

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    consider(child)
        elif path.is_file():
            consider(path)

    return sorted(found.values(), key=lambda r: str(r.photo))
