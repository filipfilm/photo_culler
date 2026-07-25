"""Loading pixels out of RAW and standard image files."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import io
import logging

from PIL import Image

try:
    import rawpy

    RAWPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    RAWPY_AVAILABLE = False

RAW_EXTENSIONS = {
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

# EXIF tag ids, avoiding a PIL.ExifTags lookup per file.
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


class RawThumbnailExtractor:
    """Extracts a usable image from RAW and standard formats."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.raw_extensions = RAW_EXTENSIONS

    # ------------------------------------------------------------------ public

    def extract(self, filepath: Path) -> Optional[Image.Image]:
        image, _ = self.extract_with_info(filepath)
        return image

    def extract_thumbnail(self, filepath: Path) -> Optional[Image.Image]:
        """Backwards-compatible alias used by older callers."""
        return self.extract(filepath)

    def extract_with_info(self, filepath: Path) -> Tuple[Optional[Image.Image], Dict]:
        """Return the image plus what we learned while decoding it.

        Capture time comes back here rather than from a second pass because decoding a
        RAW file twice to read one timestamp is the most expensive way to get it.
        """
        try:
            if filepath.suffix.lower() in self.raw_extensions:
                if not RAWPY_AVAILABLE:
                    self.logger.warning(
                        f"rawpy not installed, skipping RAW file {filepath.name}"
                    )
                    return None, {}
                return self._extract_raw(filepath)

            with Image.open(filepath) as image:
                info = {"capture_time": self._capture_time_from_exif(image, filepath)}
                return image.copy(), info

        except Exception as e:
            self.logger.error(f"Failed to extract image from {filepath.name}: {e}")
            return None, {}

    # ------------------------------------------------------------------ internals

    def _extract_raw(self, filepath: Path) -> Tuple[Optional[Image.Image], Dict]:
        try:
            with rawpy.imread(str(filepath)) as raw:
                # The embedded JPEG preview is both the fastest path to pixels and the
                # place the camera's EXIF survives, so try it first.
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        with Image.open(io.BytesIO(thumb.data)) as image:
                            info = {
                                "capture_time": self._capture_time_from_exif(image, filepath)
                            }
                            return image.copy(), info
                except Exception:
                    pass

                # No usable preview: demosaic at half size, which is still far more
                # resolution than any of the analysis needs.
                rgb = raw.postprocess(
                    use_camera_wb=True, half_size=True, no_auto_bright=False, output_bps=8
                )
                return Image.fromarray(rgb), {"capture_time": self._mtime(filepath)}

        except Exception as e:
            self.logger.error(f"Failed to process RAW file {filepath.name}: {e}")
            return None, {}

    def _capture_time_from_exif(
        self, image: Image.Image, filepath: Path
    ) -> Optional[datetime]:
        try:
            exif = image.getexif()
            if exif:
                raw_value = exif.get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
                if raw_value:
                    return datetime.strptime(str(raw_value), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
        return self._mtime(filepath)

    @staticmethod
    def _mtime(filepath: Path) -> Optional[datetime]:
        try:
            return datetime.fromtimestamp(filepath.stat().st_mtime)
        except OSError:
            return None
