"""Loading config.yaml.

The previous config.yaml was decorative -- nothing read it, while the values that
actually governed behaviour were hardcoded. Anything documented here is now genuinely
wired up, and a missing or partial file is fine: defaults fill the gaps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "config.yaml"

DEFAULT_EXTENSIONS = [
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf", ".jpg", ".jpeg",
]


@dataclass
class Config:
    model_name: Optional[str] = None
    host: str = "http://localhost:11434"
    timeout_seconds: int = 300
    verify_vision: bool = True
    context_tokens: int = 8192

    workers: int = 1
    fast_workers: int = 8
    tagging: bool = True
    recursive: bool = True

    grouping_enabled: bool = True
    burst_gap_seconds: float = 3.0

    sharp_evidence_vetoes_delete: bool = True

    csv_dir: str = "cull_runs"
    # Caching is on by default: at five-plus seconds per photograph, losing a
    # half-finished overnight run to a crash or a Ctrl-C is the expensive accident.
    cache_dir: str = "~/.cache/photo_culler"
    extensions: List[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))

    def resolved_cache_dir(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return Path(self.cache_dir).expanduser()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Read config.yaml, falling back to defaults for anything absent."""
        config = cls()

        if path is None:
            path = Path(__file__).parent / DEFAULT_CONFIG_NAME
        if not path.exists():
            return config

        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed - ignoring %s and using defaults", path)
            return config

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not parse %s (%s) - using defaults", path, e)
            return config

        model = data.get("model") or {}
        config.model_name = model.get("name") or None
        config.host = model.get("host", config.host)
        config.timeout_seconds = int(model.get("timeout_seconds", config.timeout_seconds))
        config.verify_vision = bool(model.get("verify_vision", config.verify_vision))
        config.context_tokens = max(2048, int(model.get("context_tokens", config.context_tokens)))

        processing = data.get("processing") or {}
        config.workers = max(1, int(processing.get("workers", config.workers)))
        config.fast_workers = max(1, int(processing.get("fast_workers", config.fast_workers)))
        config.tagging = bool(processing.get("tagging", config.tagging))
        config.recursive = bool(processing.get("recursive", config.recursive))

        grouping = data.get("grouping") or {}
        config.grouping_enabled = bool(grouping.get("enabled", config.grouping_enabled))
        config.burst_gap_seconds = float(
            grouping.get("burst_gap_seconds", config.burst_gap_seconds)
        )

        decisions = data.get("decisions") or {}
        config.sharp_evidence_vetoes_delete = bool(
            decisions.get("sharp_evidence_vetoes_delete", config.sharp_evidence_vetoes_delete)
        )

        output = data.get("output") or {}
        config.csv_dir = output.get("csv_dir", config.csv_dir)
        cache = data.get("cache") or {}
        if "dir" in cache:
            config.cache_dir = cache["dir"] or ""
        extensions = output.get("extensions")
        if extensions:
            config.extensions = [
                e if str(e).startswith(".") else f".{e}" for e in extensions
            ]

        return config

    def normalized_extensions(self, override: Optional[str] = None) -> List[str]:
        """Extensions from a comma-separated CLI override, or the configured list."""
        if not override:
            return list(self.extensions)
        return [
            e if e.startswith(".") else f".{e}"
            for e in (part.strip().lower() for part in override.split(","))
            if e
        ]
