"""Folder-level orchestration: caching, threading, and the fast/accurate modes."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import hashlib
import json
import logging
import time

from PIL import Image
from tqdm import tqdm

try:
    from .blur_detector import BlurDetector
    from .decision import CullDecider, DELETE, FAILED, KEEP, REVIEW
    from .exposure import ExposureMeter
    from .extractor import RawThumbnailExtractor
    from .grouping import annotate_results, compute_phash
    from .models import CullResult, ImageMetrics
    from .vision import OllamaVisionAnalyzer
except ImportError:
    from blur_detector import BlurDetector
    from decision import CullDecider, DELETE, FAILED, KEEP, REVIEW
    from exposure import ExposureMeter
    from extractor import RawThumbnailExtractor
    from grouping import annotate_results, compute_phash
    from models import CullResult, ImageMetrics
    from vision import OllamaVisionAnalyzer

# Bumped whenever prompts, schemas or decision rules change, so that cached results from
# an older version of the tool are recomputed instead of silently reused.
ANALYSIS_VERSION = "2"

DEFAULT_EXTENSIONS = (
    ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".rw2", ".jpg", ".jpeg",
)


class BatchCuller:
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        mode: str = "accurate",
        max_workers: int = 4,
        use_ollama: bool = True,
        ollama_model: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        timeout: int = 180,
        with_tags: bool = True,
        verify_vision: bool = True,
        context_tokens: int = 8192,
        learning_enabled: bool = False,
    ):
        self.cache_dir = cache_dir
        self.mode = mode.lower()
        self.max_workers = max(1, max_workers)
        self.ollama_host = ollama_host
        self.with_tags = with_tags
        self.learning_enabled = learning_enabled
        self.logger = logging.getLogger(__name__)

        self.extractor = RawThumbnailExtractor(cache_dir)
        self.blur_detector = BlurDetector()
        self.exposure_meter = ExposureMeter()
        self.decider = CullDecider()

        self._session_results: List[CullResult] = []
        self._session_summary_cache: Optional[Dict] = None

        if self.mode not in {"accurate", "fast"}:
            raise ValueError(f"Unsupported mode: {mode}. Use 'accurate' or 'fast'.")

        if self.mode == "accurate":
            if not use_ollama:
                raise ValueError(
                    "Accurate mode needs a vision model. Use --fast for measurement-only triage."
                )
            self.analyzer = OllamaVisionAnalyzer(
                model=ollama_model,
                host=ollama_host,
                timeout=timeout,
                verify_vision=verify_vision,
                context_tokens=context_tokens,
            )
            self.ollama_model = self.analyzer.model
        else:
            self.analyzer = None
            self.ollama_model = None

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ caching

    def _get_cache_path(self, filepath: Path) -> Optional[Path]:
        if not self.cache_dir:
            return None

        file_key = "|".join([
            str(filepath.resolve()),
            str(filepath.stat().st_mtime_ns),
            self.mode,
            self.ollama_model or "local",
            "tags" if self.with_tags else "notags",
            ANALYSIS_VERSION,
        ])
        return self.cache_dir / (hashlib.md5(file_key.encode()).hexdigest() + ".json")

    def _load_cached_result(self, filepath: Path) -> Optional[CullResult]:
        cache_path = self._get_cache_path(filepath)
        if not cache_path or not cache_path.exists():
            return None

        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            metrics = ImageMetrics(**cached["metrics"])
            captured = cached.get("capture_time")
            return CullResult(
                filepath=Path(cached["filepath"]),
                decision=cached["decision"],
                confidence=cached["confidence"],
                metrics=metrics,
                issues=cached.get("issues", []),
                processing_ms=cached.get("processing_ms", 0.0),
                capture_time=datetime.fromisoformat(captured) if captured else None,
                phash=cached.get("phash"),
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.logger.warning(f"Ignoring unreadable cache entry for {filepath.name}: {e}")
            return None

    def _save_cached_result(self, result: CullResult):
        cache_path = self._get_cache_path(result.filepath)
        if not cache_path or result.decision == FAILED:
            return

        try:
            cache_path.write_text(
                json.dumps({
                    "filepath": str(result.filepath),
                    "decision": result.decision,
                    "confidence": result.confidence,
                    "issues": result.issues,
                    "processing_ms": result.processing_ms,
                    "capture_time": result.capture_time.isoformat() if result.capture_time else None,
                    "phash": result.phash,
                    "metrics": vars(result.metrics),
                }, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            self.logger.warning(f"Failed to cache result for {result.filepath.name}: {e}")

    # ------------------------------------------------------------------ analysis

    def _analyze_fast(self, image: Image.Image) -> ImageMetrics:
        """Measurement-only triage: no model, therefore no deletions.

        With only one witness (see decision.py) fast mode is not entitled to reject a
        photograph, so its worst verdict is 'review'. It exists to shrink the pile a
        human or the accurate pass has to look at, not to make final calls.
        """
        cv_stats = self.blur_detector.measure(image)
        exposure_stats = self.exposure_meter.measure(image)

        cv_score = cv_stats.get("sharpness_score")
        if cv_score is None:
            sharpness = "acceptable"
        elif cv_stats["has_sharp_evidence"]:
            sharpness = "sharp"
        elif cv_score >= 0.35:
            sharpness = "acceptable"
        else:
            sharpness = "soft"

        exposure_category = exposure_stats["category"]
        suspect = sharpness == "soft" or exposure_category != "good"

        return ImageMetrics.from_triage(
            {
                "subject": "",
                "subject_sharpness": sharpness,
                "exposure": exposure_category,
                "framing": "fine",
                "technical_issues": [],
                "verdict": "review" if suspect else "keep",
                "verdict_reason": "measurement-only triage (fast mode)",
            },
            cv_sharpness=cv_score,
            exposure_stats=exposure_stats,
        )

    def _build_failed_result(self, filepath: Path, issue: str, start_time: float) -> CullResult:
        return CullResult(
            filepath=filepath,
            decision=FAILED,
            confidence=0.0,
            metrics=ImageMetrics(0.0, 0.0, 0.0, 0.0, keywords=[], description=""),
            issues=[issue],
            processing_ms=(time.time() - start_time) * 1000,
        )

    def _process_single_image(self, filepath: Path) -> CullResult:
        start_time = time.time()

        cached = self._load_cached_result(filepath)
        if cached is not None:
            self._record_session_result(cached)
            return cached

        try:
            image, info = self.extractor.extract_with_info(filepath)
            if image is None:
                result = self._build_failed_result(
                    filepath, "unsupported or unreadable image", start_time
                )
                self._record_session_result(result)
                return result

            if self.mode == "fast":
                metrics = self._analyze_fast(image)
            else:
                metrics = self.analyzer.analyze(image, with_tags=self.with_tags)

            decision, confidence, issues = self.decider.decide(metrics)
            result = CullResult(
                filepath=filepath,
                decision=decision,
                confidence=confidence,
                metrics=metrics,
                issues=issues,
                processing_ms=(time.time() - start_time) * 1000,
                capture_time=info.get("capture_time"),
                phash=compute_phash(image),
            )
            self._save_cached_result(result)
            self._record_session_result(result)
            return result

        except Exception as e:
            self.logger.error(f"Failed to process {filepath.name}: {e}")
            result = self._build_failed_result(filepath, str(e), start_time)
            self._record_session_result(result)
            return result

    def process_image(self, filepath: Path) -> CullResult:
        return self._process_single_image(filepath)

    # ------------------------------------------------------------------ folders

    def find_image_files(
        self, folder_path: Path, extensions: Sequence[str], recursive: bool = True
    ) -> List[Path]:
        normalized = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
        }
        pattern = "**/*" if recursive else "*"
        return sorted(
            p for p in folder_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in normalized
        )

    def cull_folder(
        self,
        folder_path: Path,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        progress_callback=None,
        recursive: bool = True,
        group_bursts: bool = True,
    ) -> List[CullResult]:
        image_files = self.find_image_files(folder_path, extensions, recursive)
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []

        self.logger.info(f"Found {len(image_files)} images to process")
        results: List[CullResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_image, path): path for path in image_files
            }
            with tqdm(total=len(image_files), desc="Analysing", unit="img") as pbar:
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        self.logger.error(f"Processing failed for {path.name}: {e}")
                        result = self._build_failed_result(path, str(e), time.time())
                        self._record_session_result(result)

                    results.append(result)
                    if progress_callback:
                        progress_callback(result)
                    pbar.update(1)

        if group_bursts:
            self.grouping_summary = annotate_results(results)
            self.logger.info(
                f"Burst grouping: {self.grouping_summary['bursts']} bursts, "
                f"{self.grouping_summary['demoted_to_review']} redundant frames moved to Review"
            )

        results.sort(key=lambda r: r.filepath.name)
        return results

    def process_folder_batch(
        self, folder_path: Path, extensions: Sequence[str]
    ) -> Dict[str, List[CullResult]]:
        grouped: Dict[str, List[CullResult]] = {KEEP: [], DELETE: [], REVIEW: [], FAILED: []}
        for result in self.cull_folder(folder_path, extensions):
            grouped.setdefault(result.decision, []).append(result)
        return grouped

    # ------------------------------------------------------------------ session

    def _record_session_result(self, result: CullResult):
        self._session_results.append(result)
        self._session_summary_cache = None

    def get_session_summary(self) -> Dict:
        if self._session_summary_cache is not None:
            return self._session_summary_cache

        valid = [r for r in self._session_results if r.decision != FAILED]
        if not valid:
            self._session_summary_cache = {"total_processed": 0, "detected_style": {}}
            return self._session_summary_cache

        keywords = Counter(
            kw for r in valid for kw in (r.metrics.keywords or []) if kw
        )
        subjects = Counter(
            r.metrics.subject for r in valid if r.metrics.subject
        )

        summary = {
            "total_processed": len(valid),
            "model": self.ollama_model or "local",
            "avg_blur": sum(r.metrics.blur_score for r in valid) / len(valid),
            "avg_exposure": sum(r.metrics.exposure_score for r in valid) / len(valid),
            "avg_composition": sum(r.metrics.composition_score for r in valid) / len(valid),
            "detected_style": {},
        }
        if keywords:
            summary["detected_style"]["common_keywords"] = [k for k, _ in keywords.most_common(5)]
        if subjects:
            summary["detected_style"]["common_subjects"] = [s for s, _ in subjects.most_common(3)]

        self._session_summary_cache = summary
        return summary

    def save_session(self):
        if not self.cache_dir:
            return
        summary = self.get_session_summary()
        if summary.get("total_processed", 0) == 0:
            return
        (self.cache_dir / "session_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def get_statistics(self, results: Iterable[CullResult]) -> Dict:
        results = list(results)
        if not results:
            return {}

        valid = [r for r in results if r.decision != FAILED]
        stats = {"total_images": len(results), "decisions": {}, "avg_scores": {}, "keywords_found": 0}

        for result in results:
            stats["decisions"][result.decision] = stats["decisions"].get(result.decision, 0) + 1

        if valid:
            stats["avg_scores"] = {
                "blur": sum(r.metrics.blur_score for r in valid) / len(valid),
                "exposure": sum(r.metrics.exposure_score for r in valid) / len(valid),
                "composition": sum(r.metrics.composition_score for r in valid) / len(valid),
                "overall": sum(r.metrics.overall_quality for r in valid) / len(valid),
            }
            stats["keywords_found"] = len([r for r in valid if r.metrics.keywords])

        return stats
