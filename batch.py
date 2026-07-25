from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import hashlib
import json
import logging
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .extractor import RawThumbnailExtractor
    from .models import CullResult, ImageMetrics
    from .ollama_vision import OllamaVisionAnalyzer
except ImportError:
    from extractor import RawThumbnailExtractor
    from models import CullResult, ImageMetrics
    from ollama_vision import OllamaVisionAnalyzer

try:
    from .blur_detector import HybridBlurDetector
except ImportError:
    try:
        from blur_detector import HybridBlurDetector
    except ImportError:
        HybridBlurDetector = None


class BatchCuller:
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        mode: str = "accurate",
        max_workers: int = 4,
        batch_size: int = 8,
        use_ollama: bool = True,
        ollama_model: str = "gemma4:e4b",
        ollama_host: str = "http://localhost:11434",
        force_cpu: bool = False,
        learning_enabled: bool = False,
    ):
        self.cache_dir = cache_dir
        self.mode = mode.lower()
        self.max_workers = max(1, max_workers)
        self.batch_size = max(1, batch_size)
        self.use_ollama = use_ollama and self.mode != "fast"
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.force_cpu = force_cpu
        self.learning_enabled = learning_enabled
        self.logger = logging.getLogger(__name__)
        self.extractor = RawThumbnailExtractor(cache_dir)
        self._session_results: List[CullResult] = []
        self._session_summary_cache: Optional[Dict] = None
        self.blur_detector = HybridBlurDetector() if HybridBlurDetector is not None else None

        if self.mode not in {"accurate", "fast"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if self.mode == "accurate":
            if not self.use_ollama:
                raise ValueError("Accurate mode requires Ollama. Use --fast for local analysis.")
            self.analyzer = OllamaVisionAnalyzer(model=ollama_model, host=ollama_host)
        else:
            self.analyzer = None

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, filepath: Path) -> Optional[Path]:
        """Get cache file path for a given image"""
        if not self.cache_dir:
            return None

        file_key = "|".join(
            [
                str(filepath.resolve()),
                str(filepath.stat().st_mtime_ns),
                self.mode,
                self.ollama_model if self.use_ollama else "local",
            ]
        )
        cache_name = hashlib.md5(file_key.encode()).hexdigest() + ".json"
        return self.cache_dir / cache_name

    def _load_cached_result(self, filepath: Path) -> Optional[CullResult]:
        """Load cached analysis result if available"""
        cache_path = self._get_cache_path(filepath)
        if not cache_path or not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            metrics = ImageMetrics(
                blur_score=cached["metrics"]["blur_score"],
                exposure_score=cached["metrics"]["exposure_score"],
                composition_score=cached["metrics"]["composition_score"],
                overall_quality=cached["metrics"]["overall_quality"],
                keywords=cached["metrics"].get("keywords", []),
                description=cached["metrics"].get("description", ""),
            )

            return CullResult(
                filepath=Path(cached["filepath"]),
                decision=cached["decision"],
                confidence=cached["confidence"],
                metrics=metrics,
                issues=cached.get("issues", self._identify_issues(metrics)),
                processing_ms=cached.get("processing_ms", 0.0),
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to load cache for {filepath}: {e}")
            return None

    def _save_cached_result(self, result: CullResult):
        """Save analysis result to cache"""
        cache_path = self._get_cache_path(result.filepath)
        if not cache_path or result.decision == "Failed":
            return

        try:
            cached_data = {
                "filepath": str(result.filepath),
                "decision": result.decision,
                "confidence": result.confidence,
                "issues": result.issues,
                "processing_ms": result.processing_ms,
                "metrics": {
                    "blur_score": result.metrics.blur_score,
                    "exposure_score": result.metrics.exposure_score,
                    "composition_score": result.metrics.composition_score,
                    "overall_quality": result.metrics.overall_quality,
                    "keywords": result.metrics.keywords or [],
                    "description": result.metrics.description or "",
                },
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2)
        except OSError as e:
            self.logger.warning(f"Failed to save cache for {result.filepath}: {e}")

    def _record_session_result(self, result: CullResult):
        self._session_results.append(result)
        self._session_summary_cache = None

    def _make_decision(self, metrics: ImageMetrics) -> tuple[str, float]:
        """Decision logic optimized for the simplified analyzer set."""
        if metrics.blur_score < 0.35:
            return "Delete", 0.8
        if metrics.blur_score < 0.50:
            return "Delete", 0.7
        if metrics.exposure_score < 0.3:
            return "Delete", 0.6
        if metrics.overall_quality < 0.4:
            return "Delete", 0.5
        if metrics.overall_quality > 0.7:
            return "Keep", 0.8
        return "Review", 0.5

    def _identify_issues(self, metrics: ImageMetrics) -> List[str]:
        issues = []
        if metrics.blur_score < 0.35:
            issues.append("critical blur")
        elif metrics.blur_score < 0.50:
            issues.append("soft focus")

        if metrics.exposure_score < 0.30:
            issues.append("poor exposure")

        if metrics.composition_score < 0.35:
            issues.append("weak composition")

        if metrics.overall_quality < 0.40 and not issues:
            issues.append("low overall quality")

        return issues

    def _fallback_blur_score(self, gray_image: np.ndarray) -> float:
        gradient_y = np.abs(np.diff(gray_image, axis=0)).mean() if gray_image.shape[0] > 1 else 0.0
        gradient_x = np.abs(np.diff(gray_image, axis=1)).mean() if gray_image.shape[1] > 1 else 0.0
        return float(np.clip((gradient_x + gradient_y) / 24.0, 0.0, 1.0))

    def _analyze_fast(self, image: Image.Image) -> ImageMetrics:
        local_image = image.convert("RGB")
        local_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        gray_image = np.asarray(local_image.convert("L"), dtype=np.float32)

        if self.blur_detector is not None:
            blur_score = self.blur_detector.detect_cv_blur(local_image)["cv_sharpness_score"]
        else:
            blur_score = self._fallback_blur_score(gray_image)

        mean_exposure = float(gray_image.mean() / 255.0)
        exposure_score = float(np.clip(1.0 - abs(mean_exposure - 0.5) / 0.5, 0.0, 1.0))

        contrast_score = float(np.clip(gray_image.std() / 64.0, 0.0, 1.0))
        thirds_rows = [gray_image.shape[0] // 3, (2 * gray_image.shape[0]) // 3]
        thirds_cols = [gray_image.shape[1] // 3, (2 * gray_image.shape[1]) // 3]
        interest_values = [
            gray_image[min(r, gray_image.shape[0] - 1), min(c, gray_image.shape[1] - 1)]
            for r in thirds_rows
            for c in thirds_cols
        ]
        interest_score = float(np.clip(np.std(interest_values) / 48.0, 0.0, 1.0))
        composition_score = float(np.clip(0.65 * contrast_score + 0.35 * interest_score, 0.0, 1.0))

        overall_quality = float(
            np.clip(
                0.5 * blur_score + 0.3 * exposure_score + 0.2 * composition_score,
                0.0,
                1.0,
            )
        )

        return ImageMetrics(
            blur_score=blur_score,
            exposure_score=exposure_score,
            composition_score=composition_score,
            overall_quality=overall_quality,
            keywords=[],
            description="Fast local analysis",
        )

    def _build_failed_result(self, filepath: Path, issue: str, start_time: float) -> CullResult:
        processing_ms = (time.time() - start_time) * 1000
        return CullResult(
            filepath=filepath,
            decision="Failed",
            confidence=0.0,
            metrics=ImageMetrics(
                blur_score=0.0,
                exposure_score=0.0,
                composition_score=0.0,
                overall_quality=0.0,
                keywords=[],
                description="",
            ),
            issues=[issue],
            processing_ms=processing_ms,
        )

    def _process_single_image(self, filepath: Path) -> CullResult:
        """Process a single image file."""
        start_time = time.time()

        cached_result = self._load_cached_result(filepath)
        if cached_result is not None:
            self._record_session_result(cached_result)
            return cached_result

        try:
            image = self.extractor.extract_thumbnail(filepath)
            if image is None:
                result = self._build_failed_result(filepath, "unsupported or unreadable image", start_time)
                self._record_session_result(result)
                return result

            if self.mode == "fast":
                metrics = self._analyze_fast(image)
            else:
                metrics = self.analyzer.analyze(image)

            decision, confidence = self._make_decision(metrics)
            result = CullResult(
                filepath=filepath,
                decision=decision,
                confidence=confidence,
                metrics=metrics,
                issues=self._identify_issues(metrics),
                processing_ms=(time.time() - start_time) * 1000,
            )
            self._save_cached_result(result)
            self._record_session_result(result)
            return result
        except Exception as e:
            self.logger.error(f"Failed to process {filepath}: {e}")
            result = self._build_failed_result(filepath, str(e), start_time)
            self._record_session_result(result)
            return result

    def process_image(self, filepath: Path) -> CullResult:
        return self._process_single_image(filepath)

    def _find_image_files(self, folder_path: Path, extensions: Sequence[str]) -> List[Path]:
        normalized_exts = []
        for ext in extensions:
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = "." + normalized
            normalized_exts.append(normalized)

        image_files = set()
        for ext in normalized_exts:
            image_files.update(folder_path.glob(f"**/*{ext}"))
            image_files.update(folder_path.glob(f"**/*{ext.upper()}"))

        return sorted(image_files)

    def cull_folder(
        self,
        folder_path: Path,
        extensions: Sequence[str] = (".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".jpg", ".jpeg"),
        progress_callback=None,
    ) -> List[CullResult]:
        """Process all images in a folder."""
        image_files = self._find_image_files(folder_path, extensions)

        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []

        self.logger.info(f"Found {len(image_files)} images to process")
        results: List[CullResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_image, filepath): filepath for filepath in image_files
            }

            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        self.logger.error(f"Processing failed for {filepath}: {e}")
                        result = self._build_failed_result(filepath, str(e), time.time())
                        self._record_session_result(result)

                    results.append(result)

                    if progress_callback:
                        progress_callback(result)

                    pbar.update(1)

        return results

    def process_folder_batch(self, folder_path: Path, extensions: Sequence[str]) -> Dict[str, List[CullResult]]:
        grouped_results = {"Keep": [], "Delete": [], "Review": [], "Failed": []}
        for result in self.cull_folder(folder_path, extensions):
            grouped_results.setdefault(result.decision, []).append(result)
        return grouped_results

    def get_session_summary(self) -> Dict:
        if self._session_summary_cache is not None:
            return self._session_summary_cache

        valid_results = [result for result in self._session_results if result.decision != "Failed"]
        if not valid_results:
            self._session_summary_cache = {"total_processed": 0, "detected_style": {}}
            return self._session_summary_cache

        keywords = Counter(
            keyword
            for result in valid_results
            for keyword in (result.metrics.keywords or [])
            if keyword
        )

        summary = {
            "total_processed": len(valid_results),
            "avg_blur": sum(r.metrics.blur_score for r in valid_results) / len(valid_results),
            "avg_exposure": sum(r.metrics.exposure_score for r in valid_results) / len(valid_results),
            "avg_composition": sum(r.metrics.composition_score for r in valid_results) / len(valid_results),
            "detected_style": {},
        }

        if keywords:
            summary["detected_style"]["common_subjects"] = [
                keyword for keyword, _ in keywords.most_common(3)
            ]

        self._session_summary_cache = summary
        return summary

    def save_session(self):
        if not self.cache_dir:
            return

        session_summary = self.get_session_summary()
        if session_summary.get("total_processed", 0) == 0:
            return

        session_path = self.cache_dir / "session_summary.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_summary, f, indent=2)

    def get_statistics(self, results: Iterable[CullResult]) -> Dict:
        results = list(results)
        valid_results = [result for result in results if result.decision != "Failed"]
        if not results:
            return {}

        stats = {
            "total_images": len(results),
            "decisions": {},
            "avg_scores": {},
            "keywords_found": 0,
        }

        for result in results:
            stats["decisions"][result.decision] = stats["decisions"].get(result.decision, 0) + 1

        if valid_results:
            stats["avg_scores"] = {
                "blur": sum(r.metrics.blur_score for r in valid_results) / len(valid_results),
                "exposure": sum(r.metrics.exposure_score for r in valid_results) / len(valid_results),
                "composition": sum(r.metrics.composition_score for r in valid_results) / len(valid_results),
                "overall": sum(r.metrics.overall_quality for r in valid_results) / len(valid_results),
            }
            stats["keywords_found"] = len([r for r in valid_results if r.metrics.keywords])

        return stats
