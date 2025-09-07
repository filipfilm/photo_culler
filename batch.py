from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from extractor import RawThumbnailExtractor
from analyzer import HybridAnalyzer
from decision import CullingDecisionEngine
from models import ImageMetrics, CullResult, ProcessingMode
import logging
from PIL import Image
import json
import hashlib


class BatchCuller:
    def __init__(self, cache_dir: Optional[Path] = None, 
                 mode: ProcessingMode = ProcessingMode.ACCURATE,
                 max_workers: int = 4,
                 batch_size: int = 8,
                 force_cpu: bool = False,
                 use_ollama: bool = False,
                 ollama_model: str = "llava:13b"):
        
        self.cache_dir = cache_dir
        self.mode = mode
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.extractor = RawThumbnailExtractor(cache_dir)
        self.analyzer = HybridAnalyzer(
            mode=mode, 
            force_cpu=force_cpu,
            use_ollama=use_ollama,
            ollama_model=ollama_model
        )
        self.decision_engine = CullingDecisionEngine()
        
        # Results cache
        if cache_dir:
            self.results_cache_file = cache_dir / "cull_results.json"
            self.results_cache = self._load_cache()
        else:
            self.results_cache_file = None
            self.results_cache = {}
            
    def _load_cache(self) -> Dict:
        """Load cached results"""
        if self.results_cache_file and self.results_cache_file.exists():
            try:
                with open(self.results_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save results cache"""
        if self.results_cache_file:
            self.results_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_cache_file, 'w') as f:
                json.dump(self.results_cache, f)
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Get file hash for cache key"""
        stat = filepath.stat()
        # Include model in cache key
        model_id = self.ollama_model if self.use_ollama else "clip"
        return f"{filepath.name}_{stat.st_size}_{stat.st_mtime}_{model_id}"
    
    def process_image(self, filepath: Path) -> Optional[CullResult]:
        """Process single image"""
        start = time.perf_counter()
        
        # Check cache
        file_hash = self._get_file_hash(filepath)
        cache_key = f"{file_hash}_{self.mode.value}"
        
        if cache_key in self.results_cache:
            cached = self.results_cache[cache_key]
            self.logger.debug(f"Using cached result for {filepath.name}")
            return CullResult(
                filepath=filepath,
                decision=cached['decision'],
                confidence=cached['confidence'],
                metrics=ImageMetrics(
                    blur_score=cached['metrics']['blur_score'],
                    exposure_score=cached['metrics']['exposure_score'],
                    composition_score=cached['metrics']['composition_score'],
                    overall_quality=cached['metrics']['overall_quality'],
                    processing_mode=ProcessingMode(cached['metrics']['processing_mode'])
                ),
                issues=cached['issues'],
                processing_ms=cached['processing_ms'],
                mode=ProcessingMode(cached['mode'])
            )
        
        # Extract thumbnail
        image = self.extractor.extract(filepath)
        if image is None:
            return None
            
        # Analyze
        metrics = self.analyzer.analyze(image)
        
        # Decide
        processing_ms = int((time.perf_counter() - start) * 1000)
        result = self.decision_engine.decide(filepath, metrics, processing_ms)
        
        # Cache result
        self.results_cache[cache_key] = {
            'filepath': str(filepath),
            'decision': result.decision,
            'confidence': result.confidence,
            'metrics': {
                'blur_score': metrics.blur_score,
                'exposure_score': metrics.exposure_score,
                'composition_score': metrics.composition_score,
                'overall_quality': metrics.overall_quality,
                'processing_mode': metrics.processing_mode.value
            },
            'issues': result.issues,
            'processing_ms': result.processing_ms,
            'mode': result.mode.value
        }
        
        return result
    
    def process_folder_batch(self, folder: Path, 
                           extensions: List[str] = ['.nef', '.cr2', '.arw', '.jpg', '.jpeg']) -> Dict:
        """Process folder using batch processing for vision model"""
        
        # Find all image files
        files = []
        for ext in extensions:
            files.extend(folder.glob(f'*{ext}'))
            files.extend(folder.glob(f'*{ext.upper()}'))
        
        files = sorted(files)  # Consistent ordering
        self.logger.info(f"Processing {len(files)} files in {self.mode.value} mode")
        
        results = {
            'Keep': [],
            'Delete': [],
            'Review': [],
            'Failed': []
        }
        
        if self.mode == ProcessingMode.ACCURATE and self.analyzer.vision_analyzer:
            # Batch processing for vision model
            processed = 0
            
            for i in range(0, len(files), self.batch_size):
                batch_files = files[i:i + self.batch_size]
                batch_images = []
                batch_paths = []
                cached_results = []
                
                # Extract thumbnails and check cache
                for filepath in batch_files:
                    file_hash = self._get_file_hash(filepath)
                    cache_key = f"{file_hash}_{self.mode.value}"
                    
                    if cache_key in self.results_cache:
                        cached = self.results_cache[cache_key]
                        result = CullResult(
                            filepath=filepath,
                            decision=cached['decision'],
                            confidence=cached['confidence'],
                            metrics=ImageMetrics(
                                blur_score=cached['metrics']['blur_score'],
                                exposure_score=cached['metrics']['exposure_score'],
                                composition_score=cached['metrics']['composition_score'],
                                overall_quality=cached['metrics']['overall_quality'],
                                processing_mode=ProcessingMode(cached['metrics']['processing_mode'])
                            ),
                            issues=cached['issues'],
                            processing_ms=cached['processing_ms'],
                            mode=ProcessingMode(cached['mode'])
                        )
                        cached_results.append(result)
                    else:
                        image = self.extractor.extract(filepath)
                        if image:
                            batch_images.append(image)
                            batch_paths.append(filepath)
                        else:
                            results['Failed'].append(filepath)
                
                # Add cached results
                for result in cached_results:
                    results[result.decision].append(result)
                
                # Process uncached batch
                if batch_images:
                    start = time.perf_counter()
                    metrics_list = self.analyzer.analyze_batch(batch_images)
                    batch_time = (time.perf_counter() - start) * 1000
                    
                    for filepath, metrics in zip(batch_paths, metrics_list):
                        processing_ms = int(batch_time / len(batch_images))
                        result = self.decision_engine.decide(filepath, metrics, processing_ms)
                        results[result.decision].append(result)
                        
                        # Cache
                        file_hash = self._get_file_hash(filepath)
                        cache_key = f"{file_hash}_{self.mode.value}"
                        self.results_cache[cache_key] = {
                            'filepath': str(filepath),
                            'decision': result.decision,
                            'confidence': result.confidence,
                            'metrics': {
                                'blur_score': metrics.blur_score,
                                'exposure_score': metrics.exposure_score,
                                'composition_score': metrics.composition_score,
                                'overall_quality': metrics.overall_quality,
                                'processing_mode': metrics.processing_mode.value
                            },
                            'issues': result.issues,
                            'processing_ms': result.processing_ms,
                            'mode': result.mode.value
                        }
                
                processed += len(batch_files)
                self.logger.info(f"Processed {processed}/{len(files)}")
                
        else:
            # Fast mode - use parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_image, f): f 
                    for f in files
                }
                
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            results[result.decision].append(result)
                        else:
                            results['Failed'].append(filepath)
                    except Exception as e:
                        self.logger.error(f"Error processing {filepath}: {e}")
                        results['Failed'].append(filepath)
        
        # Save cache
        self._save_cache()
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print processing summary"""
        total = sum(len(v) for v in results.values())
        self.logger.info(f"\nProcessed {total} files:")
        for decision, items in results.items():
            if items:
                self.logger.info(f"  {decision}: {len(items)}")
                
        # Calculate average processing time
        all_results = []
        for items in results.values():
            if isinstance(items, list) and items and hasattr(items[0], 'processing_ms'):
                all_results.extend(items)
        
        if all_results:
            avg_time = sum(r.processing_ms for r in all_results) / len(all_results)
            self.logger.info(f"Average processing time: {avg_time:.0f}ms")