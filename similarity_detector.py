"""
Duplicate/Near-Duplicate Detection using multiple image hashing techniques
"""
import imagehash
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import logging
from models import ImageMetrics


class SimilarityDetector:
    """Detect duplicate and similar images in a collection"""
    
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.hash_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def find_similar_groups(self, images: List[Path]) -> List[List[Path]]:
        """Group similar images together"""
        groups = []
        processed = set()
        
        self.logger.info(f"Finding similar groups among {len(images)} images...")
        
        # Calculate hashes for all images
        for i, img_path in enumerate(images):
            if img_path not in self.hash_cache:
                try:
                    with Image.open(img_path) as img:
                        # Resize for consistent hashing
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        
                        # Multiple hash types for robustness
                        self.hash_cache[img_path] = {
                            'average': imagehash.average_hash(img),
                            'perceptual': imagehash.phash(img),
                            'difference': imagehash.dhash(img),
                            'wavelet': imagehash.whash(img)
                        }
                except Exception as e:
                    self.logger.warning(f"Error hashing {img_path}: {e}")
                    continue
                
                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Hashed {i + 1}/{len(images)} images")
        
        # Find similar groups
        for i, img1 in enumerate(images):
            if img1 in processed or img1 not in self.hash_cache:
                continue
            
            group = [img1]
            processed.add(img1)
            
            for img2 in images[i+1:]:
                if img2 in processed or img2 not in self.hash_cache:
                    continue
                
                if self._are_similar(img1, img2):
                    group.append(img2)
                    processed.add(img2)
            
            if len(group) > 1:
                groups.append(group)
                self.logger.debug(f"Found similar group of {len(group)} images")
        
        self.logger.info(f"Found {len(groups)} similar groups")
        return groups
    
    def _are_similar(self, img1: Path, img2: Path) -> bool:
        """Check if two images are similar"""
        if img1 not in self.hash_cache or img2 not in self.hash_cache:
            return False
        
        hashes1 = self.hash_cache[img1]
        hashes2 = self.hash_cache[img2]
        
        # Calculate similarity scores for each hash type
        similarities = []
        weights = {'average': 0.25, 'perceptual': 0.35, 'difference': 0.25, 'wavelet': 0.15}
        
        for hash_type in hashes1:
            if hash_type in hashes2:
                # Hamming distance converted to similarity
                distance = hashes1[hash_type] - hashes2[hash_type]
                similarity = 1 - (distance / 64.0)  # 64 bits max for most hashes
                similarities.append(similarity * weights[hash_type])
        
        # Weighted average similarity across all hash types
        avg_similarity = sum(similarities) if similarities else 0
        return avg_similarity >= self.threshold
    
    def pick_best_from_group(self, group: List[Path], 
                            metrics: Dict[Path, ImageMetrics]) -> Path:
        """Select the best image from a similar group"""
        best_score = -1
        best_image = group[0]
        
        # Score based on overall quality from metrics
        for img_path in group:
            if img_path in metrics:
                score = metrics[img_path].overall_quality
                if score > best_score:
                    best_score = score
                    best_image = img_path
        
        # Fallback: if no metrics available, pick by file size (larger likely better quality)
        if best_score == -1:
            best_image = max(group, key=lambda p: p.stat().st_size)
        
        return best_image
    
    def analyze_group_patterns(self, group: List[Path]) -> Dict:
        """Analyze patterns in a similar group (burst shots, etc.)"""
        analysis = {
            'is_burst_sequence': False,
            'time_span_seconds': 0,
            'filename_pattern': None,
            'likely_burst': False
        }
        
        try:
            # Check file modification times
            times = [p.stat().st_mtime for p in group]
            analysis['time_span_seconds'] = max(times) - min(times)
            
            # If all photos taken within 30 seconds, likely burst
            if analysis['time_span_seconds'] <= 30:
                analysis['is_burst_sequence'] = True
                analysis['likely_burst'] = True
            
            # Check filename patterns
            names = [p.stem for p in group]
            if len(names) > 1:
                # Look for sequential numbering
                try:
                    numbers = [int(''.join(filter(str.isdigit, name))) for name in names if any(c.isdigit() for c in name)]
                    if len(numbers) == len(names):
                        sorted_nums = sorted(numbers)
                        if sorted_nums[-1] - sorted_nums[0] == len(sorted_nums) - 1:
                            analysis['filename_pattern'] = 'sequential'
                            analysis['likely_burst'] = True
                except:
                    pass
            
        except Exception as e:
            self.logger.warning(f"Error analyzing group patterns: {e}")
        
        return analysis
    
    def get_deletion_candidates(self, groups: List[List[Path]], 
                               metrics: Dict[Path, ImageMetrics]) -> List[Tuple[Path, str]]:
        """Get deletion candidates from similar groups with reasons"""
        candidates = []
        
        for group in groups:
            if len(group) < 2:
                continue
            
            best_image = self.pick_best_from_group(group, metrics)
            group_analysis = self.analyze_group_patterns(group)
            
            for img_path in group:
                if img_path != best_image:
                    reason = "Similar to better image"
                    
                    # More specific reasons based on analysis
                    if group_analysis['likely_burst']:
                        reason = "Duplicate from burst sequence"
                    elif img_path in metrics and best_image in metrics:
                        best_quality = metrics[best_image].overall_quality
                        this_quality = metrics[img_path].overall_quality
                        if this_quality < best_quality - 0.1:
                            reason = f"Lower quality ({this_quality:.2f} vs {best_quality:.2f})"
                    
                    candidates.append((img_path, reason))
        
        return candidates
    
    def get_similarity_stats(self, images: List[Path]) -> Dict:
        """Get statistics about image similarity in the collection"""
        groups = self.find_similar_groups(images)
        
        total_similar = sum(len(group) for group in groups)
        total_groups = len(groups)
        
        # Group size distribution
        group_sizes = [len(group) for group in groups]
        
        stats = {
            'total_images': len(images),
            'similar_images': total_similar,
            'similarity_groups': total_groups,
            'unique_images': len(images) - total_similar + total_groups,  # Remove duplicates, keep one from each group
            'potential_deletions': total_similar - total_groups,
            'largest_group_size': max(group_sizes) if group_sizes else 0,
            'average_group_size': np.mean(group_sizes) if group_sizes else 0,
            'storage_savings_potential': total_similar - total_groups
        }
        
        # Burst sequence detection
        burst_groups = 0
        for group in groups:
            analysis = self.analyze_group_patterns(group)
            if analysis['likely_burst']:
                burst_groups += 1
        
        stats['burst_sequences'] = burst_groups
        stats['burst_percentage'] = (burst_groups / total_groups * 100) if total_groups > 0 else 0
        
        return stats