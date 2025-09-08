"""
Smart Image Sequencing for Batch Processing
Intelligently groups and sequences images for optimal processing
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class ImageGroup:
    """A group of related images"""
    files: List[Path]
    group_type: str  # 'burst', 'session', 'location', 'single'
    time_span: float  # seconds
    priority: int  # Processing priority (lower = higher priority)
    metadata: Dict  # Additional group metadata


class SmartSequencer:
    """Intelligent image sequencing for batch processing"""
    
    def __init__(self, burst_threshold_seconds: int = 10, 
                 session_threshold_minutes: int = 30):
        self.burst_threshold = burst_threshold_seconds
        self.session_threshold = session_threshold_minutes * 60  # Convert to seconds
        self.logger = logging.getLogger(__name__)
    
    def sequence_for_processing(self, files: List[Path]) -> List[ImageGroup]:
        """Group and sequence files for optimal processing"""
        self.logger.info(f"Sequencing {len(files)} files for optimal processing...")
        
        # Step 1: Extract timing information
        file_times = self._extract_file_times(files)
        
        # Step 2: Group by time proximity
        time_groups = self._group_by_time_proximity(file_times)
        
        # Step 3: Analyze each group for patterns
        analyzed_groups = []
        for group in time_groups:
            analyzed_group = self._analyze_group(group)
            analyzed_groups.append(analyzed_group)
        
        # Step 4: Sort by processing priority
        analyzed_groups.sort(key=lambda g: g.priority)
        
        self.logger.info(f"Created {len(analyzed_groups)} processing groups")
        self._log_group_summary(analyzed_groups)
        
        return analyzed_groups
    
    def _extract_file_times(self, files: List[Path]) -> List[Tuple[Path, float, Dict]]:
        """Extract timing and metadata for each file"""
        file_data = []
        
        for filepath in files:
            try:
                stat = filepath.stat()
                
                # Use modification time as proxy for capture time
                # In practice, you might want to extract EXIF date taken
                timestamp = stat.st_mtime
                
                # Extract additional metadata for grouping
                metadata = {
                    'size': stat.st_size,
                    'name_pattern': self._extract_name_pattern(filepath),
                    'extension': filepath.suffix.lower(),
                }
                
                file_data.append((filepath, timestamp, metadata))
                
            except Exception as e:
                self.logger.warning(f"Failed to get timing for {filepath}: {e}")
                # Use current time as fallback
                file_data.append((filepath, datetime.now().timestamp(), {}))
        
        # Sort by timestamp
        file_data.sort(key=lambda x: x[1])
        return file_data
    
    def _extract_name_pattern(self, filepath: Path) -> Optional[str]:
        """Extract naming pattern from filename"""
        name = filepath.stem
        
        # Look for common camera naming patterns
        patterns = [
            r'([A-Z]+)(\d+)',  # e.g., DSC1234, IMG5678
            r'(\d{8})_(\d{6})',  # e.g., 20231201_143022
            r'([A-Z]+_\d{4})(\d+)',  # e.g., IMG_1234
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                return match.group(1)
        
        # Generic pattern: letters followed by numbers
        match = re.match(r'([A-Za-z_-]+)', name)
        if match:
            return match.group(1)
        
        return None
    
    def _group_by_time_proximity(self, file_data: List[Tuple[Path, float, Dict]]) -> List[List[Tuple[Path, float, Dict]]]:
        """Group files by time proximity"""
        if not file_data:
            return []
        
        groups = []
        current_group = [file_data[0]]
        
        for i in range(1, len(file_data)):
            current_file = file_data[i]
            previous_file = file_data[i-1]
            
            time_diff = current_file[1] - previous_file[1]
            
            # Check if this file belongs to current group
            if self._should_group_together(current_file, previous_file, time_diff):
                current_group.append(current_file)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [current_file]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _should_group_together(self, current: Tuple[Path, float, Dict], 
                              previous: Tuple[Path, float, Dict], 
                              time_diff: float) -> bool:
        """Determine if two files should be grouped together"""
        
        # Very close in time (likely burst)
        if time_diff <= self.burst_threshold:
            return True
        
        # Moderately close in time with similar naming pattern
        if (time_diff <= self.session_threshold and 
            current[2].get('name_pattern') == previous[2].get('name_pattern')):
            return True
        
        # Same file type and reasonable time gap
        if (time_diff <= self.session_threshold * 2 and
            current[2].get('extension') == previous[2].get('extension')):
            return True
        
        return False
    
    def _analyze_group(self, group: List[Tuple[Path, float, Dict]]) -> ImageGroup:
        """Analyze a group to determine its characteristics and priority"""
        
        files = [item[0] for item in group]
        timestamps = [item[1] for item in group]
        metadata_list = [item[2] for item in group]
        
        # Calculate time span
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        # Determine group type and priority
        group_type, priority = self._classify_group(group, time_span)
        
        # Extract group metadata
        group_metadata = self._extract_group_metadata(group, metadata_list)
        
        return ImageGroup(
            files=files,
            group_type=group_type,
            time_span=time_span,
            priority=priority,
            metadata=group_metadata
        )
    
    def _classify_group(self, group: List[Tuple[Path, float, Dict]], 
                       time_span: float) -> Tuple[str, int]:
        """Classify group type and assign processing priority"""
        
        group_size = len(group)
        
        # Single image
        if group_size == 1:
            return 'single', 3  # Medium priority
        
        # Burst sequence (multiple images in short time)
        if time_span <= self.burst_threshold and group_size >= 3:
            return 'burst', 1  # High priority - process burst together for comparison
        
        # Short session (moderate time span, fewer images)
        if time_span <= self.session_threshold and group_size <= 10:
            return 'session', 2  # Higher priority - likely intentional session
        
        # Long session or large group
        if group_size > 10 or time_span > self.session_threshold:
            return 'location', 4  # Lower priority - can be processed in chunks
        
        # Default
        return 'session', 3
    
    def _extract_group_metadata(self, group: List[Tuple[Path, float, Dict]], 
                               metadata_list: List[Dict]) -> Dict:
        """Extract metadata about the group"""
        
        group_metadata = {
            'size': len(group),
            'total_file_size': sum(m.get('size', 0) for m in metadata_list),
            'file_types': list(set(m.get('extension', '') for m in metadata_list)),
            'name_patterns': list(set(m.get('name_pattern', '') for m in metadata_list if m.get('name_pattern'))),
        }
        
        # Detect potential burst characteristics
        if len(group) >= 3:
            files = [item[0] for item in group]
            group_metadata['likely_burst'] = self._is_likely_burst_sequence(files)
            group_metadata['sequential_names'] = self._has_sequential_names(files)
        
        # Calculate processing estimate
        avg_file_size = group_metadata['total_file_size'] / len(group) if len(group) > 0 else 0
        group_metadata['estimated_processing_time'] = self._estimate_processing_time(
            len(group), avg_file_size, group_metadata.get('likely_burst', False)
        )
        
        return group_metadata
    
    def _is_likely_burst_sequence(self, files: List[Path]) -> bool:
        """Check if files represent a burst sequence"""
        
        # Check for sequential numbering in filenames
        numbers = []
        for f in files:
            # Extract numbers from filename
            nums = re.findall(r'\d+', f.stem)
            if nums:
                try:
                    numbers.append(int(nums[-1]))  # Use last number found
                except ValueError:
                    continue
        
        if len(numbers) >= len(files) * 0.8:  # At least 80% have numbers
            numbers.sort()
            # Check if mostly sequential
            sequential_count = 0
            for i in range(1, len(numbers)):
                if numbers[i] - numbers[i-1] <= 2:  # Allow small gaps
                    sequential_count += 1
            
            return sequential_count >= len(numbers) * 0.7  # At least 70% sequential
        
        return False
    
    def _has_sequential_names(self, files: List[Path]) -> bool:
        """Check if files have sequential naming"""
        if len(files) < 2:
            return False
        
        # Extract base names and numbers
        patterns = []
        for f in files:
            name = f.stem
            # Find pattern of letters followed by numbers
            match = re.match(r'([A-Za-z_-]+)(\d+)', name)
            if match:
                base, num = match.groups()
                patterns.append((base, int(num)))
        
        if len(patterns) >= len(files) * 0.8:  # Most files follow pattern
            # Group by base name
            by_base = defaultdict(list)
            for base, num in patterns:
                by_base[base].append(num)
            
            # Check if any base has sequential numbers
            for base, numbers in by_base.items():
                if len(numbers) >= 3:  # At least 3 sequential files
                    numbers.sort()
                    sequential = all(
                        numbers[i] - numbers[i-1] <= 2 
                        for i in range(1, len(numbers))
                    )
                    if sequential:
                        return True
        
        return False
    
    def _estimate_processing_time(self, file_count: int, avg_file_size: float, 
                                 is_burst: bool) -> float:
        """Estimate processing time for the group in seconds"""
        
        # Base time per file (depends on processing mode and file size)
        base_time_per_file = 2.0  # seconds for fast mode
        
        # Adjust for file size (larger files take longer)
        size_mb = avg_file_size / (1024 * 1024)
        size_factor = max(1.0, size_mb / 10)  # Scale by 10MB chunks
        
        # Burst sequences can be processed more efficiently together
        burst_factor = 0.8 if is_burst and file_count > 3 else 1.0
        
        total_time = file_count * base_time_per_file * size_factor * burst_factor
        
        return total_time
    
    def _log_group_summary(self, groups: List[ImageGroup]):
        """Log summary of created groups"""
        
        type_counts = defaultdict(int)
        total_files = 0
        total_time_estimate = 0
        
        for group in groups:
            type_counts[group.group_type] += 1
            total_files += len(group.files)
            total_time_estimate += group.metadata.get('estimated_processing_time', 0)
        
        self.logger.info("Group summary:")
        for group_type, count in type_counts.items():
            self.logger.info(f"  {group_type}: {count} groups")
        
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Estimated processing time: {total_time_estimate:.1f} seconds")
    
    def get_processing_order_summary(self, groups: List[ImageGroup]) -> Dict:
        """Get summary of processing order for reporting"""
        
        summary = {
            'total_groups': len(groups),
            'total_files': sum(len(g.files) for g in groups),
            'groups_by_type': defaultdict(int),
            'groups_by_priority': defaultdict(int),
            'processing_order': []
        }
        
        for i, group in enumerate(groups):
            summary['groups_by_type'][group.group_type] += 1
            summary['groups_by_priority'][group.priority] += 1
            
            summary['processing_order'].append({
                'order': i + 1,
                'type': group.group_type,
                'file_count': len(group.files),
                'time_span_minutes': group.time_span / 60,
                'estimated_time': group.metadata.get('estimated_processing_time', 0),
                'priority': group.priority
            })
        
        return summary