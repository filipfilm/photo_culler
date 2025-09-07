from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from enum import Enum

class ProcessingMode(Enum):
    FAST = "fast"  # Traditional CV
    ACCURATE = "accurate"  # Vision model
    
@dataclass
class ImageMetrics:
    blur_score: float  # 0-1, higher is sharper
    exposure_score: float  # 0-1, higher is better exposed
    composition_score: float  # 0-1, higher is more interesting
    overall_quality: float  # 0-1, weighted combination
    processing_mode: ProcessingMode
    keywords: Optional[List[str]] = None  # AI-generated keywords
    description: Optional[str] = None  # Natural language description
    
@dataclass
class CullResult:
    filepath: Path
    decision: str  # "Keep", "Delete", "Review"
    confidence: float  # 0-1
    metrics: ImageMetrics
    issues: List[str]
    processing_ms: int
    mode: ProcessingMode