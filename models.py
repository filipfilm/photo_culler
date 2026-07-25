from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class ImageMetrics:
    blur_score: float  # 0-1, higher is sharper
    exposure_score: float  # 0-1, higher is better exposed
    composition_score: float  # 0-1, higher is more interesting
    overall_quality: float  # 0-1, weighted combination
    keywords: Optional[List[str]] = None  # AI-generated keywords
    description: Optional[str] = None  # Natural language description
    
@dataclass
class CullResult:
    filepath: Path
    decision: str  # "Keep", "Delete", "Review"
    confidence: float  # 0-1
    metrics: ImageMetrics
    issues: List[str] = field(default_factory=list)
    processing_ms: float = 0.0
