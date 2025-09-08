from typing import List
from pathlib import Path
from models import ImageMetrics, CullResult, ProcessingMode
import logging


class CullingDecisionEngine:
    """Make culling decisions based on image metrics"""
    
    def __init__(self, 
                 blur_threshold: float = 0.15,  # Much more realistic - was way too high
                 exposure_threshold: float = 0.25, # More realistic
                 overall_threshold: float = 0.35,  # More realistic  
                 delete_confidence_threshold: float = 0.7):
        
        self.blur_threshold = blur_threshold
        self.exposure_threshold = exposure_threshold
        self.overall_threshold = overall_threshold
        self.delete_confidence_threshold = delete_confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def decide(self, filepath: Path, metrics: ImageMetrics, processing_ms: int) -> CullResult:
        """Make culling decision based on metrics"""
        
        issues = []
        decision = "Keep"
        confidence = 0.0
        
        # Check for critical issues
        if metrics.blur_score < self.blur_threshold:
            issues.append("blurry")
            severity = 1.0 - metrics.blur_score
            confidence = max(confidence, severity)
        
        if metrics.exposure_score < self.exposure_threshold:
            issues.append("poor exposure")
            severity = 1.0 - metrics.exposure_score
            confidence = max(confidence, severity * 0.8)  # Slightly less critical than blur
        
        if metrics.composition_score < 0.2:
            issues.append("poor composition")
            confidence = max(confidence, 0.3)  # Lower confidence for composition alone
        
        if metrics.overall_quality < self.overall_threshold:
            issues.append("low overall quality")
            severity = 1.0 - metrics.overall_quality
            confidence = max(confidence, severity * 0.6)
        
        # Make decision
        if confidence >= self.delete_confidence_threshold:
            decision = "Delete"
        elif confidence >= 0.4 or len(issues) >= 2:
            decision = "Review"
        else:
            decision = "Keep"
            confidence = 1.0 - confidence  # Invert for keep confidence
        
        # Adjust confidence based on processing mode
        if metrics.processing_mode == ProcessingMode.FAST and decision == "Delete":
            # Be more conservative with fast mode deletes
            confidence *= 0.8
            if confidence < self.delete_confidence_threshold:
                decision = "Review"
        
        return CullResult(
            filepath=filepath,
            decision=decision,
            confidence=confidence,
            metrics=metrics,
            issues=issues,
            processing_ms=processing_ms,
            mode=metrics.processing_mode
        )