"""
Adaptive decision engine that learns from user feedback
and adjusts thresholds based on photography style
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from models import ImageMetrics, CullResult, ProcessingMode
import logging
import numpy as np
from datetime import datetime

@dataclass
class DecisionThresholds:
    """Configurable thresholds for different photo types"""
    # Base thresholds - made more lenient
    blur_delete: float = 0.2  # Lowered from 0.3
    blur_review: float = 0.4  # Lowered from 0.5
    exposure_delete: float = 0.3
    exposure_review: float = 0.5
    composition_delete: float = 0.2
    composition_review: float = 0.4
    overall_delete: float = 0.25  # Lowered from 0.3
    overall_review: float = 0.4  # Lowered from 0.5

    # Confidence thresholds - made more conservative
    delete_confidence: float = 0.8  # Raised from 0.7
    review_confidence: float = 0.5  # Raised from 0.4

    # Style-specific adjustments
    shallow_dof_adjustment: float = 0.2  # Be more lenient with blur for shallow DOF
    portrait_eye_threshold: float = 0.4  # Lowered from 0.6 - more lenient for portraits
    landscape_overall_threshold: float = 0.6  # Higher standard for landscapes

@dataclass 
class PhotoStyle:
    """Detected photography style preferences"""
    uses_shallow_dof: bool = False
    average_aperture: float = 2.8
    prefers_high_contrast: bool = False
    prefers_dark_mood: bool = False
    common_subjects: List[str] = None
    
    def __post_init__(self):
        if self.common_subjects is None:
            self.common_subjects = []

class AdaptiveDecisionEngine:
    """
    Decision engine that adapts to user's photography style
    and learns from feedback
    """
    
    def __init__(self, 
                 config_dir: Optional[Path] = None,
                 learning_enabled: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir or Path.home() / ".config" / "photo_culler"
        
        # Try to create config directory with error handling
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to a temp directory in the current working directory
            fallback_dir = Path.cwd() / ".photo_culler_config"
            self.logger.warning(f"Permission denied for {self.config_dir}, using fallback: {fallback_dir}")
            self.config_dir = fallback_dir
            try:
                self.config_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # If we can't create any config directory, disable learning
                self.logger.warning(f"Cannot create config directory, disabling learning: {e}")
                self.config_dir = None
        
        self.learning_enabled = learning_enabled and (self.config_dir is not None)
        
        # Load or initialize configurations
        self.thresholds = self._load_thresholds()
        self.style = self._load_style()
        self.feedback_history = self._load_feedback()
        
        # Statistics for adaptive learning
        self.session_stats = {
            'total_processed': 0,
            'decisions': {'Keep': 0, 'Delete': 0, 'Review': 0},
            'average_scores': {'blur': [], 'exposure': [], 'composition': []},
            'detected_subjects': []
        }
    
    def _load_thresholds(self) -> DecisionThresholds:
        """Load saved thresholds or use defaults"""
        if not self.config_dir:
            return DecisionThresholds()
            
        threshold_file = self.config_dir / "thresholds.json"
        
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r') as f:
                    data = json.load(f)
                return DecisionThresholds(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load thresholds: {e}")
        
        return DecisionThresholds()
    
    def _load_style(self) -> PhotoStyle:
        """Load detected photography style"""
        if not self.config_dir:
            return PhotoStyle()
            
        style_file = self.config_dir / "photo_style.json"
        
        if style_file.exists():
            try:
                with open(style_file, 'r') as f:
                    data = json.load(f)
                return PhotoStyle(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load style: {e}")
        
        return PhotoStyle()
    
    def _load_feedback(self) -> List[Dict]:
        """Load user feedback history"""
        if not self.config_dir:
            return []
            
        feedback_file = self.config_dir / "feedback.json"
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load feedback: {e}")
        
        return []
    
    def _save_configs(self):
        """Save current configurations"""
        if not self.config_dir:
            self.logger.debug("No config directory available, skipping save")
            return
            
        try:
            # Save thresholds
            with open(self.config_dir / "thresholds.json", 'w') as f:
                json.dump(asdict(self.thresholds), f, indent=2)
            
            # Save style
            with open(self.config_dir / "photo_style.json", 'w') as f:
                json.dump(asdict(self.style), f, indent=2)
            
            # Save feedback
            with open(self.config_dir / "feedback.json", 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save configurations: {e}")
    
    def decide(self, filepath: Path, metrics: ImageMetrics, 
               processing_ms: int, 
               focus_analysis: Optional[Dict] = None) -> CullResult:
        """
        Make culling decision with style-aware adjustments
        """
        
        # Update session statistics
        self.session_stats['total_processed'] += 1
        self.session_stats['average_scores']['blur'].append(metrics.blur_score)
        self.session_stats['average_scores']['exposure'].append(metrics.exposure_score)
        self.session_stats['average_scores']['composition'].append(metrics.composition_score)
        
        # Get adjusted thresholds based on detected style
        adjusted_thresholds = self._adjust_thresholds_for_context(
            metrics, focus_analysis
        )
        
        # Evaluate against thresholds
        issues = []
        severity_scores = []
        
        # Blur evaluation (context-aware)
        blur_eval = self._evaluate_blur(
            metrics.blur_score, 
            adjusted_thresholds,
            focus_analysis
        )
        if blur_eval['issue']:
            issues.append(blur_eval['issue'])
            severity_scores.append(blur_eval['severity'])
        
        # Exposure evaluation
        if metrics.exposure_score < adjusted_thresholds['exposure_delete']:
            issues.append("poor exposure")
            severity = 1.0 - metrics.exposure_score
            severity_scores.append(severity * 0.8)  # Slightly less critical
        elif metrics.exposure_score < adjusted_thresholds['exposure_review']:
            issues.append("exposure needs review")
            severity = 1.0 - metrics.exposure_score
            severity_scores.append(severity * 0.5)
        
        # Composition evaluation
        if metrics.composition_score < adjusted_thresholds['composition_delete']:
            issues.append("poor composition")
            severity = 1.0 - metrics.composition_score
            severity_scores.append(severity * 0.6)
        elif metrics.composition_score < adjusted_thresholds['composition_review']:
            issues.append("composition could be better")
            severity = 1.0 - metrics.composition_score  
            severity_scores.append(severity * 0.3)
        
        # Overall quality check
        if metrics.overall_quality < adjusted_thresholds['overall_delete']:
            issues.append("low overall quality")
            severity = 1.0 - metrics.overall_quality
            severity_scores.append(severity * 0.7)
        
        # Calculate confidence
        if severity_scores:
            confidence = max(severity_scores)
        else:
            confidence = 1.0 - max(
                1.0 - metrics.blur_score,
                1.0 - metrics.exposure_score,
                1.0 - metrics.composition_score,
                1.0 - metrics.overall_quality
            )
        
        # Make decision
        if confidence >= self.thresholds.delete_confidence and len(issues) > 0:
            decision = "Delete"
        elif confidence >= self.thresholds.review_confidence or len(issues) >= 2:
            decision = "Review"
        else:
            decision = "Keep"
            confidence = 1.0 - confidence  # Invert for keep confidence
        
        # Add context to issues if shallow DOF detected
        if focus_analysis and focus_analysis.get('is_shallow_dof'):
            if "blurry" not in ' '.join(issues):
                issues.append("(shallow DOF detected)")
        
        # Update statistics
        self.session_stats['decisions'][decision] += 1
        
        # Store subject type if available
        if focus_analysis:
            subject = focus_analysis.get('subject_type', 'unknown')
            self.session_stats['detected_subjects'].append(subject)
        
        result = CullResult(
            filepath=filepath,
            decision=decision,
            confidence=confidence,
            metrics=metrics,
            issues=issues,
            processing_ms=processing_ms,
            mode=metrics.processing_mode
        )
        
        # Learn from high-confidence decisions if enabled
        if self.learning_enabled and confidence > 0.8:
            self._update_learning(result, metrics, focus_analysis)
        
        return result
    
    def _adjust_thresholds_for_context(self, 
                                       metrics: ImageMetrics,
                                       focus_analysis: Optional[Dict]) -> Dict:
        """Adjust thresholds based on detected context"""
        
        thresholds = {
            'blur_delete': self.thresholds.blur_delete,
            'blur_review': self.thresholds.blur_review,
            'exposure_delete': self.thresholds.exposure_delete,
            'exposure_review': self.thresholds.exposure_review,
            'composition_delete': self.thresholds.composition_delete,
            'composition_review': self.thresholds.composition_review,
            'overall_delete': self.thresholds.overall_delete,
            'overall_review': self.thresholds.overall_review
        }
        
        if not focus_analysis:
            return thresholds
        
        # Adjust for shallow DOF style
        if focus_analysis.get('is_shallow_dof') or self.style.uses_shallow_dof:
            # Be more lenient with blur thresholds
            thresholds['blur_delete'] *= (1 - self.thresholds.shallow_dof_adjustment)
            thresholds['blur_review'] *= (1 - self.thresholds.shallow_dof_adjustment * 0.5)
        
        # Adjust for subject type
        subject_type = focus_analysis.get('subject_type', 'unknown')
        
        if subject_type == 'portrait':
            # Stricter requirements for eye sharpness
            if focus_analysis.get('subject_sharpness', 1.0) < self.thresholds.portrait_eye_threshold:
                thresholds['blur_delete'] = self.thresholds.portrait_eye_threshold
        
        elif subject_type == 'landscape':
            # Higher overall standards
            thresholds['overall_delete'] = self.thresholds.landscape_overall_threshold
            thresholds['blur_review'] = 0.6  # Expect more overall sharpness
        
        elif subject_type == 'macro':
            # Very strict on subject sharpness, lenient on background
            thresholds['blur_delete'] = 0.4  # Only subject needs to be sharp
        
        # Adjust for dark/moody style if detected
        if self.style.prefers_dark_mood:
            thresholds['exposure_delete'] *= 0.7  # More tolerant of underexposure
        
        return thresholds
    
    def _evaluate_blur(self, blur_score: float, 
                      thresholds: Dict,
                      focus_analysis: Optional[Dict]) -> Dict:
        """Context-aware blur evaluation"""
        
        result = {'issue': None, 'severity': 0}
        
        # If we have focus analysis, use subject sharpness instead
        if focus_analysis:
            subject_sharpness = focus_analysis.get('subject_sharpness', blur_score)
            
            if subject_sharpness < thresholds['blur_delete']:
                result['issue'] = "subject not sharp"
                result['severity'] = 1.0 - subject_sharpness
                
                # Add context
                subject_type = focus_analysis.get('subject_type', 'unknown')
                if subject_type == 'portrait':
                    result['issue'] = "eyes/face not sharp"
                elif subject_type == 'macro':
                    result['issue'] = "main subject soft"
                    
            elif subject_sharpness < thresholds['blur_review']:
                result['issue'] = "slight softness"
                result['severity'] = (1.0 - subject_sharpness) * 0.6
        
        else:
            # Fallback to simple blur evaluation
            if blur_score < thresholds['blur_delete']:
                result['issue'] = "blurry"
                result['severity'] = 1.0 - blur_score
            elif blur_score < thresholds['blur_review']:
                result['issue'] = "slightly soft"
                result['severity'] = (1.0 - blur_score) * 0.6
        
        return result
    
    def _update_learning(self, result: CullResult, 
                        metrics: ImageMetrics,
                        focus_analysis: Optional[Dict]):
        """Update style preferences based on decisions"""
        
        if not self.learning_enabled:
            return
        
        # Detect shallow DOF preference
        if focus_analysis and focus_analysis.get('is_shallow_dof'):
            # If we're keeping photos with shallow DOF, user likes this style
            if result.decision == "Keep":
                self.style.uses_shallow_dof = True
        
        # Update common subjects
        if focus_analysis:
            subject = focus_analysis.get('subject_type', 'unknown')
            if subject != 'unknown' and subject not in self.style.common_subjects:
                self.style.common_subjects.append(subject)
                # Keep only top 5 most common
                if len(self.style.common_subjects) > 5:
                    self.style.common_subjects = self.style.common_subjects[-5:]
        
        # Detect exposure preferences
        if result.decision == "Keep":
            if metrics.exposure_score < 0.5:
                # User keeps underexposed photos
                self.style.prefers_dark_mood = True
            elif metrics.exposure_score > 0.8 and metrics.composition_score > 0.7:
                # User prefers well-exposed, high contrast
                self.style.prefers_high_contrast = True
    
    def record_feedback(self, filepath: Path, 
                       original_decision: str,
                       user_decision: str,
                       metrics: ImageMetrics):
        """Record user feedback when they disagree with decisions"""
        
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'file': str(filepath),
            'original': original_decision,
            'corrected': user_decision,
            'metrics': {
                'blur': metrics.blur_score,
                'exposure': metrics.exposure_score,
                'composition': metrics.composition_score,
                'overall': metrics.overall_quality
            }
        }
        
        self.feedback_history.append(feedback)
        
        # Learn from feedback
        if len(self.feedback_history) >= 10:
            self._adjust_from_feedback()
    
    def _adjust_from_feedback(self):
        """Adjust thresholds based on user feedback patterns"""
        
        if not self.learning_enabled or len(self.feedback_history) < 10:
            return
        
        # Analyze recent feedback
        recent = self.feedback_history[-20:]  # Last 20 corrections
        
        # Check if we're too aggressive with deletions
        false_deletes = [f for f in recent 
                         if f['original'] == 'Delete' and f['corrected'] == 'Keep']
        
        if len(false_deletes) > 5:
            # We're deleting too many, raise thresholds
            self.thresholds.blur_delete *= 0.9
            self.thresholds.delete_confidence *= 1.1
            self.logger.info("Adjusted thresholds: being less aggressive with deletions")
        
        # Check if we're missing bad photos
        false_keeps = [f for f in recent 
                      if f['original'] == 'Keep' and f['corrected'] == 'Delete']
        
        if len(false_keeps) > 5:
            # We're keeping too many bad photos
            self.thresholds.blur_delete *= 1.1
            self.thresholds.delete_confidence *= 0.9
            self.logger.info("Adjusted thresholds: being more aggressive with deletions")
        
        self._save_configs()
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        
        summary = {
            'total_processed': self.session_stats['total_processed'],
            'decisions': self.session_stats['decisions'],
            'detected_style': {
                'uses_shallow_dof': self.style.uses_shallow_dof,
                'common_subjects': self.style.common_subjects[:3] if self.style.common_subjects else [],
                'prefers_dark_mood': self.style.prefers_dark_mood
            }
        }
        
        # Calculate average scores
        for metric in ['blur', 'exposure', 'composition']:
            scores = self.session_stats['average_scores'][metric]
            if scores:
                summary[f'avg_{metric}'] = np.mean(scores)
        
        return summary
    
    def save_session(self):
        """Save session data and learned preferences"""
        self._save_configs()
        
        # Log summary
        summary = self.get_session_summary()
        self.logger.info(f"Session complete: {summary['total_processed']} photos processed")
        self.logger.info(f"Decisions: {summary['decisions']}")
        
        if summary['detected_style']['uses_shallow_dof']:
            self.logger.info("Detected preference for shallow depth of field")
        
        if summary['detected_style']['common_subjects']:
            self.logger.info(f"Common subjects: {', '.join(summary['detected_style']['common_subjects'])}")
