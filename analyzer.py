import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Dict
import time
import logging
from pathlib import Path
from models import ProcessingMode, ImageMetrics

# Optional imports for vision models
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from ollama_vision import OllamaVisionAnalyzer, ImprovedOllamaVisionAnalyzer
    OLLAMA_AVAILABLE = True
    IMPROVED_OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from ollama_vision import OllamaVisionAnalyzer
        OLLAMA_AVAILABLE = True
        IMPROVED_OLLAMA_AVAILABLE = False
    except ImportError:
        OLLAMA_AVAILABLE = False
        IMPROVED_OLLAMA_AVAILABLE = False

try:
    from subject_detector import SubjectDetector
    SUBJECT_DETECTOR_AVAILABLE = True
except ImportError:
    SUBJECT_DETECTOR_AVAILABLE = False

try:
    from enhanced_focus_analyzer import EnhancedFocusAnalyzer
    ENHANCED_FOCUS_AVAILABLE = True
except ImportError:
    ENHANCED_FOCUS_AVAILABLE = False

VISION_AVAILABLE = CLIP_AVAILABLE or OLLAMA_AVAILABLE


class TechnicalAnalyzer:
    """Fast traditional CV analyzer with enhanced technical QC"""
    def __init__(self):
        self.blur_threshold = 100
        self.highlight_clip_percent = 0.01
        self.shadow_clip_percent = 0.01

        # Initialize enhanced focus analyzer if available
        self.enhanced_focus = None
        if ENHANCED_FOCUS_AVAILABLE:
            try:
                self.enhanced_focus = EnhancedFocusAnalyzer()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to initialize enhanced focus analyzer: {e}")

        # Disable modern analyzers temporarily due to array dimension issues
        # TODO: Re-enable after fixing array handling
        self.advanced_focus = None
        self.smart_exposure = None  
        self.smart_composition = None
        
        logging.getLogger(__name__).info("Using traditional CV analysis (modern analyzers disabled)")
        
        # Keep this code for future re-enabling:
        # try:
        #     from advanced_focus_detector import AdvancedFocusDetector
        #     self.advanced_focus = AdvancedFocusDetector()
        # except ImportError:
        #     self.advanced_focus = None

    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Return metrics using modern advanced analysis"""
        # Resize for consistent processing
        analysis_image = image.copy()
        analysis_image.thumbnail((800, 800), Image.Resampling.LANCZOS)

        # Use modern analyzers if available
        focus_score = 0.5
        exposure_score = 0.5
        composition_score = 0.5
        
        advanced_analysis = {}
        
        # Advanced focus detection
        if self.advanced_focus:
            try:
                focus_analysis = self.advanced_focus.analyze_focus(analysis_image)
                focus_score = focus_analysis['focus_score']
                advanced_analysis['focus'] = focus_analysis
            except Exception as e:
                logging.getLogger(__name__).warning(f"Advanced focus analysis failed: {e}")
                # Fallback to traditional
                rgb = np.array(analysis_image.convert('RGB'))
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                focus_score, _ = self._blur_score(analysis_image, gray)
        
        # Smart exposure analysis
        if self.smart_exposure:
            try:
                exposure_analysis = self.smart_exposure.analyze_exposure(analysis_image)
                exposure_score = exposure_analysis['exposure_score']
                advanced_analysis['exposure'] = exposure_analysis
            except Exception as e:
                logging.getLogger(__name__).warning(f"Smart exposure analysis failed: {e}")
                # Fallback to traditional
                rgb = np.array(analysis_image.convert('RGB'))
                exposure_score = self._exposure_score(rgb)
        
        # Intelligent composition analysis
        if self.smart_composition:
            try:
                composition_analysis = self.smart_composition.analyze_composition(analysis_image)
                composition_score = composition_analysis['composition_score']
                advanced_analysis['composition'] = composition_analysis
            except Exception as e:
                logging.getLogger(__name__).warning(f"Intelligent composition analysis failed: {e}")
                # Fallback to traditional
                rgb = np.array(analysis_image.convert('RGB'))
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                composition_score = self._composition_score(gray)

        # If no modern analyzers available, fall back to traditional methods
        if not any([self.advanced_focus, self.smart_exposure, self.smart_composition]):
            rgb = np.array(analysis_image.convert('RGB'))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            
            focus_score, enhanced_focus_data = self._blur_score(analysis_image, gray)
            exposure_score = self._exposure_score(rgb)
            composition_score = self._composition_score(gray)
            
            if enhanced_focus_data:
                advanced_analysis['enhanced_focus'] = enhanced_focus_data

        # Weighted overall with emphasis on focus (most important)
        overall = focus_score * 0.5 + exposure_score * 0.3 + composition_score * 0.2

        return ImageMetrics(
            blur_score=focus_score,
            exposure_score=exposure_score,
            composition_score=composition_score,
            overall_quality=overall,
            processing_mode=ProcessingMode.FAST,
            keywords=None,  # Traditional CV doesn't generate keywords
            description=None,  # Traditional CV doesn't generate descriptions
            enhanced_focus=advanced_analysis.get('enhanced_focus'),
            technical_issues=advanced_analysis  # Store all advanced analysis
        )
    
    def _blur_score(self, image: Image.Image, gray: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """0=blurry, 1=sharp using multiple focus metrics"""

        # Try enhanced focus analyzer first if available
        if self.enhanced_focus:
            try:
                focus_result = self.enhanced_focus.analyze_focus(image)
                enhanced_score = focus_result.get("focus_score", 0.5)
                # Use enhanced score as primary, but blend with traditional for robustness
                traditional_score = self._traditional_blur_score(gray)
                combined_score = (enhanced_score * 0.7 + traditional_score * 0.3)
                return combined_score, focus_result
            except Exception as e:
                logging.getLogger(__name__).warning(f"Enhanced focus analysis failed, falling back to traditional: {e}")

        # Fall back to traditional method
        traditional_score = self._traditional_blur_score(gray)
        return traditional_score, None

    def _traditional_blur_score(self, gray: np.ndarray) -> float:
        """Traditional blur scoring method"""
        # Multiple focus detection methods for better accuracy

        # 1. Laplacian variance (edge-based)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_score = np.clip(lap_var / 500, 0, 1)  # Increased threshold for better sensitivity

        # 2. Sobel gradient magnitude (directional edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.var(np.sqrt(sobelx**2 + sobely**2))
        sobel_score = np.clip(sobel_var / 1000, 0, 1)

        # 3. Normalized Variance of Laplacian (more robust to different image types)
        norm_lap = cv2.Laplacian(gray, cv2.CV_64F)
        mean_val = gray.mean()
        if mean_val > 0:
            nvol = norm_lap.var() / mean_val
            nvol_score = np.clip(nvol / 10, 0, 1)
        else:
            nvol_score = 0

        # 4. High-frequency content analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Focus on high-frequency components (sharp details)
        h, w = gray.shape
        center_h, center_w = h//2, w//2
        high_freq_mask = np.zeros((h, w))
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
        high_freq_mask = 1 - high_freq_mask  # Invert to get high frequencies

        high_freq_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude)
        if total_energy > 0:
            hf_ratio = high_freq_energy / total_energy
            hf_score = np.clip(hf_ratio * 5, 0, 1)  # Scale appropriately
        else:
            hf_score = 0

        # Combine scores with weights (Laplacian and Sobel are most reliable)
        combined_score = (lap_score * 0.4 + sobel_score * 0.3 + nvol_score * 0.2 + hf_score * 0.1)

        return combined_score
    
    def _exposure_score(self, rgb: np.ndarray) -> float:
        """0=clipped, 1=well exposed"""
        pixels = rgb.size
        blown = np.sum(rgb > 250) / pixels
        blocked = np.sum(rgb < 5) / pixels
        
        score = 1.0
        if blown > self.highlight_clip_percent:
            score -= min(0.5, blown * 50)
        if blocked > self.shadow_clip_percent:
            score -= min(0.5, blocked * 50)
            
        return max(0, score)
    
    def _composition_score(self, gray: np.ndarray) -> float:
        """0=boring, 1=interesting"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = np.clip(edge_density * 10, 0, 1)
        
        contrast = gray.std() / 60
        contrast_score = np.clip(contrast, 0, 1)
        
        return (edge_score + contrast_score) / 2


class VisionAnalyzer:
    """Accurate vision model analyzer using CLIP or Ollama"""
    def __init__(self, 
                 device: Optional[str] = None, 
                 model_size: str = "ViT-B/32",
                 use_ollama: bool = False,
                 ollama_model: str = "llava:13b",  # Changed default
                 ollama_host: str = "http://localhost:11434"):
        
        self.logger = logging.getLogger(__name__)
        self.use_ollama = use_ollama
        
        # Add subject detector
        if SUBJECT_DETECTOR_AVAILABLE:
            self.subject_detector = SubjectDetector()
        else:
            self.subject_detector = None
        
        if not VISION_AVAILABLE:
            raise ImportError("No vision model available. Install torch+clip-by-openai OR setup Ollama")
        
        if use_ollama or not CLIP_AVAILABLE:
            # Try Ollama first if requested or if CLIP unavailable
            if OLLAMA_AVAILABLE:
                try:
                    # Use improved analyzer if available, fall back to basic one
                    if IMPROVED_OLLAMA_AVAILABLE:
                        self.analyzer = ImprovedOllamaVisionAnalyzer(model=ollama_model, host=ollama_host)
                        self.logger.info(f"Using Improved Ollama vision model: {ollama_model} at {ollama_host}")
                    else:
                        self.analyzer = OllamaVisionAnalyzer(model=ollama_model, host=ollama_host)
                        self.logger.info(f"Using Ollama vision model: {ollama_model} at {ollama_host}")
                    self.use_ollama = True
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Ollama: {e}")
                    if use_ollama:  # User specifically requested Ollama
                        raise
        
        # Fall back to CLIP if available
        if CLIP_AVAILABLE and not use_ollama:
            # Auto-detect device
            if device:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            self.logger.info(f"Using CLIP on device: {self.device}")
            
            # Load CLIP
            try:
                self.model, self.preprocess = clip.load(model_size, device=self.device)
                self.model.eval()
                self.use_ollama = False
            except Exception as e:
                self.logger.error(f"Failed to load CLIP: {e}")
                raise
        else:
            raise ImportError("No vision model could be initialized")
        
        # Only setup CLIP prompts if using CLIP
        if not self.use_ollama:
            # Quality assessment prompts - improved for better focus detection
            self.quality_prompts = {
                'sharp': [
                    "a tack-sharp photograph with crisp details and perfect focus", 
                    "a blurry photograph with soft focus and motion blur"
                ],
                'exposure': [
                    "a perfectly exposed photograph with balanced lighting and rich detail", 
                    "an overexposed or underexposed photograph with clipped highlights or blocked shadows"
                ],
                'composition': [
                    "a beautifully composed professional photograph with excellent framing", 
                    "a poorly composed snapshot with bad framing and distracting elements"
                ],
                'quality': [
                    "a high quality professional photograph that should definitely be kept", 
                    "a low quality amateur photograph that should be rejected and deleted"
                ]
            }
            
            self._encode_prompts()
        
    def _encode_prompts(self):
        """Pre-encode all text prompts"""
        self.encoded_prompts = {}
        
        with torch.no_grad():
            for category, (pos, neg) in self.quality_prompts.items():
                pos_tokens = clip.tokenize([pos]).to(self.device)
                neg_tokens = clip.tokenize([neg]).to(self.device)
                
                self.encoded_prompts[category] = {
                    'positive': self.model.encode_text(pos_tokens),
                    'negative': self.model.encode_text(neg_tokens)
                }
    
    def analyze_batch(self, images: List[Image.Image]) -> List[ImageMetrics]:
        """Process multiple images at once for efficiency"""
        if self.use_ollama:
            # Use Ollama analyzer
            return self.analyzer.analyze_batch(images)
        
        # Use CLIP analyzer
        # Preprocess all images
        image_tensors = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        metrics_list = []
        
        with torch.no_grad():
            # Encode all images at once
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Score each image
            for img_feat in image_features:
                img_feat = img_feat.unsqueeze(0)
                
                scores = {}
                for category, prompts in self.encoded_prompts.items():
                    pos_sim = torch.cosine_similarity(img_feat, prompts['positive'])
                    neg_sim = torch.cosine_similarity(img_feat, prompts['negative'])
                    
                    # Convert to 0-1 score (higher is better)
                    score = (pos_sim - neg_sim + 1) / 2
                    scores[category] = float(score.cpu())
                
                metrics = ImageMetrics(
                    blur_score=scores['sharp'],
                    exposure_score=scores['exposure'],
                    composition_score=scores['composition'],
                    overall_quality=scores['quality'],
                    processing_mode=ProcessingMode.ACCURATE,
                    keywords=None,  # CLIP doesn't generate keywords in this implementation
                    description=None  # CLIP doesn't generate descriptions in this implementation
                )
                metrics_list.append(metrics)
                
        return metrics_list
    
    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Single image analysis with subject detection"""
        
        # First check subject focus with CV
        subject_focus = 0.5  # Default fallback
        if self.subject_detector:
            try:
                subject_focus = self.subject_detector.check_subject_focus(image)
            except Exception as e:
                self.logger.warning(f"Subject detection failed: {e}")
        
        # Get base metrics from vision model
        if self.use_ollama:
            metrics = self.analyzer.analyze(image)
        else:
            metrics = self.analyze_batch([image])[0]
        
        # Adjust blur score based on subject detection
        # Weight: 70% vision model, 30% CV subject detection
        metrics.blur_score = (metrics.blur_score * 0.7 + subject_focus * 0.3)
        
        # If portrait with bad eye focus, override to low score
        if self.subject_detector:
            try:
                portrait_info = self.subject_detector.detect_portrait_subject(image)
                if portrait_info["is_portrait"] and portrait_info.get("eye_sharpness", 1.0) < 0.3:
                    metrics.blur_score = min(metrics.blur_score, 0.3)
                    self.logger.debug(f"Portrait with soft eyes detected, capping blur score at 0.3")
            except Exception as e:
                self.logger.warning(f"Portrait detection failed: {e}")
        
        return metrics


class HybridAnalyzer:
    """Combines both analyzers with smart routing"""
    def __init__(self, mode: ProcessingMode = ProcessingMode.ACCURATE, 
                 force_cpu: bool = False,
                 use_ollama: bool = False,
                 ollama_model: str = "llava:13b",
                 ollama_host: str = "http://localhost:11434"):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Always initialize fast analyzer
        self.fast_analyzer = TechnicalAnalyzer()
        
        # Initialize vision analyzer if needed
        self.vision_analyzer = None
        if mode == ProcessingMode.ACCURATE:
            try:
                # Try Ollama first if available and no CLIP, or if explicitly requested
                if use_ollama or not CLIP_AVAILABLE:
                    device = "cpu" if force_cpu else None
                    self.vision_analyzer = VisionAnalyzer(
                        device=device, 
                        use_ollama=True, 
                        ollama_model=ollama_model,
                        ollama_host=ollama_host
                    )
                    self.logger.info(f"Vision model loaded successfully (Ollama: {ollama_model})")
                else:
                    device = "cpu" if force_cpu else None
                    self.vision_analyzer = VisionAnalyzer(device=device, use_ollama=False)
                    self.logger.info("Vision model loaded successfully (CLIP)")
            except Exception as e:
                self.logger.warning(f"Failed to load vision model, falling back to fast mode: {e}")
                self.mode = ProcessingMode.FAST
                
    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Route to appropriate analyzer"""
        if self.mode == ProcessingMode.FAST or self.vision_analyzer is None:
            return self.fast_analyzer.analyze(image)
        else:
            return self.vision_analyzer.analyze(image)
            
    def analyze_batch(self, images: List[Image.Image]) -> List[ImageMetrics]:
        """Batch processing for efficiency"""
        if self.mode == ProcessingMode.FAST or self.vision_analyzer is None:
            return [self.fast_analyzer.analyze(img) for img in images]
        else:
            return self.vision_analyzer.analyze_batch(images)
