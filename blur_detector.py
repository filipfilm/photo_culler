"""
Hybrid blur detection combining vision model with computer vision techniques
"""
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class HybridBlurDetector:
    """Combines vision model with CV techniques for reliable blur detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _image_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        # Convert PIL to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and then to OpenCV format (BGR)
        cv_image = np.array(pil_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image
    
    def _laplacian_variance(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance - higher values indicate sharper images"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    def _sobel_variance(self, image: np.ndarray) -> float:
        """Calculate Sobel edge variance - higher values indicate sharper edges"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return float(sobel_magnitude.var())
    
    def _brenner_focus(self, image: np.ndarray) -> float:
        """Brenner focus measure - sum of squared differences"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal and vertical differences
        diff_x = np.diff(gray.astype(np.float64), axis=1)
        diff_y = np.diff(gray.astype(np.float64), axis=0)
        
        # Sum of squared differences
        focus_measure = np.sum(diff_x**2) + np.sum(diff_y**2)
        return float(focus_measure)
    
    def _detect_edges_density(self, image: np.ndarray) -> float:
        """Detect edge density - more edges indicate sharper image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        return float(edge_density)
    
    def detect_cv_blur(self, image: Image.Image) -> Dict[str, float]:
        """Detect blur using computer vision techniques"""
        try:
            cv_image = self._image_to_cv2(image)
            
            # Calculate multiple blur metrics
            laplacian_var = self._laplacian_variance(cv_image)
            sobel_var = self._sobel_variance(cv_image)
            brenner_focus = self._brenner_focus(cv_image)
            edge_density = self._detect_edges_density(cv_image)
            
            # Normalize scores to 0-1 range (approximate thresholds based on testing)
            normalized_scores = {
                'laplacian_sharpness': min(1.0, max(0.0, laplacian_var / 1000.0)),  # Higher = sharper
                'sobel_sharpness': min(1.0, max(0.0, sobel_var / 10000.0)),  # Higher = sharper
                'brenner_sharpness': min(1.0, max(0.0, brenner_focus / 1000000.0)),  # Higher = sharper
                'edge_density': min(1.0, max(0.0, edge_density * 10.0)),  # Higher = sharper
            }
            
            # Calculate overall CV sharpness score
            cv_sharpness = np.mean(list(normalized_scores.values()))
            
            return {
                'cv_sharpness_score': float(cv_sharpness),
                'laplacian_var': laplacian_var,
                'sobel_var': sobel_var,  
                'brenner_focus': brenner_focus,
                'edge_density': edge_density,
                **normalized_scores
            }
            
        except Exception as e:
            self.logger.warning(f"CV blur detection failed: {e}")
            return {
                'cv_sharpness_score': 0.5,  # Neutral score if CV fails
                'laplacian_var': 0,
                'sobel_var': 0,
                'brenner_focus': 0,
                'edge_density': 0,
                'laplacian_sharpness': 0.5,
                'sobel_sharpness': 0.5,
                'brenner_sharpness': 0.5
            }
    
    def combine_vision_and_cv(self, vision_blur_score: float, cv_metrics: Dict[str, float]) -> float:
        """Combine vision model score with CV metrics for more reliable blur detection"""
        
        cv_score = cv_metrics['cv_sharpness_score']
        
        # If both agree (both high or both low), trust them
        if (vision_blur_score > 0.6 and cv_score > 0.6) or (vision_blur_score < 0.4 and cv_score < 0.4):
            # They agree - use weighted average favoring vision model
            combined_score = 0.7 * vision_blur_score + 0.3 * cv_score
        else:
            # They disagree - be conservative and use the lower score
            # This helps catch blur that one method missed
            combined_score = min(vision_blur_score, cv_score)
            
            # Log the disagreement for debugging
            self.logger.debug(f"Vision/CV disagreement: vision={vision_blur_score:.3f}, cv={cv_score:.3f}, using={combined_score:.3f}")
        
        return float(combined_score)

def install_opencv_dependencies():
    """Install OpenCV if not available"""
    try:
        import cv2
        return True
    except ImportError:
        print("⚠️  OpenCV not available for hybrid blur detection")
        print("   Install with: pip install opencv-python")
        print("   Falling back to vision-only blur detection")
        return False