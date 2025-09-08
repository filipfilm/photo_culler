"""
Advanced Focus Detection - Multiple modern methods for perfect focus assessment
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import feature, filters, measure


class AdvancedFocusDetector:
    """State-of-the-art focus detection using multiple modern methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_focus(self, image: Image.Image) -> Dict:
        """Comprehensive focus analysis using multiple methods"""
        
        # Convert to array for processing
        rgb = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # Multiple focus metrics
        focus_metrics = {}
        
        # 1. Brenner Gradient - excellent for general sharpness
        focus_metrics['brenner'] = self._brenner_gradient(gray)
        
        # 2. Tenengrad Variance - great for edge-rich content
        focus_metrics['tenengrad'] = self._tenengrad_variance(gray)
        
        # 3. Laplacian Variance - classic but reliable
        focus_metrics['laplacian_var'] = self._laplacian_variance(gray)
        
        # 4. Wavelet-based focus measure - modern approach
        focus_metrics['wavelet'] = self._wavelet_focus(gray)
        
        # 5. Local Binary Pattern variance - texture-based
        focus_metrics['lbp_variance'] = self._lbp_focus(gray)
        
        # 6. Spectral focus measure - frequency domain
        focus_metrics['spectral'] = self._spectral_focus(gray)
        
        # 7. Edge density and sharpness
        focus_metrics['edge_density'] = self._edge_density_focus(gray)
        
        # 8. Regional focus analysis
        regional_analysis = self._regional_focus_analysis(gray)
        
        # Combine metrics with weights based on reliability
        weights = {
            'brenner': 0.20,
            'tenengrad': 0.20,
            'laplacian_var': 0.15,
            'wavelet': 0.15,
            'lbp_variance': 0.10,
            'spectral': 0.10,
            'edge_density': 0.10
        }
        
        # Normalize each metric to 0-1 scale
        normalized_metrics = self._normalize_metrics(focus_metrics)
        
        # Calculate weighted combined score
        combined_score = sum(
            normalized_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        # Analyze focus distribution across image
        focus_map = self._create_focus_map(gray)
        
        return {
            'focus_score': combined_score,
            'individual_metrics': normalized_metrics,
            'raw_metrics': focus_metrics,
            'regional_analysis': regional_analysis,
            'focus_map_stats': self._analyze_focus_map(focus_map),
            'is_sharp': combined_score > 0.7,
            'focus_quality': self._classify_focus_quality(combined_score),
            'recommendations': self._get_focus_recommendations(combined_score, regional_analysis)
        }
    
    def _brenner_gradient(self, gray: np.ndarray) -> float:
        """Brenner gradient - excellent for general sharpness"""
        try:
            # Calculate gradients
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Brenner focus measure
            brenner = np.sum(gx**2 + gy**2)
            return float(brenner)
        except:
            return 0.0
    
    def _tenengrad_variance(self, gray: np.ndarray) -> float:
        """Tenengrad variance - great for edge-rich content"""
        try:
            # Sobel gradients
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            # Tenengrad measure
            tenengrad = np.sum(gradient_mag**2)
            return float(tenengrad)
        except:
            return 0.0
    
    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Laplacian variance - classic reliable method"""
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except:
            return 0.0
    
    def _wavelet_focus(self, gray: np.ndarray) -> float:
        """Wavelet-based focus measure using high-frequency components"""
        try:
            # Simple wavelet-like analysis using difference of Gaussians
            blur1 = cv2.GaussianBlur(gray, (3, 3), 0.5)
            blur2 = cv2.GaussianBlur(gray, (7, 7), 1.5)
            
            # High frequency components
            high_freq = np.abs(blur1 - blur2)
            
            # Focus measure based on high frequency energy
            focus_measure = np.sum(high_freq**2)
            return float(focus_measure)
        except:
            return 0.0
    
    def _lbp_focus(self, gray: np.ndarray) -> float:
        """Local Binary Pattern variance for texture-based focus"""
        try:
            # Simple LBP-like texture analysis
            # Calculate local variance as texture measure
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray, -1, kernel)
            local_variance = cv2.filter2D((gray - local_mean)**2, -1, kernel)
            
            # Variance of local variances as focus measure
            focus_measure = np.var(local_variance)
            return float(focus_measure)
        except:
            return 0.0
    
    def _spectral_focus(self, gray: np.ndarray) -> float:
        """Spectral focus measure using frequency domain analysis"""
        try:
            # FFT-based analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Focus on high frequencies (sharp details)
            h, w = gray.shape
            center_h, center_w = h//2, w//2
            
            # Create high-pass filter
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) * 0.1)**2
            
            # High frequency energy
            high_freq_energy = np.sum(magnitude * mask)
            total_energy = np.sum(magnitude)
            
            if total_energy > 0:
                spectral_ratio = high_freq_energy / total_energy
                return float(spectral_ratio * 1e6)  # Scale for better range
            else:
                return 0.0
        except:
            return 0.0
    
    def _edge_density_focus(self, gray: np.ndarray) -> float:
        """Edge density and sharpness analysis"""
        try:
            # Multi-scale edge detection
            edges_fine = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edges_coarse = cv2.Canny(gray.astype(np.uint8), 100, 200)
            
            # Edge density
            edge_density = np.sum(edges_fine > 0) / edges_fine.size
            
            # Edge sharpness (fine vs coarse ratio)
            fine_edges = np.sum(edges_fine > 0)
            coarse_edges = np.sum(edges_coarse > 0)
            
            if coarse_edges > 0:
                edge_sharpness = fine_edges / coarse_edges
            else:
                edge_sharpness = 0
            
            # Combined measure
            focus_measure = edge_density * edge_sharpness * 1000
            return float(focus_measure)
        except:
            return 0.0
    
    def _regional_focus_analysis(self, gray: np.ndarray) -> Dict:
        """Analyze focus across different regions of the image"""
        try:
            h, w = gray.shape
            
            # Divide image into 9 regions (3x3 grid)
            regions = {}
            region_names = [
                'top_left', 'top_center', 'top_right',
                'middle_left', 'center', 'middle_right', 
                'bottom_left', 'bottom_center', 'bottom_right'
            ]
            
            for i, name in enumerate(region_names):
                row = i // 3
                col = i % 3
                
                y1 = int(row * h / 3)
                y2 = int((row + 1) * h / 3)
                x1 = int(col * w / 3)
                x2 = int((col + 1) * w / 3)
                
                region = gray[y1:y2, x1:x2]
                
                # Quick focus measure for region
                laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
                regions[name] = float(laplacian_var)
            
            # Find sharpest and softest regions
            sharpest_region = max(regions.items(), key=lambda x: x[1])
            softest_region = min(regions.items(), key=lambda x: x[1])
            
            # Calculate focus uniformity
            focus_values = list(regions.values())
            focus_uniformity = 1.0 - (np.std(focus_values) / np.mean(focus_values)) if np.mean(focus_values) > 0 else 0
            
            return {
                'regions': regions,
                'sharpest_region': sharpest_region[0],
                'sharpest_value': sharpest_region[1],
                'softest_region': softest_region[0],
                'softest_value': softest_region[1],
                'focus_uniformity': focus_uniformity,
                'center_focus': regions['center']
            }
        except:
            return {'regions': {}, 'focus_uniformity': 0.5}
    
    def _create_focus_map(self, gray: np.ndarray) -> np.ndarray:
        """Create a focus map showing sharp and soft areas"""
        try:
            # Local focus measure using sliding window
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Local Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            local_variance = cv2.filter2D(laplacian**2, -1, kernel)
            
            return local_variance
        except:
            return np.zeros_like(gray)
    
    def _analyze_focus_map(self, focus_map: np.ndarray) -> Dict:
        """Analyze the focus map for distribution statistics"""
        try:
            if focus_map.size == 0:
                return {'mean_focus': 0, 'focus_std': 0, 'sharp_percentage': 0}
            
            mean_focus = np.mean(focus_map)
            focus_std = np.std(focus_map)
            
            # Percentage of image that's sharp
            threshold = np.percentile(focus_map, 75)  # Top 25% as sharp
            sharp_percentage = np.sum(focus_map > threshold) / focus_map.size
            
            return {
                'mean_focus': float(mean_focus),
                'focus_std': float(focus_std),
                'sharp_percentage': float(sharp_percentage)
            }
        except:
            return {'mean_focus': 0, 'focus_std': 0, 'sharp_percentage': 0}
    
    def _normalize_metrics(self, metrics: Dict) -> Dict:
        """Normalize metrics to 0-1 scale based on empirical thresholds"""
        
        # Empirically determined thresholds for normalization (much more realistic)
        thresholds = {
            'brenner': 1e6,           # Lower threshold - was way too high
            'tenengrad': 1e6,         # Lower threshold - was way too high
            'laplacian_var': 100,     # Much lower - 500 was unrealistic
            'wavelet': 1e4,           # Much lower threshold  
            'lbp_variance': 200,      # Much lower threshold
            'spectral': 20,           # Much lower threshold
            'edge_density': 2         # Much lower threshold
        }
        
        normalized = {}
        for metric, value in metrics.items():
            if metric in thresholds:
                # Much more generous normalization
                normalized_value = value / (value + thresholds[metric])
                normalized[metric] = min(1.0, normalized_value * 3.0)  # Much bigger boost for real photos
            else:
                normalized[metric] = 0.0
                
        return normalized
    
    def _classify_focus_quality(self, score: float) -> str:
        """Classify focus quality based on score (more realistic thresholds)"""
        if score >= 0.75:
            return "Tack Sharp"
        elif score >= 0.55:
            return "Sharp"
        elif score >= 0.35:
            return "Acceptable"
        elif score >= 0.20:
            return "Soft"
        else:
            return "Blurry"
    
    def _get_focus_recommendations(self, score: float, regional: Dict) -> List[str]:
        """Get recommendations based on focus analysis"""
        recommendations = []
        
        if score < 0.15:
            recommendations.append("Consider deletion - very blurry")
        elif score < 0.25:
            recommendations.append("Review for deletion - soft focus")
        elif score > 0.75:
            recommendations.append("Excellent sharpness - portfolio quality")
        
        # Regional analysis recommendations
        if regional.get('focus_uniformity', 0) < 0.3:
            recommendations.append("Uneven focus across frame")
        
        center_focus = regional.get('center_focus', 0)
        if center_focus < score * 0.5:  # Center much softer than overall
            recommendations.append("Center of frame soft - check composition")
        
        return recommendations