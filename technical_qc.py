"""
Advanced Technical Quality Control for Photo Analysis
Detects chromatic aberration, noise, banding, moiré patterns, and other technical issues
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class TechnicalIssues:
    """Container for technical quality assessment results"""
    chromatic_aberration: float  # 0-1, lower is better
    noise_level: float  # 0-1, higher is better (less noise)
    banding_score: float  # 0-1, higher is better (less banding)
    moire_score: float  # 0-1, higher is better (less moiré)
    vignetting_score: float  # 0-1, higher is better (less vignetting)
    dust_spots: int  # Number of detected dust spots
    overall_technical_score: float  # 0-1, higher is better


class TechnicalQC:
    """Advanced technical quality control"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, image: Image.Image) -> TechnicalIssues:
        """Comprehensive technical analysis of image"""
        # Convert to numpy array for processing
        rgb = np.array(image.convert('RGB'))
        
        # Run all technical checks
        chromatic_aberration = self.check_chromatic_aberration(rgb)
        noise_level = self.check_noise_level(rgb)
        banding_score = self.check_banding(rgb)
        moire_score = self.check_moire(rgb)
        vignetting_score = self.check_vignetting(rgb)
        dust_spots = self.detect_dust_spots(rgb)
        
        # Calculate overall technical score
        overall_score = (
            (1 - chromatic_aberration) * 0.25 +
            noise_level * 0.25 +
            banding_score * 0.20 +
            moire_score * 0.15 +
            vignetting_score * 0.10 +
            max(0, 1 - dust_spots / 10) * 0.05  # Penalize dust spots
        )
        
        return TechnicalIssues(
            chromatic_aberration=chromatic_aberration,
            noise_level=noise_level,
            banding_score=banding_score,
            moire_score=moire_score,
            vignetting_score=vignetting_score,
            dust_spots=dust_spots,
            overall_technical_score=overall_score
        )
    
    def check_chromatic_aberration(self, image: np.ndarray) -> float:
        """Detect purple/green fringing (chromatic aberration)"""
        try:
            # Split channels
            b, g, r = cv2.split(image)
            
            # Find edges where CA is most visible
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to include nearby pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_region = cv2.dilate(edges, kernel, iterations=1)
            
            edge_coords = np.where(edge_region > 0)
            if len(edge_coords[0]) == 0:
                return 0.0
            
            # Sample edge pixels for CA detection
            sample_size = min(1000, len(edge_coords[0]))
            indices = np.random.choice(len(edge_coords[0]), sample_size, replace=False)
            
            fringing_score = 0
            for idx in indices:
                y, x = edge_coords[0][idx], edge_coords[1][idx]
                
                # Get local region around edge
                y1, y2 = max(0, y-2), min(image.shape[0], y+3)
                x1, x2 = max(0, x-2), min(image.shape[1], x+3)
                
                local_region = image[y1:y2, x1:x2]
                if local_region.size == 0:
                    continue
                
                # Check for purple/magenta fringing (high blue+red, low green)
                mean_color = np.mean(local_region.reshape(-1, 3), axis=0)
                r_val, g_val, b_val = mean_color[0], mean_color[1], mean_color[2]
                
                # Purple fringing detection
                if r_val > 150 and b_val > 150 and g_val < 100:
                    purple_strength = (r_val + b_val) / 2 - g_val
                    fringing_score += max(0, purple_strength) / 255.0
                
                # Green fringing detection
                if g_val > 150 and r_val < 100 and b_val < 100:
                    green_strength = g_val - (r_val + b_val) / 2
                    fringing_score += max(0, green_strength) / 255.0
            
            # Normalize and cap
            normalized_score = fringing_score / sample_size if sample_size > 0 else 0
            return min(normalized_score * 2, 1.0)  # Scale for better sensitivity
            
        except Exception as e:
            self.logger.warning(f"Chromatic aberration detection failed: {e}")
            return 0.0
    
    def check_noise_level(self, image: np.ndarray) -> float:
        """Estimate image noise level using multiple methods"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Laplacian-based noise estimation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_estimate_lap = np.std(laplacian)
            
            # Method 2: Median filter difference
            median_filtered = cv2.medianBlur(gray, 5)
            noise_map = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
            noise_estimate_med = np.mean(noise_map)
            
            # Method 3: Local variance in smooth regions
            # Find smooth regions (low local variance)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            local_mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
            local_var = cv2.morphologyEx((gray.astype(np.float32) - local_mean) ** 2, cv2.MORPH_CLOSE, kernel)
            
            smooth_mask = local_var < np.percentile(local_var, 20)  # Bottom 20% variance regions
            if np.sum(smooth_mask) > 0:
                noise_in_smooth = np.std(gray[smooth_mask])
            else:
                noise_in_smooth = noise_estimate_lap
            
            # Combine estimates with weights
            combined_noise = (
                noise_estimate_lap * 0.4 +
                noise_estimate_med * 0.4 +
                noise_in_smooth * 0.2
            )
            
            # Convert to 0-1 score (higher = less noise = better)
            # Typical noise levels: 0-5 excellent, 5-15 good, 15-30 fair, >30 poor
            noise_score = max(0, 1 - combined_noise / 30)
            return noise_score
            
        except Exception as e:
            self.logger.warning(f"Noise level detection failed: {e}")
            return 0.5  # Default moderate score
    
    def check_banding(self, image: np.ndarray) -> float:
        """Detect banding in gradients (particularly in sky/backgrounds)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find smooth gradient regions (sky, backgrounds)
            # Apply Gaussian blur to identify large smooth areas
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            gradient_magnitude = cv2.magnitude(
                cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3),
                cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            )
            
            # Identify smooth regions with low gradient
            smooth_regions = gradient_magnitude < np.percentile(gradient_magnitude, 30)
            
            if np.sum(smooth_regions) < 100:  # Not enough smooth area to analyze
                return 1.0
            
            # Extract smooth regions for analysis
            smooth_pixels = gray[smooth_regions]
            
            if len(smooth_pixels) < 100:
                return 1.0
            
            # Sort pixels to create a gradient profile
            sorted_pixels = np.sort(smooth_pixels)
            
            # Look for regular steps in the gradient (banding signature)
            # Calculate second derivative to find discontinuities
            if len(sorted_pixels) > 10:
                # Smooth the gradient profile
                smoothed_profile = cv2.GaussianBlur(sorted_pixels.reshape(1, -1).astype(np.float32), (1, 5), 0).flatten()
                
                # Calculate differences
                first_diff = np.diff(smoothed_profile)
                second_diff = np.diff(first_diff)
                
                # Look for regular patterns in second derivative
                if len(second_diff) > 0:
                    # High variance in second derivative suggests banding
                    banding_metric = np.std(second_diff)
                    
                    # Normalize (lower values = less banding = better score)
                    banding_score = max(0, 1 - banding_metric / 10)
                    return banding_score
            
            return 1.0  # No banding detected
            
        except Exception as e:
            self.logger.warning(f"Banding detection failed: {e}")
            return 1.0
    
    def check_moire(self, image: np.ndarray) -> float:
        """Detect moiré patterns using frequency analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply FFT to detect regular patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Remove DC component (center)
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            
            # Mask out low frequencies (remove general image structure)
            mask = np.ones((h, w))
            mask_radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            center_mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= mask_radius ** 2
            mask[center_mask] = 0
            
            # Apply mask to focus on mid-to-high frequencies where moiré appears
            masked_magnitude = magnitude * mask
            
            # Look for strong periodic patterns
            # Calculate the ratio of peak energy to mean energy
            mean_energy = np.mean(masked_magnitude)
            if mean_energy == 0:
                return 1.0
            
            # Find peaks significantly above mean
            threshold = mean_energy + 3 * np.std(masked_magnitude)
            peaks = masked_magnitude > threshold
            num_peaks = np.sum(peaks)
            
            # Strong regular patterns suggest moiré
            peak_ratio = num_peaks / (h * w)
            
            # Convert to score (fewer peaks = better)
            moire_score = max(0, 1 - peak_ratio * 1000)  # Scale appropriately
            
            return moire_score
            
        except Exception as e:
            self.logger.warning(f"Moiré detection failed: {e}")
            return 1.0
    
    def check_vignetting(self, image: np.ndarray) -> float:
        """Detect lens vignetting (darkening at corners)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            
            # Create radial distance map from center
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
            normalized_distances = distances / max_distance
            
            # Sample brightness at different radial distances
            # Create concentric rings
            rings = []
            for r in np.linspace(0.1, 0.9, 8):
                ring_mask = (normalized_distances >= r - 0.05) & (normalized_distances <= r + 0.05)
                if np.sum(ring_mask) > 0:
                    ring_brightness = np.mean(gray[ring_mask])
                    rings.append((r, ring_brightness))
            
            if len(rings) < 3:
                return 1.0  # Can't assess vignetting
            
            # Check if brightness decreases towards edges
            radii, brightness_values = zip(*rings)
            
            # Calculate brightness falloff from center to edge
            center_brightness = brightness_values[0]  # Innermost ring
            edge_brightness = brightness_values[-1]   # Outermost ring
            
            if center_brightness == 0:
                return 1.0
            
            # Vignetting ratio (1.0 = no vignetting, <1.0 = vignetting present)
            vignetting_ratio = edge_brightness / center_brightness
            
            # Convert to score (closer to 1.0 = less vignetting = better)
            # Typical good images have ratio > 0.85, heavy vignetting < 0.6
            if vignetting_ratio >= 0.85:
                return 1.0
            elif vignetting_ratio <= 0.6:
                return 0.0
            else:
                # Linear scale between 0.6 and 0.85
                return (vignetting_ratio - 0.6) / 0.25
            
        except Exception as e:
            self.logger.warning(f"Vignetting detection failed: {e}")
            return 1.0
    
    def detect_dust_spots(self, image: np.ndarray) -> int:
        """Detect sensor dust spots (small dark circular artifacts)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use morphological operations to detect small dark spots
            # Create a structuring element (small disk)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Apply tophat transform to isolate small dark features
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Invert to make dark spots bright
            inverted = cv2.bitwise_not(tophat)
            
            # Threshold to find potential dust spots
            threshold_value = np.percentile(inverted, 98)  # Top 2% brightest pixels
            _, dust_mask = cv2.threshold(inverted, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Find contours of potential dust spots
            contours, _ = cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to identify likely dust spots
            dust_spots = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Dust spots are typically small and circular
                if 5 < area < 100 and perimeter > 0:  # Size limits
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.6:  # Reasonably circular
                        dust_spots += 1
            
            return dust_spots
            
        except Exception as e:
            self.logger.warning(f"Dust spot detection failed: {e}")
            return 0
    
    def get_quality_issues(self, technical_issues: TechnicalIssues, 
                          thresholds: Optional[Dict] = None) -> List[str]:
        """Get list of quality issues based on thresholds"""
        if thresholds is None:
            thresholds = {
                'chromatic_aberration': 0.3,
                'noise_level': 0.4,
                'banding_score': 0.5,
                'moire_score': 0.5,
                'vignetting_score': 0.6,
                'dust_spots': 5
            }
        
        issues = []
        
        if technical_issues.chromatic_aberration > thresholds['chromatic_aberration']:
            issues.append(f"chromatic aberration ({technical_issues.chromatic_aberration:.2f})")
        
        if technical_issues.noise_level < thresholds['noise_level']:
            issues.append(f"high noise ({1-technical_issues.noise_level:.2f})")
        
        if technical_issues.banding_score < thresholds['banding_score']:
            issues.append(f"gradient banding ({1-technical_issues.banding_score:.2f})")
        
        if technical_issues.moire_score < thresholds['moire_score']:
            issues.append(f"moiré patterns ({1-technical_issues.moire_score:.2f})")
        
        if technical_issues.vignetting_score < thresholds['vignetting_score']:
            issues.append(f"heavy vignetting ({1-technical_issues.vignetting_score:.2f})")
        
        if technical_issues.dust_spots > thresholds['dust_spots']:
            issues.append(f"dust spots ({technical_issues.dust_spots})")
        
        return issues