"""
Smart Exposure Analyzer - Real exposure analysis beyond simple clipping
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging


class SmartExposureAnalyzer:
    """Advanced exposure analysis that actually works"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_exposure(self, image: Image.Image) -> Dict:
        """Comprehensive exposure analysis"""
        
        # Convert to arrays - ensure proper format
        rgb = np.array(image.convert('RGB'), dtype=np.uint8)
        if len(rgb.shape) != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Invalid RGB array shape: {rgb.shape}")
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Multiple exposure assessments
        exposure_analysis = {}
        
        # 1. Advanced histogram analysis
        exposure_analysis['histogram'] = self._analyze_histogram(gray, rgb)
        
        # 2. Tonal distribution analysis
        exposure_analysis['tonal'] = self._analyze_tonal_distribution(gray)
        
        # 3. Dynamic range assessment
        exposure_analysis['dynamic_range'] = self._assess_dynamic_range(rgb)
        
        # 4. Shadow and highlight detail
        exposure_analysis['detail_retention'] = self._analyze_detail_retention(rgb)
        
        # 5. Per-channel analysis
        exposure_analysis['channels'] = self._analyze_channels(rgb)
        
        # 6. Regional exposure analysis
        exposure_analysis['regional'] = self._regional_exposure_analysis(gray)
        
        # 7. Skin tone exposure (if applicable)
        exposure_analysis['skin_tones'] = self._analyze_skin_tone_exposure(rgb)
        
        # Calculate overall exposure score
        overall_score = self._calculate_exposure_score(exposure_analysis)
        
        # Get specific issues and recommendations
        issues = self._identify_exposure_issues(exposure_analysis)
        recommendations = self._get_exposure_recommendations(exposure_analysis, issues)
        
        return {
            'exposure_score': overall_score,
            'detailed_analysis': exposure_analysis,
            'issues': issues,
            'recommendations': recommendations,
            'is_well_exposed': overall_score > 0.7,
            'exposure_category': self._categorize_exposure(overall_score, issues)
        }
    
    def _analyze_histogram(self, gray: np.ndarray, rgb: np.ndarray) -> Dict:
        """Advanced histogram analysis"""
        
        # Calculate histograms - ensure proper input format
        hist_gray = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([rgb[:,:,0]], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([rgb[:,:,1]], [0], None, [256], [0, 256]) 
        hist_b = cv2.calcHist([rgb[:,:,2]], [0], None, [256], [0, 256])
        
        total_pixels = gray.size
        
        # Analyze distribution
        # Shadows (0-63), Midtones (64-191), Highlights (192-255)
        shadows = np.sum(hist_gray[:64]) / total_pixels
        midtones = np.sum(hist_gray[64:192]) / total_pixels  
        highlights = np.sum(hist_gray[192:]) / total_pixels
        
        # Hard clipping analysis
        black_clipped = hist_gray[0] / total_pixels
        white_clipped = hist_gray[255] / total_pixels
        
        # Soft clipping (near clipping)
        near_black = np.sum(hist_gray[:16]) / total_pixels  # Bottom 6.25%
        near_white = np.sum(hist_gray[240:]) / total_pixels  # Top 6.25%
        
        # Histogram shape analysis
        peak_location = np.argmax(hist_gray)
        histogram_spread = np.std(np.repeat(np.arange(256), hist_gray.astype(int)))
        
        # Check for gaps in histogram (posterization)
        non_zero_bins = np.count_nonzero(hist_gray)
        histogram_continuity = non_zero_bins / 256
        
        return {
            'shadows_percentage': float(shadows),
            'midtones_percentage': float(midtones),
            'highlights_percentage': float(highlights),
            'black_clipped': float(black_clipped),
            'white_clipped': float(white_clipped),
            'near_black': float(near_black),
            'near_white': float(near_white),
            'peak_location': int(peak_location),
            'histogram_spread': float(histogram_spread),
            'histogram_continuity': float(histogram_continuity)
        }
    
    def _analyze_tonal_distribution(self, gray: np.ndarray) -> Dict:
        """Analyze tonal distribution quality"""
        
        # Calculate percentiles for tonal analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        tonal_values = [np.percentile(gray, p) for p in percentiles]
        
        # Tonal separation - good photos have good separation between tones
        tonal_separation = []
        for i in range(1, len(tonal_values)):
            separation = tonal_values[i] - tonal_values[i-1]
            tonal_separation.append(separation)
        
        avg_separation = np.mean(tonal_separation)
        min_separation = np.min(tonal_separation)
        
        # Check for blocked shadows/blown highlights
        shadow_detail = tonal_values[2] - tonal_values[0]  # 10th percentile - 1st percentile
        highlight_detail = tonal_values[-1] - tonal_values[-3]  # 99th - 90th percentile
        
        # Overall tonal range
        tonal_range = tonal_values[-1] - tonal_values[0]
        
        return {
            'tonal_percentiles': {str(p): float(v) for p, v in zip(percentiles, tonal_values)},
            'average_tonal_separation': float(avg_separation),
            'minimum_tonal_separation': float(min_separation),
            'shadow_detail': float(shadow_detail),
            'highlight_detail': float(highlight_detail),
            'tonal_range': float(tonal_range)
        }
    
    def _assess_dynamic_range(self, rgb: np.ndarray) -> Dict:
        """Assess the dynamic range utilization"""
        
        # Convert to different color spaces for analysis
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance = lab[:, :, 0]  # L channel
        
        # Effective dynamic range (excluding extreme 1% on each end)
        effective_min = np.percentile(luminance, 1)
        effective_max = np.percentile(luminance, 99)
        effective_range = effective_max - effective_min
        
        # Theoretical maximum range
        theoretical_max = 100  # LAB L channel goes 0-100
        range_utilization = effective_range / theoretical_max
        
        # Local contrast assessment using local standard deviation
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(luminance.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((luminance.astype(np.float32) - local_mean)**2, -1, kernel)
        local_contrast = np.mean(np.sqrt(local_variance))
        
        return {
            'effective_range': float(effective_range),
            'range_utilization': float(range_utilization),
            'local_contrast': float(local_contrast),
            'dynamic_range_score': min(1.0, range_utilization * 2)  # Score based on utilization
        }
    
    def _analyze_detail_retention(self, rgb: np.ndarray) -> Dict:
        """Analyze detail retention in shadows and highlights"""
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Define shadow and highlight regions
        shadow_mask = gray < 50   # Dark areas
        highlight_mask = gray > 200  # Bright areas
        
        shadow_detail_score = 0.5  # Default
        highlight_detail_score = 0.5
        
        if np.sum(shadow_mask) > 0:
            # Analyze detail in shadows using local variance
            shadow_region = gray[shadow_mask]
            shadow_variance = np.var(shadow_region)
            # More variance = more detail retained
            shadow_detail_score = min(1.0, shadow_variance / 50)
        
        if np.sum(highlight_mask) > 0:
            # Analyze detail in highlights
            highlight_region = gray[highlight_mask]
            highlight_variance = np.var(highlight_region)
            highlight_detail_score = min(1.0, highlight_variance / 50)
        
        # Overall detail retention
        overall_detail = (shadow_detail_score + highlight_detail_score) / 2
        
        return {
            'shadow_detail_score': float(shadow_detail_score),
            'highlight_detail_score': float(highlight_detail_score),
            'overall_detail_retention': float(overall_detail),
            'shadow_area_percentage': float(np.sum(shadow_mask) / gray.size),
            'highlight_area_percentage': float(np.sum(highlight_mask) / gray.size)
        }
    
    def _analyze_channels(self, rgb: np.ndarray) -> Dict:
        """Analyze individual color channels for exposure issues"""
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        channels = {'red': r, 'green': g, 'blue': b}
        channel_analysis = {}
        
        for name, channel in channels.items():
            # Clipping analysis per channel
            clipped_low = np.sum(channel <= 5) / channel.size
            clipped_high = np.sum(channel >= 250) / channel.size
            
            # Mean and distribution
            mean_value = np.mean(channel)
            std_value = np.std(channel)
            
            # Dynamic range per channel
            channel_range = np.percentile(channel, 99) - np.percentile(channel, 1)
            
            channel_analysis[name] = {
                'mean': float(mean_value),
                'std': float(std_value),
                'clipped_low': float(clipped_low),
                'clipped_high': float(clipped_high),
                'dynamic_range': float(channel_range)
            }
        
        # Color balance assessment
        r_mean, g_mean, b_mean = [channel_analysis[c]['mean'] for c in ['red', 'green', 'blue']]
        color_balance_score = 1.0 - np.std([r_mean, g_mean, b_mean]) / np.mean([r_mean, g_mean, b_mean])
        
        return {
            'channels': channel_analysis,
            'color_balance_score': float(color_balance_score)
        }
    
    def _regional_exposure_analysis(self, gray: np.ndarray) -> Dict:
        """Analyze exposure across different regions"""
        
        h, w = gray.shape
        
        # Define regions
        regions = {
            'center': gray[h//4:3*h//4, w//4:3*w//4],
            'top': gray[:h//3, :],
            'bottom': gray[2*h//3:, :],
            'left': gray[:, :w//3],
            'right': gray[:, 2*w//3:]
        }
        
        regional_stats = {}
        
        for region_name, region in regions.items():
            if region.size > 0:
                regional_stats[region_name] = {
                    'mean': float(np.mean(region)),
                    'std': float(np.std(region)),
                    'min': float(np.min(region)),
                    'max': float(np.max(region))
                }
        
        # Calculate exposure consistency
        means = [stats['mean'] for stats in regional_stats.values()]
        exposure_consistency = 1.0 - (np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0
        
        return {
            'regions': regional_stats,
            'exposure_consistency': float(exposure_consistency)
        }
    
    def _analyze_skin_tone_exposure(self, rgb: np.ndarray) -> Dict:
        """Analyze skin tone exposure if skin tones are present"""
        
        try:
            # Simple skin detection in RGB
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
            # Basic skin tone detection (simplified)
            skin_mask = (
                (r > 95) & (g > 40) & (b > 20) &
                (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) > 15) &
                (np.abs(r - g) > 15) & (r > g) & (r > b)
            )
            
            skin_percentage = np.sum(skin_mask) / skin_mask.size
            
            if skin_percentage > 0.02:  # If >2% of image is skin
                skin_region = rgb[skin_mask]
                if len(skin_region) > 0:
                    skin_luminance = 0.299 * skin_region[:, 0] + 0.587 * skin_region[:, 1] + 0.114 * skin_region[:, 2]
                    
                    # Good skin tone exposure is typically 120-180 in 8-bit
                    optimal_skin_range = (120, 180)
                    skin_in_range = np.sum((skin_luminance >= optimal_skin_range[0]) & 
                                         (skin_luminance <= optimal_skin_range[1])) / len(skin_luminance)
                    
                    return {
                        'skin_detected': True,
                        'skin_percentage': float(skin_percentage),
                        'skin_exposure_score': float(skin_in_range),
                        'average_skin_luminance': float(np.mean(skin_luminance))
                    }
            
            return {
                'skin_detected': False,
                'skin_percentage': float(skin_percentage)
            }
            
        except Exception as e:
            self.logger.warning(f"Skin tone analysis failed: {e}")
            return {
                'skin_detected': False,
                'skin_percentage': 0.0
            }
    
    def _calculate_exposure_score(self, analysis: Dict) -> float:
        """Calculate overall exposure score from all analyses"""
        
        score_components = []
        
        # Histogram analysis (30% weight)
        hist = analysis['histogram']
        histogram_score = 1.0
        
        # Penalize hard clipping (but be more realistic)
        histogram_score -= hist['black_clipped'] * 1.0  # Less harsh penalty
        histogram_score -= hist['white_clipped'] * 0.8  # Even less harsh
        
        # Penalize extreme near-clipping (much more tolerant)
        histogram_score -= max(0, hist['near_black'] - 0.15) * 1.5  # Allow more near-black
        histogram_score -= max(0, hist['near_white'] - 0.10) * 1.0   # Allow more near-white
        
        # Reward good midtone distribution
        midtone_balance = 1.0 - abs(hist['midtones_percentage'] - 0.5)  # Prefer ~50% midtones
        histogram_score *= midtone_balance
        
        score_components.append(('histogram', max(0, histogram_score), 0.30))
        
        # Tonal distribution (25% weight) 
        tonal = analysis['tonal']
        tonal_score = min(1.0, tonal['average_tonal_separation'] / 20)  # Good separation
        tonal_score *= min(1.0, tonal['tonal_range'] / 200)  # Good range utilization
        
        score_components.append(('tonal', tonal_score, 0.25))
        
        # Dynamic range (20% weight)
        dr_score = analysis['dynamic_range']['dynamic_range_score']
        score_components.append(('dynamic_range', dr_score, 0.20))
        
        # Detail retention (15% weight)
        detail_score = analysis['detail_retention']['overall_detail_retention']
        score_components.append(('detail_retention', detail_score, 0.15))
        
        # Channel analysis (10% weight)
        channel_score = analysis['channels']['color_balance_score']
        # Penalize severe channel clipping
        for channel_data in analysis['channels']['channels'].values():
            channel_score -= channel_data['clipped_low'] * 0.5
            channel_score -= channel_data['clipped_high'] * 0.3
        
        score_components.append(('channels', max(0, channel_score), 0.10))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return max(0.0, min(1.0, total_score))
    
    def _identify_exposure_issues(self, analysis: Dict) -> List[str]:
        """Identify specific exposure issues"""
        
        issues = []
        
        hist = analysis['histogram']
        
        # Clipping issues
        if hist['black_clipped'] > 0.01:
            issues.append(f"Shadow clipping ({hist['black_clipped']:.1%})")
        
        if hist['white_clipped'] > 0.005:
            issues.append(f"Highlight clipping ({hist['white_clipped']:.1%})")
        
        # Exposure problems
        if hist['shadows_percentage'] > 0.7:
            issues.append("Underexposed")
        elif hist['highlights_percentage'] > 0.7:
            issues.append("Overexposed")
        
        # Tonal issues
        tonal = analysis['tonal']
        if tonal['average_tonal_separation'] < 5:
            issues.append("Poor tonal separation")
        
        if tonal['tonal_range'] < 100:
            issues.append("Limited tonal range")
        
        # Dynamic range issues
        if analysis['dynamic_range']['range_utilization'] < 0.3:
            issues.append("Poor dynamic range utilization")
        
        # Detail retention issues
        detail = analysis['detail_retention']
        if detail['shadow_detail_score'] < 0.3:
            issues.append("Lost shadow detail")
        if detail['highlight_detail_score'] < 0.3:
            issues.append("Lost highlight detail")
        
        # Color balance issues
        if analysis['channels']['color_balance_score'] < 0.7:
            issues.append("Poor color balance")
        
        return issues
    
    def _get_exposure_recommendations(self, analysis: Dict, issues: List[str]) -> List[str]:
        """Get specific recommendations based on analysis"""
        
        recommendations = []
        
        if "Underexposed" in issues:
            recommendations.append("Consider increasing exposure in post-processing")
        
        if "Overexposed" in issues:
            recommendations.append("Consider reducing exposure or highlights")
        
        if "Shadow clipping" in issues:
            recommendations.append("Lift shadows carefully to recover detail")
        
        if "Highlight clipping" in issues:
            recommendations.append("Reduce highlights or use highlight recovery")
        
        if "Poor tonal separation" in issues:
            recommendations.append("Increase contrast to improve tonal separation")
        
        if "Limited tonal range" in issues:
            recommendations.append("Expand tonal range using curves or levels")
        
        if len(issues) == 0:
            recommendations.append("Well exposed - suitable for processing")
        
        return recommendations
    
    def _categorize_exposure(self, score: float, issues: List[str]) -> str:
        """Categorize the exposure quality"""
        
        if score >= 0.9 and len(issues) == 0:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Acceptable"
        elif score >= 0.3:
            return "Poor"
        else:
            return "Very Poor"