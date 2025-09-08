"""
Intelligent Composition Analyzer - Real framing and composition analysis
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
import math


class IntelligentCompositionAnalyzer:
    """Smart composition analysis for framing quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_composition(self, image: Image.Image) -> Dict:
        """Comprehensive composition analysis"""
        
        # Convert to arrays
        rgb = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        composition_analysis = {}
        
        # 1. Horizon detection and leveling
        composition_analysis['horizon'] = self._analyze_horizon(gray)
        
        # 2. Subject placement and rule of thirds
        composition_analysis['subject_placement'] = self._analyze_subject_placement(gray, rgb)
        
        # 3. Leading lines detection
        composition_analysis['leading_lines'] = self._detect_leading_lines(gray)
        
        # 4. Symmetry and balance
        composition_analysis['balance'] = self._analyze_balance(gray)
        
        # 5. Edge analysis (cut-off subjects, distractions)
        composition_analysis['edges'] = self._analyze_edges(gray)
        
        # 6. Background analysis
        composition_analysis['background'] = self._analyze_background(rgb, gray)
        
        # 7. Obvious framing mistakes
        composition_analysis['framing_issues'] = self._detect_framing_issues(gray, hsv)
        
        # 8. Visual weight distribution
        composition_analysis['visual_weight'] = self._analyze_visual_weight(rgb)
        
        # Calculate overall composition score
        overall_score = self._calculate_composition_score(composition_analysis)
        
        # Get specific issues and recommendations
        issues = self._identify_composition_issues(composition_analysis)
        recommendations = self._get_composition_recommendations(composition_analysis, issues)
        
        return {
            'composition_score': overall_score,
            'detailed_analysis': composition_analysis,
            'issues': issues,
            'recommendations': recommendations,
            'is_well_composed': overall_score > 0.7,
            'composition_category': self._categorize_composition(overall_score, issues)
        }
    
    def _analyze_horizon(self, gray: np.ndarray) -> Dict:
        """Detect and analyze horizon line"""
        
        try:
            # Detect lines using HoughLines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            horizon_candidates = []
            
            if lines is not None:
                for rho, theta in lines[:, 0]:
                    # Look for near-horizontal lines (within 15 degrees of horizontal)
                    angle_degrees = abs(theta * 180 / np.pi - 90)
                    if angle_degrees < 15 or angle_degrees > 165:  # Horizontal-ish lines
                        horizon_candidates.append((rho, theta, angle_degrees))
            
            if horizon_candidates:
                # Find the most prominent horizontal line
                best_horizon = min(horizon_candidates, key=lambda x: x[2])  # Most horizontal
                rho, theta, angle = best_horizon
                
                # Calculate horizon position (as fraction of image height)
                h, w = gray.shape
                horizon_y = rho * np.sin(theta)
                horizon_position = horizon_y / h if h > 0 else 0.5
                
                # Check if horizon follows rule of thirds
                rule_of_thirds_score = self._evaluate_rule_of_thirds_position(horizon_position)
                
                # Check if horizon is level
                tilt_angle = abs(90 - theta * 180 / np.pi)
                is_level = tilt_angle < 2  # Within 2 degrees
                
                return {
                    'horizon_detected': True,
                    'horizon_position': float(horizon_position),
                    'tilt_angle': float(tilt_angle),
                    'is_level': bool(is_level),
                    'rule_of_thirds_score': float(rule_of_thirds_score)
                }
            else:
                return {
                    'horizon_detected': False,
                    'horizon_position': 0.5,
                    'tilt_angle': 0.0,
                    'is_level': True,
                    'rule_of_thirds_score': 0.5
                }
                
        except Exception as e:
            self.logger.warning(f"Horizon analysis failed: {e}")
            return {'horizon_detected': False}
    
    def _analyze_subject_placement(self, gray: np.ndarray, rgb: np.ndarray) -> Dict:
        """Analyze subject placement and rule of thirds"""
        
        h, w = gray.shape
        
        # Find regions of interest (high contrast areas likely to be subjects)
        # Use SIFT-like approach to find interest points
        try:
            # Simple interest point detection using corner detection
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
            
            if corners is not None:
                # Calculate center of mass of interest points
                corners = corners.reshape(-1, 2)
                subject_center_x = np.mean(corners[:, 0]) / w
                subject_center_y = np.mean(corners[:, 1]) / h
                
                # Evaluate placement according to rule of thirds
                thirds_score = self._evaluate_rule_of_thirds_placement(subject_center_x, subject_center_y)
                
                # Check for centered composition (sometimes good, sometimes not)
                center_distance = np.sqrt((subject_center_x - 0.5)**2 + (subject_center_y - 0.5)**2)
                is_centered = center_distance < 0.1
                
                return {
                    'subject_detected': True,
                    'subject_center_x': float(subject_center_x),
                    'subject_center_y': float(subject_center_y),
                    'rule_of_thirds_score': float(thirds_score),
                    'is_centered': bool(is_centered),
                    'center_distance': float(center_distance)
                }
            else:
                return {'subject_detected': False}
                
        except Exception as e:
            self.logger.warning(f"Subject placement analysis failed: {e}")
            return {'subject_detected': False}
    
    def _detect_leading_lines(self, gray: np.ndarray) -> Dict:
        """Detect leading lines in composition"""
        
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                h, w = gray.shape
                leading_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Check if line leads toward center or rule of thirds points
                    # Calculate line direction toward image center
                    center_x, center_y = w//2, h//2
                    
                    # Line midpoint
                    mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
                    
                    # Vector from line midpoint to center
                    to_center = np.array([center_x - mid_x, center_y - mid_y])
                    line_vec = np.array([x2-x1, y2-y1])
                    
                    # Normalize vectors
                    if np.linalg.norm(to_center) > 0 and np.linalg.norm(line_vec) > 0:
                        to_center_norm = to_center / np.linalg.norm(to_center)
                        line_vec_norm = line_vec / np.linalg.norm(line_vec)
                        
                        # Calculate alignment (dot product)
                        alignment = abs(np.dot(line_vec_norm, to_center_norm))
                        
                        if alignment > 0.3 and length > min(h, w) * 0.1:  # Reasonable length
                            leading_lines.append({
                                'length': float(length),
                                'angle': float(angle),
                                'alignment_to_center': float(alignment)
                            })
                
                # Score based on number and quality of leading lines
                if leading_lines:
                    avg_alignment = np.mean([line['alignment_to_center'] for line in leading_lines])
                    leading_lines_score = min(1.0, len(leading_lines) * avg_alignment / 3)
                else:
                    leading_lines_score = 0.0
                
                return {
                    'leading_lines_detected': len(leading_lines),
                    'leading_lines_score': float(leading_lines_score),
                    'lines_details': leading_lines[:5]  # Keep top 5 for analysis
                }
            else:
                return {'leading_lines_detected': 0, 'leading_lines_score': 0.0}
                
        except Exception as e:
            self.logger.warning(f"Leading lines detection failed: {e}")
            return {'leading_lines_detected': 0, 'leading_lines_score': 0.0}
    
    def _analyze_balance(self, gray: np.ndarray) -> Dict:
        """Analyze visual balance and symmetry"""
        
        h, w = gray.shape
        
        # Horizontal balance
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_weight = np.sum(left_half) / left_half.size
        right_weight = np.sum(right_half) / right_half.size
        
        horizontal_balance = 1.0 - abs(left_weight - right_weight) / max(left_weight, right_weight, 1)
        
        # Vertical balance  
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        top_weight = np.sum(top_half) / top_half.size
        bottom_weight = np.sum(bottom_half) / bottom_half.size
        
        vertical_balance = 1.0 - abs(top_weight - bottom_weight) / max(top_weight, bottom_weight, 1)
        
        # Symmetry detection
        # Horizontal symmetry
        left_flipped = np.flip(left_half, axis=1)
        if right_half.shape == left_flipped.shape:
            horizontal_symmetry = 1.0 - np.mean(np.abs(right_half.astype(float) - left_flipped.astype(float))) / 255
        else:
            horizontal_symmetry = 0.0
        
        # Vertical symmetry
        top_flipped = np.flip(top_half, axis=0)
        if bottom_half.shape == top_flipped.shape:
            vertical_symmetry = 1.0 - np.mean(np.abs(bottom_half.astype(float) - top_flipped.astype(float))) / 255
        else:
            vertical_symmetry = 0.0
        
        return {
            'horizontal_balance': float(horizontal_balance),
            'vertical_balance': float(vertical_balance),
            'horizontal_symmetry': float(horizontal_symmetry),
            'vertical_symmetry': float(vertical_symmetry),
            'overall_balance': float((horizontal_balance + vertical_balance) / 2)
        }
    
    def _analyze_edges(self, gray: np.ndarray) -> Dict:
        """Analyze image edges for cut-off subjects and distractions"""
        
        h, w = gray.shape
        edge_thickness = max(5, min(h, w) // 50)  # Adaptive edge thickness
        
        # Extract edge regions
        top_edge = gray[:edge_thickness, :]
        bottom_edge = gray[-edge_thickness:, :]
        left_edge = gray[:, :edge_thickness]
        right_edge = gray[:, -edge_thickness:]
        
        # Analyze each edge for activity (potential cut-off subjects)
        edges_analysis = {}
        
        for edge_name, edge_region in [
            ('top', top_edge), ('bottom', bottom_edge), 
            ('left', left_edge), ('right', right_edge)
        ]:
            if edge_region.size > 0:
                # Use gradient to detect cut-off subjects
                if edge_name in ['top', 'bottom']:
                    gradient = np.abs(np.diff(edge_region, axis=0))
                else:
                    gradient = np.abs(np.diff(edge_region, axis=1))
                
                edge_activity = np.mean(gradient) if gradient.size > 0 else 0
                
                # High activity at edges suggests cut-off subjects
                edges_analysis[edge_name] = {
                    'activity_level': float(edge_activity),
                    'likely_cutoff': edge_activity > 30  # Threshold for likely cut-off
                }
        
        # Count potential cut-offs
        cutoff_edges = sum(1 for edge in edges_analysis.values() if edge.get('likely_cutoff', False))
        
        return {
            'edges_analysis': edges_analysis,
            'cutoff_edges_count': cutoff_edges,
            'has_cutoffs': cutoff_edges > 0
        }
    
    def _analyze_background(self, rgb: np.ndarray, gray: np.ndarray) -> Dict:
        """Analyze background quality and distractions"""
        
        # Simple background detection - assume edges and corners are more likely background
        h, w = gray.shape
        
        # Create a mask for likely background areas (edges of image)
        background_mask = np.zeros((h, w), dtype=bool)
        edge_size = max(10, min(h, w) // 20)
        
        background_mask[:edge_size, :] = True  # Top
        background_mask[-edge_size:, :] = True  # Bottom
        background_mask[:, :edge_size] = True  # Left  
        background_mask[:, -edge_size:] = True  # Right
        
        if np.sum(background_mask) > 0:
            background_region = gray[background_mask]
            
            # Analyze background uniformity
            bg_variance = np.var(background_region)
            bg_uniformity = 1.0 / (1.0 + bg_variance / 1000)  # Higher variance = less uniform
            
            # Check for distracting elements (high contrast areas in background)
            bg_edges = cv2.Canny(gray * background_mask.astype(np.uint8), 50, 150)
            bg_edge_density = np.sum(bg_edges > 0) / np.sum(background_mask)
            
            # Analyze background color
            if np.sum(background_mask) > 0:
                bg_rgb = rgb[background_mask]
                bg_color_variance = np.mean([np.var(bg_rgb[:, i]) for i in range(3)])
            else:
                bg_color_variance = 0
            
            return {
                'background_uniformity': float(bg_uniformity),
                'background_edge_density': float(bg_edge_density),
                'background_color_variance': float(bg_color_variance),
                'is_distracting': bg_edge_density > 0.05 or bg_color_variance > 1000
            }
        else:
            return {'background_uniformity': 0.5, 'is_distracting': False}
    
    def _detect_framing_issues(self, gray: np.ndarray, hsv: np.ndarray) -> Dict:
        """Detect obvious framing mistakes"""
        
        issues = []
        h, w = gray.shape
        
        # 1. Check for extreme orientations (sideways sky/ground)
        # Detect sky (bright area in upper portion) 
        upper_third = gray[:h//3, :]
        lower_third = gray[2*h//3:, :]
        
        upper_brightness = np.mean(upper_third)
        lower_brightness = np.mean(lower_third)
        
        # Check if image might be upside down (bright area at bottom)
        if lower_brightness > upper_brightness + 30:
            issues.append("Possible upside-down orientation")
        
        # 2. Check for severely tilted horizons
        # (This would be caught by horizon analysis, but double-check)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            for rho, theta in lines[:10, 0]:  # Check first 10 lines
                angle_degrees = abs(theta * 180 / np.pi - 90)
                if 5 < angle_degrees < 85:  # Significantly tilted
                    issues.append(f"Severely tilted horizon ({angle_degrees:.1f}°)")
                    break
        
        # 3. Check for all-sky or all-ground images
        # Very bright images might be all sky
        if np.mean(gray) > 200:
            # Check if there's any significant detail
            edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size
            if edge_density < 0.01:  # Very few edges
                issues.append("Possible overexposed sky or plain background")
        
        # Very dark images might be all ground/shadows
        if np.mean(gray) < 50:
            edge_density = np.sum(cv2.Canny(gray, 30, 100) > 0) / gray.size
            if edge_density < 0.01:
                issues.append("Possible underexposed ground or dark subject")
        
        # 4. Check for extreme aspect ratios that might indicate cropping issues
        aspect_ratio = w / h
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            issues.append(f"Extreme aspect ratio ({aspect_ratio:.2f}:1)")
        
        # 5. Check for obvious centered subjects in action shots
        # (This is basic - could be expanded)
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        center_activity = np.var(center_region)
        edge_activity = np.var(gray) - center_activity
        
        if center_activity > edge_activity * 3:  # Very active center, dead edges
            issues.append("Possible static composition for dynamic subject")
        
        return {
            'framing_issues': issues,
            'issues_count': len(issues),
            'has_major_issues': len(issues) > 2
        }
    
    def _analyze_visual_weight(self, rgb: np.ndarray) -> Dict:
        """Analyze distribution of visual weight"""
        
        # Convert to grayscale for luminance
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Calculate visual weight based on contrast and detail
        # High contrast and detailed areas have more visual weight
        
        # Local contrast calculation
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Edge density for detail
        edges = cv2.Canny(gray, 50, 150)
        edge_kernel = np.ones((10, 10), np.float32) / 100
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, edge_kernel)
        
        # Combine for visual weight
        visual_weight = local_variance * 0.7 + edge_density * 0.3
        
        # Analyze weight distribution
        # Divide into 9 regions (3x3 grid)
        weight_regions = {}
        region_names = [
            'top_left', 'top_center', 'top_right',
            'middle_left', 'center', 'middle_right',
            'bottom_left', 'bottom_center', 'bottom_right'
        ]
        
        for i, name in enumerate(region_names):
            row, col = i // 3, i % 3
            y1, y2 = int(row * h / 3), int((row + 1) * h / 3)
            x1, x2 = int(col * w / 3), int((col + 1) * w / 3)
            
            region_weight = np.mean(visual_weight[y1:y2, x1:x2])
            weight_regions[name] = float(region_weight)
        
        # Calculate weight balance
        total_weight = sum(weight_regions.values())
        if total_weight > 0:
            weight_percentages = {k: v/total_weight for k, v in weight_regions.items()}
        else:
            weight_percentages = {k: 1/9 for k in weight_regions.keys()}
        
        # Check for good distribution (not too concentrated)
        max_weight = max(weight_percentages.values())
        weight_distribution_score = 1.0 - max(0, max_weight - 0.4) * 2  # Penalize if >40% in one region
        
        return {
            'weight_regions': weight_regions,
            'weight_percentages': weight_percentages,
            'weight_distribution_score': float(weight_distribution_score),
            'dominant_region': max(weight_percentages.items(), key=lambda x: x[1])[0]
        }
    
    def _evaluate_rule_of_thirds_position(self, position: float) -> float:
        """Evaluate how well a position follows rule of thirds"""
        # Rule of thirds lines are at 1/3 and 2/3
        thirds_lines = [1/3, 2/3]
        
        # Distance to nearest third line
        distances = [abs(position - line) for line in thirds_lines]
        min_distance = min(distances)
        
        # Score based on proximity to rule of thirds (closer = better)
        # Perfect score at exactly 1/3 or 2/3, decreases with distance
        score = max(0, 1.0 - min_distance * 6)  # 6 = scaling factor
        
        return score
    
    def _evaluate_rule_of_thirds_placement(self, x: float, y: float) -> float:
        """Evaluate rule of thirds for 2D placement"""
        x_score = self._evaluate_rule_of_thirds_position(x)
        y_score = self._evaluate_rule_of_thirds_position(y)
        
        # Combined score
        return (x_score + y_score) / 2
    
    def _calculate_composition_score(self, analysis: Dict) -> float:
        """Calculate overall composition score"""
        
        score_components = []
        
        # Horizon analysis (15% weight)
        if analysis['horizon']['horizon_detected']:
            horizon_score = analysis['horizon']['rule_of_thirds_score']
            if analysis['horizon']['is_level']:
                horizon_score *= 1.2  # Bonus for level horizon
            score_components.append(('horizon', horizon_score, 0.15))
        
        # Subject placement (25% weight)
        if analysis['subject_placement']['subject_detected']:
            placement_score = analysis['subject_placement']['rule_of_thirds_score']
            score_components.append(('subject_placement', placement_score, 0.25))
        
        # Leading lines (15% weight)
        lines_score = analysis['leading_lines']['leading_lines_score']
        score_components.append(('leading_lines', lines_score, 0.15))
        
        # Balance (15% weight)
        balance_score = analysis['balance']['overall_balance']
        score_components.append(('balance', balance_score, 0.15))
        
        # Edge analysis (10% weight) - penalize cut-offs
        edge_score = 1.0 - (analysis['edges']['cutoff_edges_count'] * 0.2)
        score_components.append(('edges', max(0, edge_score), 0.10))
        
        # Background (10% weight)
        bg = analysis['background']
        bg_score = bg['background_uniformity']
        if bg['is_distracting']:
            bg_score *= 0.5  # Penalty for distracting background
        score_components.append(('background', bg_score, 0.10))
        
        # Visual weight distribution (10% weight)
        weight_score = analysis['visual_weight']['weight_distribution_score']
        score_components.append(('visual_weight', weight_score, 0.10))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        # Penalty for major framing issues
        framing_penalty = analysis['framing_issues']['issues_count'] * 0.1
        total_score = max(0, total_score - framing_penalty)
        
        return max(0.0, min(1.0, total_score))
    
    def _identify_composition_issues(self, analysis: Dict) -> List[str]:
        """Identify composition issues"""
        
        issues = []
        
        # Horizon issues
        if analysis['horizon']['horizon_detected'] and not analysis['horizon']['is_level']:
            tilt = analysis['horizon']['tilt_angle']
            issues.append(f"Tilted horizon ({tilt:.1f}°)")
        
        # Subject placement issues
        if analysis['subject_placement']['subject_detected']:
            thirds_score = analysis['subject_placement']['rule_of_thirds_score']
            if thirds_score < 0.3:
                issues.append("Poor subject placement (rule of thirds)")
        
        # Cut-off subjects
        if analysis['edges']['has_cutoffs']:
            issues.append("Possible cut-off subjects at edges")
        
        # Distracting background
        if analysis['background']['is_distracting']:
            issues.append("Distracting background elements")
        
        # Poor weight distribution
        if analysis['visual_weight']['weight_distribution_score'] < 0.4:
            dominant = analysis['visual_weight']['dominant_region']
            issues.append(f"Unbalanced composition (weight concentrated in {dominant})")
        
        # Add framing issues
        issues.extend(analysis['framing_issues']['framing_issues'])
        
        return issues
    
    def _get_composition_recommendations(self, analysis: Dict, issues: List[str]) -> List[str]:
        """Get composition recommendations"""
        
        recommendations = []
        
        if "Tilted horizon" in ' '.join(issues):
            recommendations.append("Straighten horizon in post-processing")
        
        if "Poor subject placement" in ' '.join(issues):
            recommendations.append("Consider cropping to improve subject placement using rule of thirds")
        
        if "cut-off subjects" in ' '.join(issues):
            recommendations.append("Check for important elements cut off at edges")
        
        if "Distracting background" in ' '.join(issues):
            recommendations.append("Consider background cleanup or depth of field adjustment")
        
        if "Unbalanced composition" in ' '.join(issues):
            recommendations.append("Recompose to better distribute visual weight")
        
        if "upside-down" in ' '.join(issues):
            recommendations.append("Check image orientation")
        
        if len(issues) == 0:
            recommendations.append("Well composed - good framing and balance")
        
        return recommendations
    
    def _categorize_composition(self, score: float, issues: List[str]) -> str:
        """Categorize composition quality"""
        
        if score >= 0.85 and len(issues) <= 1:
            return "Excellent"
        elif score >= 0.7:
            return "Good"  
        elif score >= 0.5:
            return "Acceptable"
        elif score >= 0.3:
            return "Poor"
        else:
            return "Very Poor"