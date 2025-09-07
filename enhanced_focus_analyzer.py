"""
Enhanced focus analyzer with subject detection and depth of field analysis
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FocusRegion:
    """Represents a focused region in the image"""
    x: int
    y: int
    width: int
    height: int
    sharpness: float
    is_subject: bool = False

class EnhancedFocusAnalyzer:
    """
    Advanced focus analyzer that understands shallow DOF photography
    and evaluates subject sharpness rather than overall image sharpness
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load OpenCV cascade classifiers for subject detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
        except Exception as e:
            self.logger.warning(f"Failed to load face cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
            self.profile_cascade = None
    
    def analyze_focus(self, image: Image.Image) -> Dict:
        """
        Comprehensive focus analysis including:
        - Subject detection and classification
        - Depth of field analysis
        - Subject-specific sharpness evaluation
        - Background/foreground separation
        """
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Analyze image structure
        subject_info = self._detect_subject_type(cv_image, gray)
        dof_analysis = self._analyze_depth_of_field(gray)
        focus_regions = self._find_focus_regions(gray)
        
        # Calculate subject-specific sharpness
        subject_sharpness = self._evaluate_subject_sharpness(
            gray, subject_info, focus_regions
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            subject_info, dof_analysis, subject_sharpness
        )
        
        # Calculate overall focus score
        focus_score = self._calculate_focus_score(
            subject_sharpness, subject_info, dof_analysis
        )
        
        return {
            'focus_score': focus_score,
            'subject_type': subject_info['type'],
            'subject_sharpness': subject_sharpness,
            'background_blur': dof_analysis['background_blur'],
            'is_shallow_dof': dof_analysis['is_shallow'],
            'focus_regions': [
                {
                    'x': r.x, 'y': r.y, 
                    'width': r.width, 'height': r.height,
                    'sharpness': r.sharpness, 'is_subject': r.is_subject
                } 
                for r in focus_regions
            ],
            'recommendations': recommendations
        }
    
    def _detect_subject_type(self, cv_image: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect and classify the main subject"""
        
        height, width = gray.shape
        subject_info = {
            'type': 'general',
            'regions': [],
            'confidence': 0.0,
            'eyes_detected': [],
            'face_regions': []
        }
        
        if not self.face_cascade:
            return subject_info
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        if len(faces) > 0:
            subject_info['type'] = 'portrait'
            subject_info['confidence'] = 0.9
            
            # Sort faces by size (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            for face in faces[:2]:  # Process up to 2 largest faces
                x, y, w, h = face
                subject_info['face_regions'].append((x, y, w, h))
                
                # Look for eyes in face region
                face_roi = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi, scaleFactor=1.05, minNeighbors=3
                )
                
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    # Convert to global coordinates
                    global_eye = (x + ex, y + ey, ew, eh)
                    subject_info['eyes_detected'].append(global_eye)
        
        # Check for profile faces if no frontal faces found
        elif self.profile_cascade:
            profiles = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(profiles) > 0:
                subject_info['type'] = 'portrait'
                subject_info['confidence'] = 0.7
                for profile in profiles[:1]:
                    subject_info['face_regions'].append(profile)
        
        # Detect other subject types based on image characteristics
        else:
            subject_info.update(self._classify_non_portrait_subject(cv_image, gray))
        
        return subject_info
    
    def _classify_non_portrait_subject(self, cv_image: np.ndarray, gray: np.ndarray) -> Dict:
        """Classify non-portrait subjects"""
        
        height, width = gray.shape
        
        # Simple heuristics for subject classification
        # This could be enhanced with ML models
        
        # Check for macro photography (high detail in center, blur elsewhere)
        center_region = gray[height//3:2*height//3, width//3:2*width//3]
        edge_region = np.concatenate([
            gray[:height//4, :].flatten(),
            gray[3*height//4:, :].flatten(),
            gray[:, :width//4].flatten(),
            gray[:, 3*width//4:].flatten()
        ])
        
        center_variance = np.var(center_region)
        edge_variance = np.var(edge_region)
        
        if center_variance > edge_variance * 2 and center_variance > 1000:
            return {
                'type': 'macro',
                'confidence': 0.6,
                'regions': [(width//3, height//3, width//3, height//3)]
            }
        
        # Check for landscape (uniform distribution of detail)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Divide image into regions and check edge distribution
        regions = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * height // 3, (i + 1) * height // 3
                x1, x2 = j * width // 3, (j + 1) * width // 3
                region_edges = edges[y1:y2, x1:x2]
                regions.append(np.mean(region_edges))
        
        edge_uniformity = 1.0 - (np.std(regions) / (np.mean(regions) + 1e-6))
        
        if edge_uniformity > 0.7:
            return {
                'type': 'landscape',
                'confidence': 0.5,
                'regions': []
            }
        
        # Check for street photography (subjects in lower 2/3 of image)
        lower_region = gray[height//3:, :]
        upper_region = gray[:height//3, :]
        
        if np.var(lower_region) > np.var(upper_region) * 1.5:
            return {
                'type': 'street',
                'confidence': 0.4,
                'regions': [(0, height//3, width, 2*height//3)]
            }
        
        return {
            'type': 'general',
            'confidence': 0.3,
            'regions': []
        }
    
    def _analyze_depth_of_field(self, gray: np.ndarray) -> Dict:
        """Analyze depth of field characteristics"""
        
        height, width = gray.shape
        
        # Calculate sharpness in different regions
        regions = {
            'center': gray[height//3:2*height//3, width//3:2*width//3],
            'edges': np.concatenate([
                gray[:height//4, :].flatten(),
                gray[3*height//4:, :].flatten(),
                gray[:, :width//4].flatten(),
                gray[:, 3*width//4:].flatten()
            ]),
            'mid_ring': None  # Will calculate below
        }
        
        # Create mid-ring region (between center and edges)
        mask = np.zeros_like(gray, dtype=bool)
        center_mask = np.zeros_like(gray, dtype=bool)
        center_mask[height//3:2*height//3, width//3:2*width//3] = True
        
        # Create ring mask
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        ring_mask = (distance > min(height, width) * 0.2) & (distance < min(height, width) * 0.4)
        
        regions['mid_ring'] = gray[ring_mask]
        
        # Calculate sharpness for each region
        sharpness = {}
        for region_name, region_data in regions.items():
            if region_data.size > 0:
                if len(region_data.shape) == 1:
                    # Ensure we can reshape to a square by trimming to perfect square size
                    total_pixels = len(region_data)
                    sqrt_size = int(np.sqrt(total_pixels))
                    perfect_square_size = sqrt_size * sqrt_size

                    if perfect_square_size < total_pixels:
                        # Trim the array to perfect square
                        region_data = region_data[:perfect_square_size]

                    if perfect_square_size > 0:
                        region_2d = region_data.reshape(sqrt_size, sqrt_size)
                        if region_2d.shape[1] < 3:  # Too small for Laplacian
                            sharpness[region_name] = 0
                            continue
                    else:
                        sharpness[region_name] = 0
                        continue
                else:
                    region_2d = region_data

                laplacian = cv2.Laplacian(region_2d.astype(np.uint8), cv2.CV_64F)
                sharpness[region_name] = laplacian.var()
            else:
                sharpness[region_name] = 0
        
        # Determine if shallow DOF
        center_sharp = sharpness.get('center', 0)
        edge_sharp = sharpness.get('edges', 0)
        ring_sharp = sharpness.get('mid_ring', 0)
        
        # Shallow DOF indicators
        is_shallow = False
        background_blur = 0.0
        
        if center_sharp > 0:
            edge_ratio = edge_sharp / center_sharp if center_sharp > 0 else 1.0
            ring_ratio = ring_sharp / center_sharp if center_sharp > 0 else 1.0
            
            # If center is significantly sharper than edges/ring
            if edge_ratio < 0.3 and ring_ratio < 0.5:
                is_shallow = True
                background_blur = 1.0 - min(edge_ratio, ring_ratio)
        
        return {
            'is_shallow': is_shallow,
            'background_blur': background_blur,
            'sharpness_distribution': sharpness,
            'center_to_edge_ratio': edge_sharp / center_sharp if center_sharp > 0 else 1.0
        }
    
    def _find_focus_regions(self, gray: np.ndarray) -> List[FocusRegion]:
        """Find regions of sharp focus in the image"""
        
        height, width = gray.shape
        
        # Calculate local sharpness using sliding window
        window_size = min(height, width) // 8
        stride = window_size // 2
        
        regions = []
        
        for y in range(0, height - window_size, stride):
            for x in range(0, width - window_size, stride):
                roi = gray[y:y+window_size, x:x+window_size]
                
                try:
                    # Calculate multiple sharpness metrics
                    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()

                    # Sobel gradient magnitude
                    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                    sobel_var = np.var(np.sqrt(sobel_x**2 + sobel_y**2))

                    # Combined sharpness score
                    sharpness = (laplacian_var + sobel_var) / 2

                    # Normalize
                    normalized_sharpness = min(sharpness / 1000, 1.0)

                    if normalized_sharpness > 0.3:  # Only include reasonably sharp regions
                        regions.append(FocusRegion(
                            x=x, y=y,
                            width=window_size, height=window_size,
                            sharpness=normalized_sharpness
                        ))
                except Exception as e:
                    # Skip regions that cause processing errors
                    self.logger.debug(f"Skipping focus region at ({x}, {y}): {e}")
                    continue
        
        # Sort by sharpness
        regions.sort(key=lambda r: r.sharpness, reverse=True)
        
        return regions[:20]  # Keep top 20 sharpest regions
    
    def _evaluate_subject_sharpness(self, gray: np.ndarray, 
                                   subject_info: Dict, 
                                   focus_regions: List[FocusRegion]) -> float:
        """Evaluate sharpness specifically for the detected subject"""
        
        if subject_info['type'] == 'portrait':
            return self._evaluate_portrait_sharpness(gray, subject_info, focus_regions)
        elif subject_info['type'] in ['macro', 'street']:
            return self._evaluate_subject_region_sharpness(gray, subject_info, focus_regions)
        else:
            # General case - use center-weighted average
            return self._evaluate_general_sharpness(gray, focus_regions)
    
    def _evaluate_portrait_sharpness(self, gray: np.ndarray, 
                                    subject_info: Dict, 
                                    focus_regions: List[FocusRegion]) -> float:
        """Evaluate portrait sharpness focusing on eyes"""
        
        if not subject_info['eyes_detected'] and not subject_info['face_regions']:
            return self._evaluate_general_sharpness(gray, focus_regions)
        
        max_sharpness = 0.0
        
        # Check eye sharpness first (highest priority)
        for eye_x, eye_y, eye_w, eye_h in subject_info['eyes_detected']:
            eye_roi = gray[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
            if eye_roi.size > 0:
                laplacian = cv2.Laplacian(eye_roi, cv2.CV_64F)
                eye_sharpness = min(laplacian.var() / 500, 1.0)
                max_sharpness = max(max_sharpness, eye_sharpness)
        
        # If no eyes detected, check face regions
        if max_sharpness == 0.0:
            for face_x, face_y, face_w, face_h in subject_info['face_regions']:
                # Focus on upper portion of face (where eyes should be)
                eye_region_y = face_y + face_h // 4
                eye_region_h = face_h // 3
                
                face_roi = gray[eye_region_y:eye_region_y+eye_region_h, 
                               face_x:face_x+face_w]
                if face_roi.size > 0:
                    laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
                    face_sharpness = min(laplacian.var() / 400, 1.0)
                    max_sharpness = max(max_sharpness, face_sharpness)
        
        return max_sharpness
    
    def _evaluate_subject_region_sharpness(self, gray: np.ndarray,
                                          subject_info: Dict,
                                          focus_regions: List[FocusRegion]) -> float:
        """Evaluate sharpness for subjects with defined regions"""
        
        if not subject_info['regions']:
            return self._evaluate_general_sharpness(gray, focus_regions)
        
        max_sharpness = 0.0
        
        for region_x, region_y, region_w, region_h in subject_info['regions']:
            roi = gray[region_y:region_y+region_h, region_x:region_x+region_w]
            if roi.size > 0:
                laplacian = cv2.Laplacian(roi, cv2.CV_64F)
                region_sharpness = min(laplacian.var() / 500, 1.0)
                max_sharpness = max(max_sharpness, region_sharpness)
        
        return max_sharpness
    
    def _evaluate_general_sharpness(self, gray: np.ndarray, 
                                   focus_regions: List[FocusRegion]) -> float:
        """Evaluate general image sharpness with center weighting"""
        
        if not focus_regions:
            # Fallback to simple center region analysis
            height, width = gray.shape
            center_roi = gray[height//3:2*height//3, width//3:2*width//3]
            laplacian = cv2.Laplacian(center_roi, cv2.CV_64F)
            return min(laplacian.var() / 500, 1.0)
        
        # Weight regions by distance from center
        height, width = gray.shape
        center_y, center_x = height // 2, width // 2
        
        weighted_sharpness = 0.0
        total_weight = 0.0
        
        for region in focus_regions[:10]:  # Top 10 sharpest regions
            # Calculate distance from center
            region_center_x = region.x + region.width // 2
            region_center_y = region.y + region.height // 2
            
            distance = np.sqrt((region_center_x - center_x)**2 + 
                             (region_center_y - center_y)**2)
            
            # Higher weight for regions closer to center
            max_distance = np.sqrt(center_x**2 + center_y**2)
            weight = 1.0 - (distance / max_distance)
            weight = max(weight, 0.1)  # Minimum weight
            
            weighted_sharpness += region.sharpness * weight
            total_weight += weight
        
        return weighted_sharpness / total_weight if total_weight > 0 else 0.0
    
    def _calculate_focus_score(self, subject_sharpness: float,
                              subject_info: Dict,
                              dof_analysis: Dict) -> float:
        """Calculate overall focus score considering context"""
        
        base_score = subject_sharpness
        
        # Adjust based on subject type and DOF
        if subject_info['type'] == 'portrait':
            # For portraits, subject sharpness is critical
            if subject_sharpness > 0.7:
                base_score = subject_sharpness
            else:
                # Penalize soft portraits
                base_score *= 0.7
        
        elif dof_analysis['is_shallow']:
            # For shallow DOF, subject sharpness is what matters
            # Don't penalize for background blur
            if subject_sharpness > 0.6:
                base_score = min(subject_sharpness * 1.1, 1.0)
        
        return base_score
    
    def _generate_recommendations(self, subject_info: Dict,
                                 dof_analysis: Dict,
                                 subject_sharpness: float) -> List[str]:
        """Generate focus-related recommendations"""
        
        recommendations = []
        
        if subject_info['type'] == 'portrait' and subject_sharpness < 0.6:
            if subject_info['eyes_detected']:
                recommendations.append("Eyes are not tack sharp - critical for portraits")
            else:
                recommendations.append("Face region appears soft - check focus on eyes")
        
        elif subject_sharpness < 0.4:
            recommendations.append("Main subject lacks sharpness - possible focus miss")
        
        if dof_analysis['is_shallow'] and subject_sharpness > 0.7:
            recommendations.append("Good shallow DOF technique - subject sharp, background blurred")
        
        if not dof_analysis['is_shallow'] and dof_analysis['center_to_edge_ratio'] < 0.5:
            recommendations.append("Possible camera shake or motion blur detected")
        
        if len(recommendations) == 0:
            if subject_sharpness > 0.8:
                recommendations.append("Excellent focus quality")
            elif subject_sharpness > 0.6:
                recommendations.append("Good focus quality")
        
        return recommendations
