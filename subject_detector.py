import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional, Dict

class SubjectDetector:
    """Detect and analyze focus on subjects, especially for portraits"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_portrait_subject(self, image: Image.Image) -> Dict:
        """Detect if image is portrait and find eye regions"""
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return {"is_portrait": False, "requires_eye_focus": False}
        
        # Find the largest face (likely the subject)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        main_face = faces[0]
        x, y, w, h = main_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.05, minNeighbors=3
        )
        
        result = {
            "is_portrait": True,
            "requires_eye_focus": len(eyes) > 0,
            "face_region": (x, y, w, h),
            "eye_regions": [(x+ex, y+ey, ew, eh) for ex, ey, ew, eh in eyes]
        }
        
        # Calculate focus quality on eyes
        if result["requires_eye_focus"] and len(eyes) > 0:
            eye_sharpness = []
            for ex, ey, ew, eh in eyes[:2]:  # Check first 2 eyes
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                sharpness = self._calculate_roi_sharpness(eye_roi)
                eye_sharpness.append(sharpness)
            
            result["eye_sharpness"] = max(eye_sharpness) if eye_sharpness else 0
            result["both_eyes_sharp"] = all(s > 0.6 for s in eye_sharpness)
        
        return result
    
    def _calculate_roi_sharpness(self, roi):
        """Calculate sharpness for a specific region"""
        if roi.size == 0:
            return 0.0
        
        # Laplacian variance - higher = sharper
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (adjust these based on your images)
        # For eye regions, we expect very high frequency detail
        normalized = np.clip(variance / 500, 0, 1)
        
        return normalized
    
    def check_subject_focus(self, image: Image.Image) -> float:
        """
        Returns focus score based on subject detection
        1.0 = Perfect focus on subject
        0.0 = Missed focus
        """
        portrait_info = self.detect_portrait_subject(image)
        
        if portrait_info["is_portrait"] and portrait_info["requires_eye_focus"]:
            # For portraits, eye sharpness is critical
            return portrait_info.get("eye_sharpness", 0.5)
        
        # For non-portraits, fall back to center-weighted focus check
        return self._check_center_focus(image)
    
    def _check_center_focus(self, image: Image.Image) -> float:
        """Check focus in center region (where subject usually is)"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        h, w = cv_image.shape
        # Extract center 30% of image
        center_y = int(h * 0.35)
        center_x = int(w * 0.35)
        center_roi = cv_image[center_y:h-center_y, center_x:w-center_x]
        
        return self._calculate_roi_sharpness(center_roi)