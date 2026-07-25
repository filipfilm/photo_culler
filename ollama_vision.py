import requests
import json
import base64
import io
from PIL import Image
from typing import List, Dict
import logging
try:
    from .models import ImageMetrics
except ImportError:
    from models import ImageMetrics

DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_MODEL_ALIASES = {"gemma4", "gemma4:e4b", "gemma4:latest"}

# Try to import hybrid blur detector
try:
    from .blur_detector import HybridBlurDetector
    HYBRID_BLUR_AVAILABLE = True
except ImportError:
    try:
        from blur_detector import HybridBlurDetector
        HYBRID_BLUR_AVAILABLE = True
    except ImportError:
        HYBRID_BLUR_AVAILABLE = False


class OllamaVisionAnalyzer:
    """Simple vision analyzer using Ollama for f/1.8 shallow DOF photography"""
    
    def __init__(self, 
                 model: str = DEFAULT_OLLAMA_MODEL,
                 host: str = "http://localhost:11434",
                 timeout: int = 180,
                 use_hybrid_blur: bool = True):
        
        self.model = model
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.use_hybrid_blur = use_hybrid_blur and HYBRID_BLUR_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Initialize hybrid blur detector if available
        if self.use_hybrid_blur:
            self.blur_detector = HybridBlurDetector()
            self.logger.info("Hybrid blur detection enabled")
        else:
            self.blur_detector = None
            self.logger.info("Using vision-only blur detection")
        
        # Test connection and model availability
        self._check_ollama_connection()
        self._ensure_model_available()

    def _model_matches(self, requested: str, available: str) -> bool:
        requested_name = requested.strip().lower()
        available_name = available.strip().lower()
        if requested_name == available_name:
            return True

        if requested_name in DEFAULT_MODEL_ALIASES and available_name in DEFAULT_MODEL_ALIASES:
            return True

        if ":" not in requested_name or requested_name.endswith(":latest"):
            requested_base = requested_name.split(":", 1)[0]
            available_base = available_name.split(":", 1)[0]
            return requested_base == available_base

        return False
        
    def _check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
            self.logger.info(f"Connected to Ollama successfully at {self.host}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            raise ConnectionError(f"Cannot connect to Ollama. Make sure it's running at {self.host}")
    
    def _ensure_model_available(self):
        """Check if the vision model is available, pull if needed"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            models = response.json()
            
            available_models = [m['name'] for m in models.get('models', [])]
            
            if not any(self._model_matches(self.model, model) for model in available_models):
                self.logger.info(f"Model {self.model} not found, pulling...")
                self._pull_model()
            else:
                self.logger.info(f"Model {self.model} is available")
                
        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
            raise
    
    def _pull_model(self):
        """Pull the vision model"""
        self.logger.info(f"Pulling model {self.model}... This may take a while.")
        
        try:
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": self.model},
                timeout=600,  # 10 minutes for model pull
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        self.logger.info(f"Pull status: {data['status']}")
                    if data.get('status') == 'success':
                        break
                        
            self.logger.info(f"Model {self.model} pulled successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        # Resize image to reasonable size for processing
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def _query_ollama(self, prompt: str, image_base64: str) -> str:
        """Query Ollama with image and prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0
                }
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} {response.text}")
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Ollama query failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from Ollama"""
        import re
        
        # Look for JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate and normalize scores
                for key in ['blur_score', 'exposure_score', 'composition_score', 'overall_quality']:
                    if key in parsed:
                        score = float(parsed[key])
                        if score > 1:
                            score = min(score / 10.0 if score <= 10 else score / 100.0, 1.0)
                        parsed[key] = max(0.0, min(1.0, score))
                    else:
                        parsed[key] = 0.5
                
                return parsed
                
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # Fallback: return defaults if JSON parsing fails
        return {
            'blur_score': 0.5,
            'exposure_score': 0.5,
            'composition_score': 0.5,
            'overall_quality': 0.5,
            'keywords': [],
            'description': ''
        }
    
    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Analyze a single image"""
        return self.analyze_batch([image])[0]
    
    def analyze_batch(self, images: List[Image.Image]) -> List[ImageMetrics]:
        """Analyze multiple images"""
        metrics_list = []
        
        # Ultra-explicit blur detection prompt with examples
        prompt = """BLUR DETECTION TASK - CRITICAL FOCUS ASSESSMENT

Your ONLY job: Detect if the MAIN SUBJECT in this photo is sharp or blurry.

IGNORE background blur completely - only evaluate the main subject.

EXAMINE these details on the MAIN SUBJECT ONLY:
1. Are edges crisp and well-defined?
2. Can you see fine details clearly?
3. Is there any softness, smearing, or lack of sharpness?

BLUR DETECTION EXAMPLES:
- SHARP (0.8-1.0): Subject edges are crisp, fine details visible, no softness
- BLURRY (0.0-0.3): Subject is soft, smeared, edges unclear, details lost

SCORING INSTRUCTIONS:
If you see ANY of these on the main subject, score LOW (0.0-0.3):
❌ Soft or fuzzy edges
❌ Smeared or streaked appearance  
❌ Lost fine details
❌ Overall unsharpness
❌ Movement blur
❌ Focus on wrong area

If the main subject looks crisp and sharp, score HIGH (0.7-1.0):
✅ Clean, defined edges
✅ Clear fine details
✅ No softness or blur
✅ Professional sharpness

CRITICAL: Be very strict about sharpness. When in doubt, score lower.

Rate the main subject's sharpness (blur_score):
- 1.0 = Perfect sharpness, every detail crisp
- 0.8 = Good sharpness, professional quality
- 0.5 = Borderline acceptable
- 0.2 = Clearly blurry, poor quality
- 0.0 = Very blurry, unusable

Also rate:
- exposure_score: How well lit is the image?
- composition_score: How well framed is the subject?
- overall_quality: Your recommendation (weight sharpness heavily)

Return this JSON format only:
{
  "blur_score": 0.XX,
  "exposure_score": 0.XX,
  "composition_score": 0.XX,
  "overall_quality": 0.XX,
  "keywords": ["subject"],
  "description": "What you see"
}"""
        
        for image in images:
            try:
                # Convert image to base64
                image_base64 = self._image_to_base64(image)
                
                # Query Ollama
                response = self._query_ollama(prompt, image_base64)
                self.logger.debug(f"Ollama response: {response}")
                
                # Parse JSON response
                parsed = self._parse_json_response(response)
                vision_blur_score = parsed['blur_score']
                
                # Apply hybrid blur detection if available
                final_blur_score = vision_blur_score
                if self.blur_detector:
                    cv_metrics = self.blur_detector.detect_cv_blur(image)
                    final_blur_score = self.blur_detector.combine_vision_and_cv(
                        vision_blur_score, cv_metrics
                    )
                    self.logger.debug(f"Blur scores - Vision: {vision_blur_score:.3f}, CV: {cv_metrics['cv_sharpness_score']:.3f}, Final: {final_blur_score:.3f}")
                
                # Create metrics with hybrid blur score
                metrics = ImageMetrics(
                    blur_score=final_blur_score,
                    exposure_score=parsed['exposure_score'],
                    composition_score=parsed['composition_score'],
                    overall_quality=parsed['overall_quality'],
                    keywords=parsed.get('keywords', []),
                    description=parsed.get('description', '')
                )
                
                metrics_list.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze image: {e}")
                # Fallback to neutral scores
                metrics = ImageMetrics(
                    blur_score=0.5,
                    exposure_score=0.5,
                    composition_score=0.5,
                    overall_quality=0.5,
                    keywords=[],
                    description=""
                )
                metrics_list.append(metrics)
        
        return metrics_list


def test_ollama_vision():
    """Test function to verify Ollama vision setup"""
    try:
        # Test with a simple image
        test_image = Image.new('RGB', (400, 300), (128, 128, 128))
        
        analyzer = OllamaVisionAnalyzer()
        metrics = analyzer.analyze(test_image)
        
        print("✅ Ollama vision test successful!")
        print(f"Blur: {metrics.blur_score:.2f}")
        print(f"Exposure: {metrics.exposure_score:.2f}")
        print(f"Composition: {metrics.composition_score:.2f}")
        print(f"Overall: {metrics.overall_quality:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama vision test failed: {e}")
        return False


if __name__ == "__main__":
    test_ollama_vision()
