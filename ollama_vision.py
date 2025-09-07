import requests
import json
import base64
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple
import logging
from models import ImageMetrics, ProcessingMode


class OllamaVisionAnalyzer:
    """Vision analyzer using Ollama with vision-capable models"""
    
    def __init__(self, 
                 model: str = "llava:7b",
                 host: str = "http://localhost:11434",
                 timeout: int = 60):
        
        self.model = model
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Test connection and model availability
        self._check_ollama_connection()
        self._ensure_model_available()
        
    def _check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
            self.logger.info("Connected to Ollama successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            raise ConnectionError(f"Cannot connect to Ollama. Make sure it's running at {self.host}")
    
    def _ensure_model_available(self):
        """Check if the vision model is available, pull if needed"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            models = response.json()
            
            available_models = [m['name'] for m in models.get('models', [])]
            
            if not any(self.model in model for model in available_models):
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
                "stream": False
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Ollama query failed: {e}")
            raise
    
    def _parse_quality_scores(self, response: str) -> Dict:
        """Parse quality scores, keywords, and description from Ollama response"""
        scores = {
            'blur': 0.5,
            'exposure': 0.5,
            'composition': 0.5,
            'quality': 0.5,
            'keywords': [],
            'description': ''
        }
        
        # Look for numerical scores in the response
        import re
        
        response_lower = response.lower()
        
        # Primary patterns - look for exact format we requested
        exact_patterns = {
            'blur': r'blur\s+score:\s*(\d+(?:\.\d+)?)',
            'exposure': r'exposure\s+score:\s*(\d+(?:\.\d+)?)',
            'composition': r'composition\s+score:\s*(\d+(?:\.\d+)?)',
            'quality': r'overall\s+quality.*?score:\s*(\d+(?:\.\d+)?)'
        }
        
        # Also look for decimal patterns in analysis text
        analysis_patterns = {
            'blur': r'focus[\/\w\s]*[:\(\)]*\s*(\d\.\d+)',
            'exposure': r'exposure[\/\w\s]*[:\(\)]*\s*(\d\.\d+)', 
            'composition': r'composition[\/\w\s]*[:\(\)]*\s*(\d\.\d+)',
            'quality': r'overall.*?(\d\.\d+)[-\d\.]*'
        }
        
        # Try exact patterns first
        found_exact = False
        for category, pattern in exact_patterns.items():
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    score = float(matches[0])
                    # Ensure score is in 0-1 range
                    if score > 1:
                        score = score / 10.0 if score <= 10 else min(score / 100.0, 1.0)
                    scores[category] = max(0, min(1, score))
                    found_exact = True
                except ValueError:
                    continue
        
        # Try analysis patterns if exact patterns didn't work
        if not found_exact:
            for category, pattern in analysis_patterns.items():
                if scores[category] == 0.5:  # Only if not found yet
                    matches = re.findall(pattern, response_lower)
                    if matches:
                        try:
                            score = float(matches[0])
                            if score > 1:
                                score = score / 10.0 if score <= 10 else min(score / 100.0, 1.0)
                            scores[category] = max(0, min(1, score))
                        except ValueError:
                            continue
        
        # Fallback patterns if exact format wasn't found
        if not found_exact:
            fallback_patterns = {
                'blur': [r'blur.*?(\d+(?:\.\d+)?)', r'sharp.*?(\d+(?:\.\d+)?)', r'focus.*?(\d+(?:\.\d+)?)', r'sharpness.*?(\d+(?:\.\d+)?)'],
                'exposure': [r'exposure.*?(\d+(?:\.\d+)?)', r'lighting.*?(\d+(?:\.\d+)?)', r'brightness.*?(\d+(?:\.\d+)?)'],
                'composition': [r'composition.*?(\d+(?:\.\d+)?)', r'framing.*?(\d+(?:\.\d+)?)', r'balance.*?(\d+(?:\.\d+)?)'],
                'quality': [r'quality.*?(\d+(?:\.\d+)?)', r'overall.*?(\d+(?:\.\d+)?)', r'keep.*?(\d+(?:\.\d+)?)']
            }
            
            for category, pattern_list in fallback_patterns.items():
                if scores[category] == 0.5:  # Only if we haven't found a score yet
                    for pattern in pattern_list:
                        matches = re.findall(pattern, response_lower)
                        if matches:
                            try:
                                # Take the first reasonable match
                                score = float(matches[0])
                                if score > 1:
                                    score = score / 10.0 if score <= 10 else min(score / 100.0, 1.0)
                                scores[category] = max(0, min(1, score))
                                break
                            except ValueError:
                                continue
        
        # Parse description
        description_patterns = [
            r'description:\s*(.+?)(?:\n\w+:|$)',  # Stop at next field
            r'summary:\s*(.+?)(?:\n\w+:|$)',
        ]
        
        for pattern in description_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if matches:
                description = matches[0].strip()
                # Clean up the description
                description = ' '.join(description.split())  # Normalize whitespace
                description = description.rstrip('.,;!?')
                if description and len(description) > 10:  # Ensure it's meaningful
                    scores['description'] = description
                    break
        
        # Parse keywords with better handling for creative, multi-word keywords
        keyword_patterns = [
            r'keywords:\s*(.+?)(?:\n\n|\nblur|\nexposure|$)',  # Stop at next section
            r'tags:\s*(.+?)(?:\n\n|\nblur|\nexposure|$)',
            r'subjects:\s*(.+?)(?:\n\n|\nblur|\nexposure|$)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, response_lower, re.MULTILINE | re.DOTALL)
            if matches:
                # Take the first match and split by commas
                keyword_string = matches[0].strip()
                if keyword_string:
                    keywords = [kw.strip() for kw in keyword_string.split(',')]
                    # Clean up keywords
                    cleaned_keywords = []
                    for kw in keywords:
                        # Remove extra whitespace and clean up
                        kw = ' '.join(kw.split())  # Normalize whitespace
                        # Remove trailing punctuation
                        kw = kw.rstrip('.,;!?')
                        # Filter out empty or very short keywords
                        if kw and len(kw) > 2 and not kw.isdigit():
                            cleaned_keywords.append(kw)
                    
                    scores['keywords'] = cleaned_keywords[:15]  # Limit to 15 keywords
                    break
        
        # Log the parsing for debugging
        self.logger.debug(f"Parsed scores: {scores}")
        
        return scores
    
    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Analyze a single image"""
        return self.analyze_batch([image])[0]
    
    def analyze_batch(self, images: List[Image.Image]) -> List[ImageMetrics]:
        """Analyze multiple images"""
        metrics_list = []
        
        # COMPREHENSIVE SUBJECT-AWARE ANALYSIS PROMPT
        prompt = """You are analyzing a photograph for technical quality AND creating searchable metadata.

=== TECHNICAL ANALYSIS ===

FOCUS EVALUATION - This was shot with shallow DOF (f/1.8-f/2.8), background blur is INTENTIONAL:

1. IDENTIFY THE MAIN SUBJECT:
   - Portrait: Person's face/eyes
   - Animal: Pet's face/eyes  
   - Product: Key feature/text
   - Street: Main person/object
   - Landscape: Foreground element

2. EVALUATE SUBJECT SHARPNESS ONLY:
   - Portrait: Are the eyes tack sharp? Individual eyelashes visible?
   - Other subjects: Is the key detail crisp and well-defined?
   
   SCORING (0.0-1.0):
   1.0 = Perfect subject focus
   0.8 = Good professional quality
   0.6 = Acceptable but not ideal
   0.4 = Soft, needs review
   0.2 = Missed focus, delete
   0.0 = Completely wrong focus

EXPOSURE EVALUATION (0.0-1.0):
- Consider artistic intent, not just technical perfection
- 1.0: Beautiful light with full tonal range
- 0.8: Well-exposed with good detail
- 0.6: Minor over/under exposure, still usable
- 0.4: Noticeable exposure issues
- 0.2: Significant problems, recoverable
- 0.0: Blown highlights or blocked shadows, unusable

COMPOSITION EVALUATION (0.0-1.0):
- Visual impact and storytelling effectiveness
- 1.0: Compelling composition, draws the eye
- 0.8: Strong framing and balance
- 0.6: Good composition with minor issues
- 0.4: Acceptable but could be better
- 0.2: Poor framing or distracting elements
- 0.0: Bad composition, major problems

=== CONTENT ANALYSIS ===

DESCRIPTION: Write a natural 1-2 sentence description of what's happening in the image.

KEYWORDS: Generate 6-8 SPECIFIC, searchable keywords. Be DESCRIPTIVE and USEFUL:

GOOD EXAMPLES:
- "toddler exploring playground"
- "golden retriever catching frisbee"
- "sunset over mountain lake"
- "grandmother teaching cooking"
- "rain-soaked city street"
- "wedding ceremony outdoor"
- "macro dewdrops spider web"

BAD EXAMPLES (avoid these):
- "person" → use "young woman", "elderly man", "teenager"
- "outdoor" → use "forest path", "beach sunset", "urban rooftop"
- "animal" → use "tabby cat", "golden retriever", "red cardinal"
- "everyday life" → be specific about the activity

RESPOND IN EXACT FORMAT:
Description: [Natural description of what's happening]
Subject: [What/who is the main subject]
Focus point: [Where focus actually is]
Subject sharpness: [sharp/soft/missed]
Blur score: 0.XX
Exposure score: 0.XX
Composition score: 0.XX
Overall quality: 0.XX
Keywords: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6"""
        
        for image in images:
            try:
                # Convert image to base64
                image_base64 = self._image_to_base64(image)
                
                # Query Ollama
                response = self._query_ollama(prompt, image_base64)
                self.logger.debug(f"Ollama response: {response}")
                
                # Parse scores from response
                scores = self._parse_quality_scores(response)
                
                # Create metrics
                metrics = ImageMetrics(
                    blur_score=scores['blur'],
                    exposure_score=scores['exposure'],
                    composition_score=scores['composition'],
                    overall_quality=scores['quality'],
                    processing_mode=ProcessingMode.ACCURATE,
                    keywords=scores['keywords'],
                    description=scores['description']
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
                    processing_mode=ProcessingMode.ACCURATE,
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