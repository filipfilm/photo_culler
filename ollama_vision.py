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
                 timeout: int = 180):
        
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


class ImprovedOllamaVisionAnalyzer:
    """
    Enhanced Ollama analyzer that uses structured JSON output for more reliable parsing
    and integrates with enhanced focus analysis
    """
    
    def __init__(self, 
                 model: str = "llava:13b",
                 host: str = "http://localhost:11434",
                 timeout: int = 180):
        
        self.model = model
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🐛 ImprovedOllamaVisionAnalyzer.__init__ received host={host}, using self.host={self.host}")
        
        # Test connection and model availability
        self._check_ollama_connection()
        self._ensure_model_available()
        
        # Enhanced focus analyzer for integration
        try:
            from enhanced_focus_analyzer import EnhancedFocusAnalyzer
            self.enhanced_focus = EnhancedFocusAnalyzer()
            self.logger.info("Enhanced focus analyzer integrated successfully")
        except ImportError:
            self.enhanced_focus = None
            self.logger.warning("Enhanced focus analyzer not available")
    
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
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
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
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent output
                    "top_p": 0.9
                }
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
    
    def _parse_structured_response(self, response: str) -> Dict:
        """Parse structured JSON response from Ollama with fallback parsing"""
        
        # Log the response for debugging
        self.logger.debug(f"Parsing Ollama response (length: {len(response)})")
        self.logger.debug(f"Response starts with: {response[:200]}...")
        
        # First try to extract JSON from the response
        import re
        
        # Look for JSON block in response - try multiple patterns
        json_patterns = [
            r'\{[\s\S]*\}',  # Basic JSON block
            r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{[\s\S]*?\})\s*```',  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response)
            if json_match:
                try:
                    json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
                    self.logger.debug(f"Found potential JSON: {json_str[:200]}...")
                    
                    parsed = json.loads(json_str)
                    
                    # Validate required fields
                    if all(key in parsed for key in ['blur_score', 'exposure_score', 'composition_score', 'overall_quality']):
                        
                        # Check for placeholder text in description and keywords
                        description = parsed.get('description', '')
                        keywords = parsed.get('keywords', [])
                        
                        if ('vivid, engaging description' in description.lower() or 
                            'tells the story' in description.lower() or
                            any('creative keyword' in str(kw).lower() for kw in keywords)):
                            self.logger.warning("JSON contains placeholder text, falling back to text parsing")
                            continue
                        
                        # Ensure scores are in 0-1 range and convert from arrays if needed
                        for score_key in ['blur_score', 'exposure_score', 'composition_score', 'overall_quality']:
                            if score_key in parsed:
                                score = parsed[score_key]
                                
                                # Handle case where model returns arrays instead of numbers
                                if isinstance(score, list) and len(score) > 0:
                                    score = score[0]
                                
                                if isinstance(score, (int, float)):
                                    if score > 1:
                                        score = min(score / 10.0 if score <= 10 else score / 100.0, 1.0)
                                    parsed[score_key] = max(0.0, min(1.0, float(score)))
                                else:
                                    # Fallback to default if we can't parse the score
                                    parsed[score_key] = 0.5
                        
                        self.logger.debug("Successfully parsed structured JSON response")
                        return parsed
                    else:
                        self.logger.warning(f"JSON missing required fields. Found keys: {list(parsed.keys())}")
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON: {e}")
                    continue
        
        # Fallback to text parsing if JSON parsing fails
        return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict:
        """Fallback text parsing for when JSON parsing fails"""
        import re
        
        # Log the actual response to debug parsing issues
        self.logger.debug(f"Falling back to text parsing. Response length: {len(response)}")
        self.logger.debug(f"Response preview: {response[:500]}...")
        
        result = {
            'blur_score': 0.5,
            'exposure_score': 0.5,
            'composition_score': 0.5,
            'overall_quality': 0.5,
            'description': '',
            'keywords': [],
            'subject_type': 'general',
            'focus_assessment': 'unknown'
        }
        
        # Skip parsing if the response looks like it contains our prompt template
        if "creative keyword" in response.lower() or "vivid, engaging description" in response.lower():
            self.logger.warning("Response appears to contain prompt template, using defaults")
            return result
        
        response_lower = response.lower()
        
        # Score patterns
        score_patterns = {
            'blur_score': [r'blur[_\s]*score[:\s]*(\d*\.?\d+)', r'focus[_\s]*score[:\s]*(\d*\.?\d+)', r'sharpness[:\s]*(\d*\.?\d+)'],
            'exposure_score': [r'exposure[_\s]*score[:\s]*(\d*\.?\d+)', r'lighting[_\s]*score[:\s]*(\d*\.?\d+)'],
            'composition_score': [r'composition[_\s]*score[:\s]*(\d*\.?\d+)', r'framing[_\s]*score[:\s]*(\d*\.?\d+)'],
            'overall_quality': [r'overall[_\s]*(?:quality[_\s]*)?score[:\s]*(\d*\.?\d+)', r'total[_\s]*score[:\s]*(\d*\.?\d+)']
        }
        
        for score_key, patterns in score_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, response_lower)
                if matches:
                    try:
                        score = float(matches[0])
                        if score > 1:
                            score = score / 10.0 if score <= 10 else min(score / 100.0, 1.0)
                        result[score_key] = max(0.0, min(1.0, score))
                        break
                    except ValueError:
                        continue
        
        # Parse description
        desc_patterns = [
            r'description[:\s]*([^\n]+)',
            r'summary[:\s]*([^\n]+)',
            r'this\s+(?:image|photo|picture)\s+shows?\s*([^\n\.]+)'
        ]
        
        for pattern in desc_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                desc = matches[0].strip().rstrip('.,;')
                if len(desc) > 10:  # Meaningful description
                    result['description'] = desc
                    break
        
        # Parse keywords
        keyword_patterns = [
            r'keywords?[:\s]*([^\n]+)',
            r'tags?[:\s]*([^\n]+)',
            r'subjects?[:\s]*([^\n]+)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                keyword_str = matches[0].strip()
                keywords = [kw.strip().rstrip('.,;') for kw in keyword_str.split(',')]
                keywords = [kw for kw in keywords if len(kw) > 2 and not kw.isdigit()]
                if keywords:
                    result['keywords'] = keywords[:10]
                    break
        
        return result
    
    def analyze(self, image: Image.Image) -> ImageMetrics:
        """Analyze a single image with enhanced focus integration"""
        
        # Get enhanced focus analysis if available
        enhanced_focus_data = None
        if self.enhanced_focus:
            try:
                enhanced_focus_data = self.enhanced_focus.analyze_focus(image)
                self.logger.debug(f"Enhanced focus analysis: {enhanced_focus_data}")
            except Exception as e:
                self.logger.warning(f"Enhanced focus analysis failed: {e}")
        
        # Structured prompt that asks for JSON response
        prompt = self._get_structured_analysis_prompt(enhanced_focus_data)
        
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Query Ollama
            response = self._query_ollama(prompt, image_base64)
            self.logger.debug(f"Ollama response: {response}")
            
            # Parse structured response
            parsed_result = self._parse_structured_response(response)
            
            # Integrate with enhanced focus data if available
            if enhanced_focus_data:
                # Adjust blur score based on enhanced focus analysis
                cv_focus_score = enhanced_focus_data.get('focus_score', 0.5)
                vision_blur_score = parsed_result.get('blur_score', 0.5)
                
                # Weighted combination: 60% vision model, 40% CV analysis
                combined_blur_score = (vision_blur_score * 0.6 + cv_focus_score * 0.4)
                parsed_result['blur_score'] = combined_blur_score
                
                # Add enhanced focus data to result
                parsed_result['enhanced_focus'] = enhanced_focus_data
            
            # Create metrics object
            metrics = ImageMetrics(
                blur_score=parsed_result.get('blur_score', 0.5),
                exposure_score=parsed_result.get('exposure_score', 0.5),
                composition_score=parsed_result.get('composition_score', 0.5),
                overall_quality=parsed_result.get('overall_quality', 0.5),
                processing_mode=ProcessingMode.ACCURATE,
                keywords=parsed_result.get('keywords', []),
                description=parsed_result.get('description', ''),
                enhanced_focus=enhanced_focus_data
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze image: {e}")
            # Fallback to neutral scores
            return ImageMetrics(
                blur_score=0.5,
                exposure_score=0.5,
                composition_score=0.5,
                overall_quality=0.5,
                processing_mode=ProcessingMode.ACCURATE,
                keywords=[],
                description="",
                enhanced_focus=enhanced_focus_data
            )
    
    def _get_structured_analysis_prompt(self, enhanced_focus_data: Optional[Dict] = None) -> str:
        """Generate enhanced analysis prompt with creative and technical expertise"""
        
        focus_context = ""
        if enhanced_focus_data:
            subject_type = enhanced_focus_data.get('subject_type', 'general')
            is_shallow_dof = enhanced_focus_data.get('is_shallow_dof', False)
            subject_sharpness = enhanced_focus_data.get('subject_sharpness', 0.5)
            
            focus_context = f"""
COMPUTER VISION FOCUS ANALYSIS:
- Subject type: {subject_type}
- Shallow DOF detected: {is_shallow_dof}
- CV subject sharpness: {subject_sharpness:.2f}
- Focus regions detected: {len(enhanced_focus_data.get('focus_regions', []))}

Use this CV analysis to inform your focus scoring, but rely primarily on your visual assessment.
"""
        
        prompt = f"""You are a master photographer and photo editor with expertise in technical analysis, artistic composition, and creative storytelling through images.

{focus_context}

Analyze this image with the eye of a professional who understands both technical excellence and artistic vision.

Analyze the image and respond with EXACTLY this JSON structure (replace ALL placeholder values with your actual analysis):

{{
    "blur_score": 0.XX,
    "exposure_score": 0.XX, 
    "composition_score": 0.XX,
    "overall_quality": 0.XX,
    "description": "Your actual description of this specific image",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
    "subject_type": "portrait",
    "focus_assessment": "acceptable"
}}

IMPORTANT: Use actual decimal numbers (like 0.75) not arrays or text for the scores.

═══ TECHNICAL ANALYSIS GUIDELINES ═══

🎯 FOCUS/BLUR SCORE (0.0-1.0):
Look closely at the intended subject:
• Portrait: Are the nearest eye's eyelashes individually visible? Can you see iris texture?
• Macro: Is the key detail (stamens, dewdrops, etc.) razor sharp with visible micro-texture?
• Landscape: Is the foreground element or hyperfocal point critically sharp?
• Street: Is the main subject (person, sign, etc.) well-defined with clear edges?

FOCUS SCORING:
• 1.0: Tack sharp - you can see individual hairs, fabric weave, eye detail
• 0.9: Critically sharp - professional quality, perfect for large prints
• 0.8: Very sharp - excellent sharpness, minor softness only under magnification
• 0.7: Sharp - good quality, suitable for most uses
• 0.6: Acceptably sharp - decent but not ideal
• 0.5: Questionably sharp - noticeable softness, needs review
• 0.4: Soft - clearly not optimal focus
• 0.3: Poor focus - significantly soft, likely unusable
• 0.2: Very poor - major focus miss
• 0.0-0.1: Completely out of focus

💡 EXPOSURE SCORE (0.0-1.0) - Artistic Understanding:
Consider the photographer's intent and the story being told:

HIGH SCORES (0.8-1.0):
• Golden hour portraits with warm, wraparound light
• Dramatic landscapes with preserved highlight detail in skies
• Moody low-key images where shadows enhance the story
• High-key bright images that feel intentionally ethereal
• Studio portraits with controlled, flattering light

MEDIUM SCORES (0.5-0.7):
• Available light portraits with good skin tones
• Outdoor scenes with slight highlight clipping but good overall balance
• Creative silhouettes or backlit subjects
• Documentary-style images with natural, unforced lighting

LOW SCORES (0.0-0.4):
• Blown highlights destroying important detail (wedding dress, sky, etc.)
• Blocked shadows hiding crucial information
• Unflattering fluorescent or harsh overhead lighting
• Color casts that detract from the image
• Technical exposure failures that can't be artistic intent

🎨 COMPOSITION SCORE (0.0-1.0) - Visual Impact:
Evaluate like a gallery curator:

EXCEPTIONAL (0.9-1.0):
• Dynamic leading lines that draw the eye powerfully
• Perfect rule of thirds with compelling subject placement
• Negative space used masterfully to create impact
• Layered depth that tells a complex visual story
• Balanced asymmetry that creates tension and interest
• Color harmony that enhances the emotional message

STRONG (0.7-0.8):
• Good use of compositional rules (thirds, golden ratio)
• Effective framing that excludes distractions
• Nice depth of field choices for the story
• Pleasant color relationships
• Clear main subject with good supporting elements

AVERAGE (0.4-0.6):
• Basic centering or rule of thirds
• Some distracting elements but manageable
• Adequate but uninspiring framing
• Missing opportunities for stronger composition

POOR (0.0-0.3):
• Tilted horizons in landscapes
• Cluttered backgrounds destroying the subject
• Trees/poles "growing" out of people's heads
• Harsh crop lines through joints
• Competing subjects fighting for attention
• Distracting foreground elements

═══ CREATIVE CONTENT ANALYSIS ═══

📝 DESCRIPTION:
Write like a storyteller, not a technical manual. Focus on:
• The emotion or mood you feel
• The story the image tells
• What makes this moment special
• The photographer's artistic vision

Good: "A toddler's face lights up with pure joy as soap bubbles float around her in golden afternoon light"
Bad: "A young child with bubbles in outdoor setting"

🏷️ KEYWORDS - Be Creative and Specific:
Think like someone searching for the perfect image for a magazine, blog, or art collection.

EMOTION & MOOD KEYWORDS:
• "pure joy", "quiet contemplation", "raw emotion", "peaceful solitude"
• "dramatic tension", "nostalgic warmth", "urban energy", "serene beauty"

SPECIFIC DESCRIPTORS:
Instead of: "dog" → Use: "golden retriever puppy", "weathered old mastiff", "playful border collie"
Instead of: "car" → Use: "vintage mustang", "sleek tesla roadster", "rusty farm truck"  
Instead of: "building" → Use: "art deco skyscraper", "weathered barn", "modern glass office tower"

ARTISTIC & TECHNICAL STYLE:
• "shallow depth of field", "dramatic chiaroscuro lighting", "film noir mood"
• "golden hour glow", "misty atmospheric", "high contrast black and white"
• "candid street photography", "studio portrait lighting", "dreamy soft focus"

ACTIVITY & CONTEXT:
• "grandmother teaching cooking", "surfer catching dawn waves", "musician lost in performance"
• "rainy day coffee shop", "bustling farmers market", "quiet library study session"

SEASONAL & ENVIRONMENTAL:
• "autumn leaves swirling", "fresh morning frost", "summer festival crowd"
• "desert sunset silhouette", "coastal storm approaching", "mountain peak reflection"

═══ FINAL REMINDER ═══
Focus scoring: Ignore background blur if it's intentional shallow DOF. Judge only the main subject's sharpness.
Exposure scoring: Consider artistic intent, not just technical perfection.
Composition scoring: Think about visual impact and storytelling power.
Keywords: Be creative, specific, and evocative - help people find the perfect image for their story.

CRITICAL: Do NOT copy the placeholder text! You must analyze THIS SPECIFIC IMAGE and provide real, unique descriptions and keywords based on what you actually see in the photo.

Respond with ONLY the JSON structure, no additional text."""

        return prompt
    
    def analyze_batch(self, images: List[Image.Image]) -> List[ImageMetrics]:
        """Analyze multiple images"""
        return [self.analyze(image) for image in images]


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