"""
Dynamic Model Selection for Ollama Optimization
Chooses the optimal Ollama model based on image content and system resources
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List
import logging
import psutil
import requests
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an Ollama model"""
    name: str
    ram_requirement_gb: float
    processing_speed: float  # Relative speed (higher = faster)
    quality_score: float  # Quality of analysis (higher = better)
    best_for: List[str]  # What this model is optimized for


class DynamicModelSelector:
    """Select optimal Ollama model based on image content and system resources"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        
        # Define available models and their characteristics
        self.models = {
            "llava:7b": ModelConfig(
                name="llava:7b",
                ram_requirement_gb=6,
                processing_speed=3.0,
                quality_score=2.5,
                best_for=["general", "landscapes", "objects", "fast_processing"]
            ),
            "llava:13b": ModelConfig(
                name="llava:13b", 
                ram_requirement_gb=10,
                processing_speed=2.0,
                quality_score=3.5,
                best_for=["portraits", "detailed_analysis", "complex_scenes", "quality_focus"]
            ),
            "llava:34b": ModelConfig(
                name="llava:34b",
                ram_requirement_gb=24,
                processing_speed=1.0,
                quality_score=4.5,
                best_for=["professional_analysis", "fine_art", "commercial_work"]
            ),
            "bakllava": ModelConfig(
                name="bakllava",
                ram_requirement_gb=5,
                processing_speed=3.5,
                quality_score=2.0,
                best_for=["speed_priority", "batch_processing", "quick_triage"]
            ),
            "moondream": ModelConfig(
                name="moondream",
                ram_requirement_gb=3,
                processing_speed=4.0,
                quality_score=1.8,
                best_for=["low_memory", "mobile", "basic_analysis"]
            )
        }
        
        # Cache for image analysis results
        self._analysis_cache = {}
        
        # Get available models from Ollama
        self.available_models = self._get_available_models()
        
        # Filter models to only those available
        self.models = {k: v for k, v in self.models.items() 
                      if k in self.available_models}
        
        if not self.models:
            self.logger.warning("No known models available, falling back to default")
            # Add a fallback default
            self.models["llava:13b"] = self.models.get("llava:13b", ModelConfig(
                name="llava:13b", ram_requirement_gb=10, processing_speed=2.0,
                quality_score=3.5, best_for=["general"]
            ))
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available = [model['name'] for model in models_data.get('models', [])]
                self.logger.info(f"Available Ollama models: {available}")
                return available
            else:
                self.logger.warning(f"Could not get models from Ollama: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama: {e}")
        
        return []
    
    def select_model(self, image: Image.Image, 
                    processing_mode: str = "quality",
                    batch_size: int = 1) -> str:
        """Choose best model for this image and processing context"""
        
        # Quick analysis for model selection
        analysis = self._analyze_image_content(image)
        
        # Get system resources
        system_resources = self._get_system_resources()
        
        # Score each available model
        model_scores = {}
        
        for model_name, model_config in self.models.items():
            score = self._score_model(model_config, analysis, system_resources, 
                                    processing_mode, batch_size)
            model_scores[model_name] = score
            
            self.logger.debug(f"Model {model_name} score: {score:.2f}")
        
        if not model_scores:
            self.logger.warning("No models available, using default")
            return "llava:13b"
        
        # Select model with highest score
        best_model = max(model_scores.keys(), key=lambda x: model_scores[x])
        
        self.logger.info(f"Selected model: {best_model} (score: {model_scores[best_model]:.2f})")
        
        return best_model
    
    def _analyze_image_content(self, image: Image.Image) -> Dict:
        """Quick analysis of image content for model selection"""
        
        # Create cache key
        image_array = np.array(image)
        cache_key = f"{image_array.shape}_{np.mean(image_array):.2f}"
        
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        analysis = {}
        
        try:
            # Resize for quick analysis
            analysis_image = image.copy()
            analysis_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            img_array = np.array(analysis_image)
            
            # Basic image properties
            height, width = img_array.shape[:2]
            analysis['aspect_ratio'] = width / height
            analysis['resolution'] = width * height
            
            # Convert to different color spaces for analysis
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            
            # Detect potential content types
            analysis.update(self._detect_content_type(rgb, gray, hsv))
            
            # Analyze complexity
            analysis['complexity'] = self._analyze_complexity(gray)
            
            # Analyze color richness
            analysis['color_richness'] = self._analyze_colors(hsv)
            
            # Cache result
            self._analysis_cache[cache_key] = analysis
            
        except Exception as e:
            self.logger.warning(f"Image analysis failed: {e}")
            analysis = {'content_type': 'unknown', 'complexity': 0.5, 'color_richness': 0.5}
        
        return analysis
    
    def _detect_content_type(self, rgb: np.ndarray, gray: np.ndarray, 
                           hsv: np.ndarray) -> Dict:
        """Detect what type of content is in the image"""
        content_info = {}
        
        # Face detection for portraits
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            content_info['has_faces'] = len(faces) > 0
            content_info['face_count'] = len(faces)
            
            # Determine if portrait-like
            if len(faces) > 0:
                # Check if faces occupy significant portion of image
                total_face_area = sum(w * h for (x, y, w, h) in faces)
                image_area = gray.shape[0] * gray.shape[1]
                face_coverage = total_face_area / image_area
                content_info['is_portrait'] = face_coverage > 0.1
            else:
                content_info['is_portrait'] = False
                
        except Exception:
            content_info.update({'has_faces': False, 'face_count': 0, 'is_portrait': False})
        
        # Detect sky (blue areas in upper portion)
        try:
            upper_third = hsv[:hsv.shape[0]//3, :]
            blue_mask = cv2.inRange(upper_third, (100, 50, 50), (130, 255, 255))
            sky_coverage = np.sum(blue_mask > 0) / blue_mask.size
            content_info['has_sky'] = sky_coverage > 0.2
        except Exception:
            content_info['has_sky'] = False
        
        # Detect green areas (landscapes/nature)
        try:
            green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
            green_coverage = np.sum(green_mask > 0) / green_mask.size
            content_info['has_nature'] = green_coverage > 0.3
        except Exception:
            content_info['has_nature'] = False
        
        # Determine primary content type
        if content_info.get('is_portrait', False):
            content_info['content_type'] = 'portrait'
        elif content_info.get('has_sky', False) and content_info.get('has_nature', False):
            content_info['content_type'] = 'landscape'
        elif content_info.get('has_nature', False):
            content_info['content_type'] = 'nature'
        else:
            content_info['content_type'] = 'general'
        
        return content_info
    
    def _analyze_complexity(self, gray: np.ndarray) -> float:
        """Analyze image complexity (0=simple, 1=complex)"""
        try:
            # Edge density as complexity measure
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Local variance as texture measure
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_complexity = np.mean(local_variance) / 255**2
            
            # Combine metrics
            complexity = (edge_density * 5 + texture_complexity) / 2
            return min(complexity, 1.0)
            
        except Exception:
            return 0.5  # Default medium complexity
    
    def _analyze_colors(self, hsv: np.ndarray) -> float:
        """Analyze color richness (0=monochrome, 1=very colorful)"""
        try:
            # Saturation analysis
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation) / 255
            
            # Color variance
            hue = hsv[:, :, 2]
            hue_std = np.std(hue) / 255
            
            # Combine metrics
            color_richness = (avg_saturation + hue_std) / 2
            return min(color_richness, 1.0)
            
        except Exception:
            return 0.5  # Default medium color richness
    
    def _get_system_resources(self) -> Dict:
        """Get current system resource availability"""
        try:
            # Get RAM info
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            
            # Get CPU info
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Estimate GPU availability (basic check)
            gpu_available = self._check_gpu_availability()
            
            return {
                'available_ram_gb': available_ram_gb,
                'cpu_count': cpu_count,
                'cpu_usage': cpu_usage,
                'gpu_available': gpu_available
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get system resources: {e}")
            return {
                'available_ram_gb': 8.0,  # Conservative default
                'cpu_count': 4,
                'cpu_usage': 50.0,
                'gpu_available': False
            }
    
    def _check_gpu_availability(self) -> bool:
        """Basic check for GPU availability"""
        try:
            # Try to detect NVIDIA GPU
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            pass
        
        try:
            # Try to detect Apple Silicon GPU
            import platform
            if platform.system() == 'Darwin' and platform.processor() == 'arm':
                return True
        except:
            pass
        
        return False
    
    def _score_model(self, model: ModelConfig, analysis: Dict, 
                    resources: Dict, processing_mode: str, batch_size: int) -> float:
        """Score a model for the given context"""
        
        score = 0.0
        
        # Resource compatibility (heavily weighted)
        if model.ram_requirement_gb <= resources['available_ram_gb']:
            score += 3.0  # Can run this model
            
            # Bonus for not using all available RAM (leaves room for other processes)
            ram_usage_ratio = model.ram_requirement_gb / resources['available_ram_gb']
            if ram_usage_ratio < 0.7:  # Using less than 70% of RAM
                score += 1.0
        else:
            # Penalize models that require more RAM than available
            score -= 5.0
        
        # Content type compatibility
        content_type = analysis.get('content_type', 'general')
        if content_type in model.best_for:
            score += 2.0
        
        # Processing mode preferences
        if processing_mode == "speed":
            score += model.processing_speed * 0.8
        elif processing_mode == "quality":
            score += model.quality_score * 0.8
        else:  # balanced
            score += (model.processing_speed + model.quality_score) * 0.4
        
        # Batch processing considerations
        if batch_size > 10:
            # For large batches, prioritize speed
            score += model.processing_speed * 0.5
        elif batch_size == 1:
            # For single images, quality matters more
            score += model.quality_score * 0.3
        
        # Image complexity considerations
        complexity = analysis.get('complexity', 0.5)
        if complexity > 0.7:  # Complex image
            score += model.quality_score * 0.3  # Favor quality models
        elif complexity < 0.3:  # Simple image
            score += model.processing_speed * 0.3  # Speed is fine
        
        # Portrait-specific scoring
        if analysis.get('is_portrait', False):
            if 'portraits' in model.best_for:
                score += 1.5
            # Portraits benefit from higher quality models
            score += model.quality_score * 0.2
        
        # System load considerations
        cpu_usage = resources.get('cpu_usage', 50)
        if cpu_usage > 80:  # High system load
            score += model.processing_speed * 0.3  # Prefer faster models
        
        return score
    
    def get_model_recommendations(self, images: List[Image.Image], 
                                processing_mode: str = "balanced") -> Dict[str, str]:
        """Get model recommendations for a batch of images"""
        
        recommendations = {}
        model_usage = {}
        
        # Analyze each image and get recommendation
        for i, image in enumerate(images):
            model = self.select_model(image, processing_mode, len(images))
            recommendations[f"image_{i}"] = model
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Determine if we should use a single model for consistency
        most_common_model = max(model_usage.keys(), key=lambda x: model_usage[x])
        
        # If one model is recommended for >60% of images, use it for all
        if model_usage[most_common_model] / len(images) > 0.6:
            unified_recommendation = most_common_model
        else:
            unified_recommendation = None
        
        return {
            'individual_recommendations': recommendations,
            'model_usage_stats': model_usage,
            'unified_recommendation': unified_recommendation,
            'total_models_needed': len(model_usage)
        }
    
    def get_optimal_concurrent_instances(self, selected_model: str) -> int:
        """Calculate optimal number of concurrent Ollama instances"""
        
        if selected_model not in self.models:
            return 1
        
        model = self.models[selected_model]
        resources = self._get_system_resources()
        
        # Calculate based on RAM availability
        available_ram = resources['available_ram_gb']
        ram_per_instance = model.ram_requirement_gb
        
        # Leave 2GB for system
        usable_ram = max(available_ram - 2, ram_per_instance)
        max_instances_by_ram = int(usable_ram / ram_per_instance)
        
        # Calculate based on CPU cores
        cpu_cores = resources['cpu_count']
        max_instances_by_cpu = max(1, cpu_cores // 2)  # 2 cores per instance
        
        # Take the minimum and cap at reasonable limits
        optimal_instances = min(max_instances_by_ram, max_instances_by_cpu, 8)
        
        self.logger.info(f"Optimal concurrent instances for {selected_model}: {optimal_instances}")
        self.logger.debug(f"RAM limit: {max_instances_by_ram}, CPU limit: {max_instances_by_cpu}")
        
        return max(1, optimal_instances)