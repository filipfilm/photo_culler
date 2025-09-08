import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Load and manage configuration"""
    
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML"""
        if not self.config_path.exists():
            return self._default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or self._default_config()
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "model": {
                "type": "ollama",
                "ollama_model": "llava:13b",
                "batch_size": 4
            },
            "thresholds": {
                "blur": {"delete": 0.15, "review": 0.30},  # Much more realistic - Ollama gives lower scores
                "exposure": {"delete": 0.25, "review": 0.45},  # More realistic
                "composition": {"delete": 0.15, "review": 0.35}  # More realistic
            },
            "subject_detection": {
                "enabled": True,
                "portrait_eye_threshold": 0.6,
                "center_weight": 0.3
            },
            "extensions": [".nef", ".cr2", ".arw", ".jpg", ".jpeg", ".dng", ".raf"],
            "cache": {
                "dir": "~/.cache/photo_culler",
                "preserve_days": 30
            }
        }
    
    def get(self, key: str, default=None):
        """Get config value by dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default