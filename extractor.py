from pathlib import Path
from PIL import Image
from typing import Optional
import logging

class RawThumbnailExtractor:
    """Simple image extractor that works with most formats"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
    def extract(self, filepath: Path) -> Optional[Image.Image]:
        """Extract/load image from file"""
        try:
            # For most formats, PIL can handle directly
            if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                return Image.open(filepath)
            
            # For RAW files, try to open directly (PIL might have support)
            # or fall back to basic loading
            try:
                return Image.open(filepath)
            except Exception:
                self.logger.warning(f"Could not process RAW file {filepath.name}, skipping")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract image from {filepath}: {e}")
            return None