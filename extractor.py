from pathlib import Path
from PIL import Image
from typing import Optional
import logging
import io

# Try to import rawpy for proper RAW support
try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

class RawThumbnailExtractor:
    """Extract images from RAW and standard formats"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # RAW extensions
        self.raw_extensions = {'.nef', '.cr2', '.arw', '.dng', '.raf', 
                               '.orf', '.rw2', '.pef', '.srw', '.x3f'}
        
    def extract(self, filepath: Path) -> Optional[Image.Image]:
        """Extract/load image from file"""
        try:
            # Check if RAW file
            if filepath.suffix.lower() in self.raw_extensions:
                if RAWPY_AVAILABLE:
                    return self._extract_raw(filepath)
                else:
                    self.logger.warning(f"rawpy not installed, skipping RAW file {filepath.name}")
                    return None
            
            # Standard formats
            return Image.open(filepath)
                
        except Exception as e:
            self.logger.error(f"Failed to extract image from {filepath}: {e}")
            return None
    
    def _extract_raw(self, filepath: Path) -> Optional[Image.Image]:
        """Extract image from RAW file using rawpy"""
        try:
            with rawpy.imread(str(filepath)) as raw:
                # Try embedded JPEG first (fastest)
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(io.BytesIO(thumb.data))
                except:
                    pass
                
                # Fall back to processing (slower but always works)
                # Use half_size for speed, still plenty of resolution for analysis
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    no_auto_bright=False,
                    output_bps=8
                )
                return Image.fromarray(rgb)
                
        except Exception as e:
            self.logger.error(f"Failed to process RAW file {filepath}: {e}")
            return None