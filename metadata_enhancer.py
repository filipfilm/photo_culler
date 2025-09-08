"""
Enhanced metadata handling with comprehensive IPTC/EXIF support
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import piexif
from datetime import datetime
from typing import Dict, Optional, List, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json


@dataclass
class CameraSettings:
    """Camera settings extracted from EXIF"""
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[str] = None
    focal_length_35mm: Optional[str] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None
    iso: Optional[int] = None
    flash: Optional[str] = None
    focus_mode: Optional[str] = None
    metering_mode: Optional[str] = None
    white_balance: Optional[str] = None
    exposure_compensation: Optional[str] = None
    exposure_program: Optional[str] = None
    scene_mode: Optional[str] = None
    image_stabilization: Optional[str] = None


@dataclass 
class LocationData:
    """GPS and location information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    direction: Optional[float] = None
    location_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


@dataclass
class ImageMetadata:
    """Complete image metadata container"""
    # Technical metadata
    camera_settings: CameraSettings
    location: LocationData
    
    # Timestamps
    date_taken: Optional[datetime] = None
    date_modified: Optional[datetime] = None
    
    # Image properties
    width: Optional[int] = None
    height: Optional[int] = None
    color_space: Optional[str] = None
    orientation: Optional[int] = None
    
    # Content metadata
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    rating: Optional[int] = None
    
    # Copyright and attribution
    artist: Optional[str] = None
    copyright: Optional[str] = None
    
    # Additional EXIF data
    software: Optional[str] = None
    lens_serial: Optional[str] = None
    camera_serial: Optional[str] = None


class MetadataEnhancer:
    """Enhanced metadata handling with IPTC support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_comprehensive_metadata(self, filepath: Path) -> ImageMetadata:
        """Extract comprehensive metadata from image file"""
        try:
            with Image.open(filepath) as img:
                # Extract EXIF data
                exifdata = img.getexif()
                
                # Extract camera settings
                camera_settings = self._extract_camera_settings(exifdata)
                
                # Extract GPS data
                location = self._extract_location_data(exifdata)
                
                # Extract timestamps
                date_taken = self._extract_date_taken(exifdata)
                
                # Extract image properties
                width, height = img.size
                orientation = exifdata.get(0x0112, 1)  # Orientation tag
                color_space = self._get_color_space(img)
                
                # Extract content metadata
                description = self._extract_description(exifdata)
                keywords = self._extract_keywords(exifdata)
                artist = exifdata.get(0x013B)  # Artist tag
                copyright_info = exifdata.get(0x8298)  # Copyright tag
                software = exifdata.get(0x0131)  # Software tag
                
                return ImageMetadata(
                    camera_settings=camera_settings,
                    location=location,
                    date_taken=date_taken,
                    date_modified=datetime.fromtimestamp(filepath.stat().st_mtime),
                    width=width,
                    height=height,
                    color_space=color_space,
                    orientation=orientation,
                    description=description,
                    keywords=keywords,
                    artist=artist,
                    copyright=copyright_info,
                    software=software
                )
                
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {filepath}: {e}")
            return ImageMetadata(
                camera_settings=CameraSettings(),
                location=LocationData(),
                date_modified=datetime.fromtimestamp(filepath.stat().st_mtime) if filepath.exists() else None
            )
    
    def _extract_camera_settings(self, exifdata: Image.Exif) -> CameraSettings:
        """Extract detailed camera settings from EXIF"""
        settings = CameraSettings()
        
        try:
            # Basic camera info
            settings.make = exifdata.get(0x010F)  # Make
            settings.model = exifdata.get(0x0110)  # Model
            settings.lens_model = exifdata.get(0xA434)  # LensModel
            
            # Focal length
            focal_length = exifdata.get(0x920A)  # FocalLength
            if focal_length:
                if isinstance(focal_length, tuple):
                    settings.focal_length = f"{focal_length[0]/focal_length[1]:.1f}mm"
                else:
                    settings.focal_length = f"{focal_length}mm"
            
            # 35mm equivalent focal length
            focal_35mm = exifdata.get(0xA405)  # FocalLengthIn35mmFilm
            if focal_35mm:
                settings.focal_length_35mm = f"{focal_35mm}mm"
            
            # Aperture
            aperture = exifdata.get(0x829D)  # FNumber
            if aperture:
                if isinstance(aperture, tuple):
                    f_num = aperture[0] / aperture[1]
                    settings.aperture = f"f/{f_num:.1f}"
                else:
                    settings.aperture = f"f/{aperture:.1f}"
            
            # Shutter speed
            shutter = exifdata.get(0x829A)  # ExposureTime
            if shutter:
                if isinstance(shutter, tuple):
                    exposure_time = shutter[0] / shutter[1]
                    if exposure_time >= 1:
                        settings.shutter_speed = f"{exposure_time:.1f}s"
                    else:
                        settings.shutter_speed = f"1/{int(1/exposure_time)}s"
                else:
                    if shutter >= 1:
                        settings.shutter_speed = f"{shutter:.1f}s"
                    else:
                        settings.shutter_speed = f"1/{int(1/shutter)}s"
            
            # ISO
            iso = exifdata.get(0x8827)  # ISOSpeedRatings
            if iso:
                settings.iso = int(iso) if isinstance(iso, (int, float)) else iso
            
            # Flash
            flash = exifdata.get(0x9209)  # Flash
            if flash is not None:
                settings.flash = 'Fired' if flash & 1 else 'No Flash'
                if flash & 0x40:  # Red-eye reduction
                    settings.flash += ' (Red-eye reduction)'
            
            # Metering mode
            metering = exifdata.get(0x9207)  # MeteringMode
            if metering is not None:
                metering_modes = {
                    0: 'Unknown', 1: 'Average', 2: 'Center-weighted', 
                    3: 'Spot', 4: 'Multi-spot', 5: 'Pattern', 6: 'Partial'
                }
                settings.metering_mode = metering_modes.get(metering, f'Unknown ({metering})')
            
            # White balance
            wb = exifdata.get(0x9208)  # WhiteBalance
            if wb is not None:
                settings.white_balance = 'Auto' if wb == 0 else 'Manual'
            
            # Exposure compensation
            exp_comp = exifdata.get(0x9204)  # ExposureBiasValue
            if exp_comp is not None:
                if isinstance(exp_comp, tuple):
                    comp_value = exp_comp[0] / exp_comp[1]
                    settings.exposure_compensation = f"{comp_value:+.1f} EV"
                else:
                    settings.exposure_compensation = f"{exp_comp:+.1f} EV"
            
            # Exposure program
            exp_prog = exifdata.get(0x8822)  # ExposureProgram
            if exp_prog is not None:
                programs = {
                    0: 'Not Defined', 1: 'Manual', 2: 'Program AE', 
                    3: 'Aperture Priority', 4: 'Shutter Priority', 
                    5: 'Creative Program', 6: 'Action Program', 
                    7: 'Portrait Mode', 8: 'Landscape Mode'
                }
                settings.exposure_program = programs.get(exp_prog, f'Unknown ({exp_prog})')
            
            # Scene mode
            scene = exifdata.get(0xA403)  # SceneCaptureType
            if scene is not None:
                scene_types = {0: 'Standard', 1: 'Landscape', 2: 'Portrait', 3: 'Night Scene'}
                settings.scene_mode = scene_types.get(scene, f'Unknown ({scene})')
            
        except Exception as e:
            self.logger.warning(f"Error extracting camera settings: {e}")
        
        return settings
    
    def _extract_location_data(self, exifdata: Image.Exif) -> LocationData:
        """Extract GPS and location data from EXIF"""
        location = LocationData()
        
        try:
            # Get GPS info if available
            gps_info = exifdata.get_ifd(0x8825)  # GPS IFD
            
            if gps_info:
                # Extract latitude
                lat_ref = gps_info.get(1)  # GPSLatitudeRef (N/S)
                lat = gps_info.get(2)      # GPSLatitude
                
                if lat_ref and lat:
                    latitude = self._convert_gps_coordinate(lat)
                    if lat_ref == 'S':
                        latitude = -latitude
                    location.latitude = latitude
                
                # Extract longitude
                lon_ref = gps_info.get(3)  # GPSLongitudeRef (E/W)
                lon = gps_info.get(4)      # GPSLongitude
                
                if lon_ref and lon:
                    longitude = self._convert_gps_coordinate(lon)
                    if lon_ref == 'W':
                        longitude = -longitude
                    location.longitude = longitude
                
                # Extract altitude
                alt_ref = gps_info.get(5)  # GPSAltitudeRef (0=above, 1=below sea level)
                alt = gps_info.get(6)      # GPSAltitude
                
                if alt is not None:
                    if isinstance(alt, tuple):
                        altitude = alt[0] / alt[1]
                    else:
                        altitude = alt
                    
                    if alt_ref == 1:  # Below sea level
                        altitude = -altitude
                    location.altitude = altitude
                
                # Extract direction
                direction = gps_info.get(17)  # GPSImgDirection
                if direction is not None:
                    if isinstance(direction, tuple):
                        location.direction = direction[0] / direction[1]
                    else:
                        location.direction = direction
                        
        except Exception as e:
            self.logger.warning(f"Error extracting GPS data: {e}")
        
        return location
    
    def _convert_gps_coordinate(self, coordinate: tuple) -> float:
        """Convert GPS coordinate from degrees/minutes/seconds to decimal"""
        if not coordinate or len(coordinate) < 3:
            return 0.0
        
        degrees, minutes, seconds = coordinate[:3]
        
        # Handle fractional values
        if isinstance(degrees, tuple):
            degrees = degrees[0] / degrees[1]
        if isinstance(minutes, tuple):
            minutes = minutes[0] / minutes[1]
        if isinstance(seconds, tuple):
            seconds = seconds[0] / seconds[1]
        
        return degrees + minutes / 60 + seconds / 3600
    
    def _extract_date_taken(self, exifdata: Image.Exif) -> Optional[datetime]:
        """Extract date taken from EXIF"""
        try:
            # Try different EXIF date fields
            date_fields = [
                0x0132,  # DateTime
                0x9003,  # DateTimeOriginal
                0x9004   # DateTimeDigitized
            ]
            
            for field in date_fields:
                date_str = exifdata.get(field)
                if date_str:
                    try:
                        return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                    except ValueError:
                        continue
            
        except Exception as e:
            self.logger.warning(f"Error extracting date taken: {e}")
        
        return None
    
    def _get_color_space(self, img: Image.Image) -> Optional[str]:
        """Determine color space of image"""
        try:
            if hasattr(img, 'mode'):
                return img.mode
        except:
            pass
        return None
    
    def _extract_description(self, exifdata: Image.Exif) -> Optional[str]:
        """Extract description from EXIF"""
        try:
            # Try different description fields
            description_fields = [
                0x010E,  # ImageDescription
                0x9286,  # UserComment
            ]
            
            for field in description_fields:
                desc = exifdata.get(field)
                if desc and desc.strip():
                    return desc.strip()
                    
        except Exception as e:
            self.logger.warning(f"Error extracting description: {e}")
        
        return None
    
    def _extract_keywords(self, exifdata: Image.Exif) -> Optional[List[str]]:
        """Extract existing keywords from EXIF/IPTC"""
        # This would need additional IPTC parsing
        # For now, return None - keywords will be added by AI analysis
        return None
    
    def generate_iptc_keywords(self, 
                              ai_keywords: List[str],
                              camera_settings: CameraSettings,
                              technical_analysis: Dict,
                              location: LocationData = None) -> List[str]:
        """Generate comprehensive IPTC keywords"""
        keywords = set()
        
        # Add AI-generated content keywords (cleaned and filtered)
        if ai_keywords:
            for keyword in ai_keywords:
                if keyword and len(keyword.strip()) > 2:  # Avoid very short keywords
                    keywords.add(keyword.strip().lower())
        
        # Technical keywords based on camera settings
        if camera_settings:
            # Aperture-based keywords
            if camera_settings.aperture:
                try:
                    f_number = float(camera_settings.aperture.replace('f/', ''))
                    if f_number <= 2.8:
                        keywords.add('shallow depth of field')
                        keywords.add('bokeh')
                    elif f_number >= 8:
                        keywords.add('deep depth of field')
                        keywords.add('landscape photography')
                except:
                    pass
            
            # Shutter speed-based keywords
            if camera_settings.shutter_speed:
                try:
                    if '1/' in camera_settings.shutter_speed:
                        speed_fraction = camera_settings.shutter_speed.replace('s', '')
                        denominator = int(speed_fraction.split('/')[1])
                        if denominator >= 500:
                            keywords.add('fast shutter')
                            keywords.add('action photography')
                    else:
                        speed = float(camera_settings.shutter_speed.replace('s', ''))
                        if speed >= 1:
                            keywords.add('long exposure')
                            keywords.add('motion blur')
                except:
                    pass
            
            # ISO-based keywords
            if camera_settings.iso:
                try:
                    iso_val = int(camera_settings.iso)
                    if iso_val >= 3200:
                        keywords.add('high ISO')
                        keywords.add('low light photography')
                    elif iso_val <= 200:
                        keywords.add('low ISO')
                        keywords.add('base ISO')
                except:
                    pass
            
            # Flash keywords
            if camera_settings.flash and 'Fired' in camera_settings.flash:
                keywords.add('flash photography')
            elif camera_settings.flash and 'No Flash' in camera_settings.flash:
                keywords.add('natural light')
            
            # Focal length keywords
            if camera_settings.focal_length_35mm or camera_settings.focal_length:
                focal_str = camera_settings.focal_length_35mm or camera_settings.focal_length
                try:
                    focal_mm = int(focal_str.replace('mm', ''))
                    if focal_mm <= 35:
                        keywords.add('wide angle')
                    elif focal_mm >= 85:
                        keywords.add('telephoto')
                        if focal_mm >= 200:
                            keywords.add('super telephoto')
                    else:
                        keywords.add('standard lens')
                except:
                    pass
        
        # Technical quality keywords
        if technical_analysis:
            overall_quality = technical_analysis.get('overall_quality', 0)
            blur_score = technical_analysis.get('blur_score', 0)
            
            if overall_quality >= 0.8:
                keywords.add('high quality')
                keywords.add('portfolio worthy')
            
            if blur_score >= 0.9:
                keywords.add('tack sharp')
                keywords.add('perfect focus')
            elif blur_score <= 0.3:
                keywords.add('soft focus')
        
        # Location-based keywords
        if location and location.latitude and location.longitude:
            keywords.add('geotagged')
            # Could add more specific location keywords based on coordinates
        
        # Equipment keywords
        if camera_settings:
            if camera_settings.make:
                brand_lower = camera_settings.make.lower()
                keywords.add(f"{brand_lower} photography")
            
            if camera_settings.lens_model:
                # Add lens-specific keywords if it's a notable lens
                lens_lower = camera_settings.lens_model.lower()
                if 'macro' in lens_lower:
                    keywords.add('macro photography')
                if 'fisheye' in lens_lower:
                    keywords.add('fisheye lens')
        
        # Convert back to list and limit length
        final_keywords = list(keywords)[:20]  # Limit to 20 keywords max
        
        # Sort for consistency
        final_keywords.sort()
        
        return final_keywords
    
    def create_workflow_keywords(self, decision: str, confidence: float, 
                               issues: List[str]) -> List[str]:
        """Create workflow-specific keywords for photo management"""
        workflow_keywords = []
        
        # Decision-based keywords
        workflow_keywords.append(f'PhotoCuller:{decision}')
        workflow_keywords.append(f'CullerConfidence:{confidence:.2f}')
        
        # Issue-based keywords
        if issues:
            issues_str = ', '.join(issues)
            workflow_keywords.append(f'CullerIssues:{issues_str}')
        else:
            workflow_keywords.append('CullerIssues:none')
        
        # Quality rating suggestion
        if confidence >= 0.8:
            if decision == 'Keep':
                workflow_keywords.append('CullerSuggestedRating:4-5')
            elif decision == 'Review':
                workflow_keywords.append('CullerSuggestedRating:3')
        elif confidence >= 0.6:
            if decision == 'Keep':
                workflow_keywords.append('CullerSuggestedRating:3-4')
            else:
                workflow_keywords.append('CullerSuggestedRating:2-3')
        else:
            workflow_keywords.append('CullerSuggestedRating:1-2')
        
        return workflow_keywords
    
    def export_metadata_report(self, metadata: ImageMetadata, 
                              filepath: Path, output_path: Path):
        """Export comprehensive metadata report to JSON"""
        try:
            # Convert to dictionary for JSON serialization
            metadata_dict = {
                'file_info': {
                    'filepath': str(filepath),
                    'filename': filepath.name,
                    'file_size': filepath.stat().st_size if filepath.exists() else 0,
                },
                'camera_settings': asdict(metadata.camera_settings),
                'location': asdict(metadata.location),
                'timestamps': {
                    'date_taken': metadata.date_taken.isoformat() if metadata.date_taken else None,
                    'date_modified': metadata.date_modified.isoformat() if metadata.date_modified else None,
                },
                'image_properties': {
                    'width': metadata.width,
                    'height': metadata.height,
                    'aspect_ratio': f"{metadata.width}:{metadata.height}" if metadata.width and metadata.height else None,
                    'megapixels': round((metadata.width * metadata.height) / 1000000, 1) if metadata.width and metadata.height else None,
                    'color_space': metadata.color_space,
                    'orientation': metadata.orientation,
                },
                'content_metadata': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'keywords': metadata.keywords,
                    'rating': metadata.rating,
                    'artist': metadata.artist,
                    'copyright': metadata.copyright,
                },
                'technical_info': {
                    'software': metadata.software,
                    'lens_serial': metadata.lens_serial,
                    'camera_serial': metadata.camera_serial,
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to export metadata report: {e}")