# 📸 Enhanced Photo Culler

Your photo culling system has been significantly enhanced with powerful new features for professional photo management and quality control.

## 🚀 New Features Added

### 1. **Duplicate/Similar Image Detection** (`similarity_detector.py`)
- **Multi-hash similarity detection** using average, perceptual, difference, and wavelet hashes
- **Burst sequence identification** for rapid-fire photography
- **Smart duplicate elimination** with quality-based selection
- **Storage savings analysis** showing potential space recovery

### 2. **Enhanced Technical Quality Control** (`technical_qc.py`)
- **Chromatic aberration detection** - identifies purple/green fringing
- **Noise level analysis** using multiple detection methods
- **Gradient banding detection** in smooth areas like skies
- **Moiré pattern detection** using frequency analysis
- **Lens vignetting assessment** 
- **Dust spot detection** for sensor cleanliness

### 3. **Comprehensive Metadata Enhancement** (`metadata_enhancer.py`)
- **Complete EXIF extraction** including camera settings, GPS, timing
- **Professional IPTC keyword generation** based on technical analysis
- **Camera-setting-aware keywords** (aperture, shutter speed, ISO based)
- **Workflow integration keywords** for photo management
- **Metadata export capabilities** for external tools

### 4. **Smart Batch Processing** (`smart_sequencer.py`)
- **Intelligent image grouping** by time proximity and naming patterns
- **Burst sequence detection** and optimized processing order
- **Processing time estimation** for better workflow planning
- **Priority-based sequencing** (bursts → sessions → single images)

### 5. **Dynamic Model Selection** (`dynamic_model_selector.py`)
- **Content-aware model selection** (portraits vs landscapes vs general)
- **System resource optimization** based on available RAM/CPU
- **Automatic concurrent instance calculation** 
- **Processing mode adaptation** (speed vs quality)

### 6. **Professional Session Reports** (`report_generator.py`)
- **Interactive HTML reports** with charts and visualizations
- **Quality score distribution analysis**
- **Technical issue frequency analysis** 
- **Deletion candidate analysis** with reasoning
- **JSON and CSV exports** for data integration

## 📋 Usage Examples

### Basic Enhanced Analysis
```bash
# Full analysis with similarity detection and reports
python enhanced_culler.py /photos --detect-similar --generate-report

# Quick duplicate detection only  
python enhanced_culler.py /photos --fast --detect-similar
```

### Professional Workflow
```bash
# Complete professional analysis with all features
python enhanced_culler.py /photos \
    --detect-similar \
    --smart-sequence \
    --dynamic-model \
    --enhanced-metadata \
    --generate-report \
    --use-ollama
```

### High-Performance Processing
```bash
# Optimized for speed with smart concurrency
python enhanced_culler.py /photos \
    --dynamic-model \
    --concurrent 4 \
    --smart-sequence \
    --use-ollama
```

## 🔧 New Command Line Options

| Option | Description |
|--------|-------------|
| `--detect-similar` | Enable duplicate/similar image detection |
| `--similarity-threshold` | Similarity threshold (0-1, default: 0.90) |
| `--smart-sequence` | Use intelligent batch sequencing |
| `--dynamic-model` | Enable dynamic Ollama model selection |
| `--generate-report` | Create detailed HTML/JSON/CSV reports |
| `--enhanced-metadata` | Extract comprehensive EXIF/IPTC metadata |
| `--report-dir` | Directory for generated reports |

## 📊 Report Features

The new HTML reports include:

- **Interactive quality distribution charts**
- **Technical issue analysis with frequency breakdown**
- **Deletion candidate list with confidence scores**
- **Processing time analytics**
- **Similarity analysis results**
- **Session metadata and configuration**

## 🎯 Key Improvements

### For Professional Photographers:
- **Burst sequence handling** - automatically identifies and optimizes processing of rapid-fire shots
- **Technical QC alerts** - flags chromatic aberration, noise, banding, and other issues
- **Equipment-based keywords** - adds lens and camera-specific metadata
- **Portfolio quality flagging** - identifies images suitable for professional use

### For Workflow Efficiency:
- **Smart processing order** - processes similar images together for better comparative analysis
- **Dynamic resource usage** - automatically optimizes based on available system resources  
- **Comprehensive reporting** - detailed analytics for post-processing decisions
- **Duplicate elimination** - significant storage savings through intelligent deduplication

### For Quality Assurance:
- **Multi-layer technical analysis** - catches issues traditional methods miss
- **Confidence-based decisions** - more reliable automated culling
- **Enhanced focus detection** - better blur/sharpness analysis
- **Professional keyword standards** - IPTC-compliant metadata generation

## 📈 Performance Optimizations

- **Intelligent caching** with file modification tracking
- **Batch processing optimization** for vision models
- **Concurrent Ollama instance management**
- **Progressive processing** with checkpoint saving
- **Memory-efficient similarity detection**

## 🔄 Migration from Original Culler

Your existing `culler_on1.py` continues to work unchanged. The new `enhanced_culler.py` provides:

- **All original functionality** preserved
- **Backward compatible** command-line interface
- **Optional feature activation** - use only what you need
- **Same performance** for basic usage, enhanced capabilities when enabled

## 🛠 Installation

Update your requirements:
```bash
pip install -r requirements_photo.txt
```

The enhanced features require these additional dependencies:
- `imagehash>=4.3.0` - for similarity detection
- `piexif>=1.1.3` - for enhanced metadata handling  
- `psutil>=5.9.0` - for system resource monitoring
- `scipy>=1.10.0` - for advanced analysis algorithms

## 🎉 Ready to Use

Your photo culling system is now a professional-grade tool with advanced features that rival commercial photo management software. Start with basic similarity detection, then gradually enable more features as needed for your workflow.

The enhanced system maintains the same ease-of-use while providing enterprise-level capabilities for serious photography workflows.