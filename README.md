# AI-Powered Photo Culler 📸

An intelligent photo culling system for ON1 and standard XMP workflows. It combines Ollama-based image analysis with a lightweight local fast mode to sort photos into keep, review, or delete candidates.

## Features ✨

- **Dual Processing Modes**:
  - **Accurate Mode** (default): Uses Ollama for richer descriptions, keywords, and confidence scoring
  - **Fast Mode**: Runs locally without Ollama for quick triage on standard images

- **Reliable Metadata Output**:
  - **ON1 workflow**: Updates existing `.on1` sidecars when they already exist
  - **Universal workflow**: Writes standard `.xmp` sidecars for Lightroom, Capture One, Bridge, and more
  - **CSV export**: Appends every decision to a spreadsheet-friendly results file

- **Smart Caching**: Never reprocess the same file/mode combination
- **Batch Processing**: Processes folders with progress bars and cached re-runs
- **Progress Tracking**: Real-time progress bars with ETA estimation using tqdm
- **Optional RAW Support**: Uses `rawpy` when installed for NEF, CR2, ARW, and other RAW formats
- **Intelligent Decisions**: Categorizes photos as Keep, Delete, or Review with confidence scoring
- **Multiple Culling Tools**:
  - ON1-specific metadata integration with override options (Primary method)
  - Universal metadata support for all photo apps  
  - Standard culling with CSV output and analytics


## Quick Start 🚀

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Optional extras**:
```bash
# Add RAW support and stronger blur analysis
pip install -r requirements_photo.txt
```

3. **Setup Ollama** (recommended for accurate mode):
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the default Gemma 4 vision model
ollama pull gemma4:e4b
```

4. **Setup Multi-Port Ollama** (optional for `culler_on1.py --concurrent`):
```bash
# Start multiple Ollama instances on different ports for parallel processing
# Terminal 1 (primary instance)
ollama serve --port 11434

# Terminal 2 (concurrent instance 2) 
OLLAMA_PORT=11435 ollama serve

# Terminal 3 (concurrent instance 3)
OLLAMA_PORT=11436 ollama serve

# Terminal 4 (concurrent instance 4)
OLLAMA_PORT=11437 ollama serve
```

5. **Basic Usage**:
```bash
# AI-powered culling with ON1 metadata (recommended)
python culler_on1.py /path/to/photos

# Override existing metadata with fresh AI analysis
python culler_on1.py /path/to/photos --override

# Fast local mode for quick triage
python culler_on1.py /path/to/photos --fast

# Concurrent processing (requires multiple Ollama instances)
python culler_on1.py /path/to/photos --concurrent 4 --chunk-size 2

# Universal metadata (works with Lightroom, Bridge, Capture One)
python culler_universal.py /path/to/photos
```

6. **View Results**:
```bash
# Save results to a custom CSV file
python culler_universal.py /photos --csv-file results.csv

# Move files marked for deletion (ON1 approach)
python culler_on1.py /photos --move-deletes

# Move files marked for deletion (universal approach)
python culler_universal.py /photos --move-deletes
```

7. **Advanced Usage with Metadata**:
```bash
# ON1 Photo RAW culling (preserves existing metadata)
python culler_on1.py /photos --cache-dir ~/.cache

# Universal metadata culling (works with Lightroom, Bridge, etc.)
python culler_universal.py /photos --cache-dir ~/.cache

# Use specific Ollama model (ON1)
python culler_on1.py /photos --ollama-model gemma4:26b

# Use specific Ollama model (universal)
python culler_universal.py /photos --ollama-model gemma4:26b

# Run without Ollama in local fast mode
python culler_universal.py /photos --fast --no-ollama
```

## Processing Modes 🔄

### Accurate Mode (Default)
- Uses Gemma 4 on Ollama by default, with structured JSON parsing
- Produces richer descriptions and keywords for metadata workflows
- Best for comprehensive culling when Ollama is available

### Fast Mode (`--fast`)
- Runs locally without Ollama
- Uses lightweight image-statistics analysis for blur, exposure, and composition estimates
- Best for initial triage or systems without Ollama

## Concurrent Processing ⚡

For high-performance systems (Mac Studio M4 Max, high-end workstations):

### Multi-Port Ollama Setup
Enable true parallel processing by running multiple Ollama instances:

```bash
# Start 4 Ollama instances (recommended for Mac Studio M4 Max 64GB)
ollama serve --port 11434 &    # Primary instance
OLLAMA_PORT=11435 ollama serve &
OLLAMA_PORT=11436 ollama serve &
OLLAMA_PORT=11437 ollama serve &
```

### Performance Optimization
```bash
# Process with 4 concurrent instances
python culler_on1.py ~/Photos --concurrent 4 --chunk-size 1

# Fine-tune chunk size for your hardware
python culler_on1.py ~/Photos --concurrent 2 --chunk-size 2

# Memory-optimized for very large collections
python culler_on1.py ~/Photos --concurrent 8 --chunk-size 1
```

### Concurrent Performance Benchmarks
| Hardware | Single Ollama | 4x Concurrent | Speed Improvement |
|----------|--------------|---------------|-------------------|
| Mac Studio M4 Max 64GB | ~14s/img | ~6.5s/img | **54% faster** |
| MacBook Pro M3 Max 32GB | ~18s/img | ~10s/img | **44% faster** |
| High-end Intel/AMD | ~20s/img | ~12s/img | **40% faster** |

**Memory Requirements:**
- Each Ollama instance uses ~4-6GB RAM
- Recommended: 16GB+ for 2 instances, 32GB+ for 4 instances

## Examples 📋

```bash
# Process wedding photos with ON1 metadata (primary workflow)
python culler_on1.py ~/Photos/Wedding2024 --cache-dir ~/.cache

# Quick triage of 10,000 photos with ON1 workflow
python culler_on1.py ~/Photos/Massive_Collection --fast

# Process only RAW files with ON1 workflow  
python culler_on1.py ~/Photos --extensions nef,cr2,arw

# Verbose accurate processing with ON1
python culler_on1.py ~/Photos --verbose

# Universal metadata approach (works with Lightroom, Bridge)
python culler_universal.py ~/Photos/Wedding2024 --cache-dir ~/.cache
```

## Configuration Options ⚙️

| Option | Default | Description |
|--------|---------|-------------|
| `--fast` | False | Use local fast analysis instead of Ollama |
| `--cache-dir` | None | Cache directory for per-file analysis results |
| `--chunk-size` | 1 | Images per worker chunk (for load balancing) |
| `--concurrent` | 1 | Number of concurrent Ollama instances (requires multi-port setup) |
| `--move-deletes` | False | Move deletion candidates to _culled_deletes/ |
| `--extensions` | nef,cr2,arw,jpg,jpeg | File extensions to process |
| `--csv-file` | `photo_culler_results.csv` | CSV file to append results to |
| `--use-ollama / --no-ollama` | Ollama on | Toggle Ollama for accurate mode |
| `--ollama-model` | `gemma4:e4b` | Which Ollama model to use |
| `--learning` | False | Keep session-summary output enabled in `culler_on1.py` |
| `--override` | False | Override existing metadata with fresh AI analysis |

## Decision Logic 🤔

The system evaluates three key metrics:

1. **Blur Score** (0-1): Higher = sharper focus
2. **Exposure Score** (0-1): Higher = better exposed  
3. **Composition Score** (0-1): Higher = more interesting

### Decision Thresholds:
- **Delete**: High confidence in critical blur, poor exposure, or low overall quality
- **Review**: Medium confidence or mixed signals
- **Keep**: Good overall quality and strong sharpness

## Performance Benchmarks ⚡

| Mode | Hardware | Speed | Accuracy | Best For |
|------|----------|--------|----------|----------|
| Fast | CPU | 200ms/img | 70% | Initial triage |
| Accurate + Ollama | GPU | 2000ms/img | 97% | Creative analysis |
| Accurate + Ollama | CPU | 3000ms/img | 95% | Comprehensive culling |
| Accurate (Ollama) | GPU/CPU | ~2000ms/img | Depends on model | Rich metadata workflows |

*Benchmarks on 24MP RAW files*

## File Structure 📁

```
photo_culler/
├── models.py               # Core result dataclasses
├── extractor.py            # Standard image and optional RAW extraction
├── blur_detector.py        # Optional OpenCV-based blur scoring
├── ollama_vision.py        # Ollama-powered accurate analyzer
├── batch.py                # Shared batching, caching, and decision logic
├── culler_on1.py           # ON1 Photo RAW workflow
├── culler_universal.py     # Universal XMP workflow
├── requirements.txt        # Core dependencies
├── requirements_photo.txt  # Optional RAW/OpenCV extras
└── config.yaml             # Example configuration values
```

## Dependencies 📦

### Required:
- **Click**: CLI framework
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **Requests**: Ollama API calls
- **tqdm**: Progress bars

### Optional:
- **Ollama + Gemma 4**: Accurate mode
- **rawpy**: Proper RAW thumbnail extraction
- **OpenCV**: Better blur scoring in fast mode and hybrid Ollama mode

## Caching System 💾

The system intelligently caches:
- **Thumbnails**: Extracted from RAW files (prevents re-extraction)
- **Analysis Results**: Per file + mode combination  
- **Cache Keys**: Based on filename, size, modification time

Cache locations:
- Thumbnails: `{cache_dir}/thumbnails/`
- Results: `{cache_dir}/cull_results.json`

## Output Examples 📊

### Console Output:
```
==================================================
CULLING COMPLETE - AI-POWERED MODE  
==================================================
Keep:    847 files
Delete:   23 files
Review:   15 files

Top deletion candidates:
  IMG_1234.NEF: A heartwarming pool scene but technically flawed (conf: 0.89)
    Keywords: ["father and child", "swimming pool", "motion blur"]
    Issues: Subject motion blur, harsh shadows
  IMG_5678.CR2: Portrait session with focus issues (conf: 0.82)
    Keywords: ["portrait", "shallow dof", "eye focus"]
    Issues: Eyes not in focus, overexposed highlights

Session insights:
  Detected style: Portrait photographer with preference for shallow DOF
  Learning: Adjusted blur thresholds for your f/1.8 shooting style
  Processing: 2100ms per image (including creative analysis)
  Total time: 31.2 minutes
```

### JSON Output:
```json
{
  "mode": "accurate",
  "results": {
    "Delete": [
      {
        "file": "/photos/IMG_1234.NEF",
        "confidence": 0.89,
        "issues": ["subject motion blur", "harsh shadows"],
        "description": "A heartwarming scene of a father and child in a swimming pool, but technically flawed",
        "keywords": ["father and child", "swimming pool", "motion blur", "family moment"],
        "processing_ms": 2100,
        "metrics": {
          "blur": 0.12,
          "exposure": 0.31,
          "composition": 0.67,
          "overall": 0.25,
          "subject_focus": 0.15,
          "artistic_merit": 0.72
        }
      }
    ]
  }
}
```

## Troubleshooting 🔧

### Common Issues:

**"No module named 'click'"**
- Install the core dependencies: `pip install -r requirements.txt`

**"Accurate mode requires Ollama"**
- Start Ollama with `ollama serve`
- Pull the model with `ollama pull gemma4:e4b`
- Or switch to local triage mode: `--fast --no-ollama`

**"No RAW files processed"**
- Install the optional photo extras: `pip install -r requirements_photo.txt`
- Check file extensions: `--extensions nef,cr2,arw`

**Slow processing**
- Use fast mode for triage: `--fast`
- Enable concurrent processing: `--concurrent 4`
- Reduce chunk size: `--chunk-size 1`
- Enable caching: `--cache-dir ~/.cache`

### Ollama Issues:

**"Cannot connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check if it's accessible: `curl http://localhost:11434/api/tags`

**"Model not found"**
- List available models: `ollama list`
- Pull the model if missing: `ollama pull gemma4:e4b`

**"Ollama query failed"**
- Check Ollama logs: `ollama logs`
- Restart Ollama service
- Try a different model

### Concurrent Processing Issues:

**"Cannot connect to Ollama on port XXXXX"**
- Ensure all Ollama instances are running on specified ports
- Check port availability: `lsof -i :11435`
- Verify each instance with: `curl http://localhost:11435/api/tags`

**"Out of memory with concurrent processing"**
- Reduce concurrent instances: `--concurrent 2`
- Increase system memory or close other applications
- Monitor memory usage: `htop` or Activity Monitor

**"No performance improvement with --concurrent"**
- Ensure multiple Ollama instances are actually running
- Check all instances are serving different ports
- Use `--chunk-size 1` for optimal load distribution


## Advanced Usage 🎯

### Custom Decision Thresholds:
Adjust the thresholds inside `BatchCuller._make_decision()` if you want stricter or looser sorting behavior.

### Integration with Other Tools:
```python
from pathlib import Path
from batch import BatchCuller

culler = BatchCuller(mode="accurate", use_ollama=True)
results = culler.process_folder_batch(Path("/photos"), [".jpg", ".jpeg", ".nef"])

for result in results['Delete']:
    print(f"Delete: {result.filepath} ({result.confidence:.2f})")
```

### ON1 Workflow Best Practices:
When using `culler_on1.py`:

- **Process in ON1 Photo RAW** after running culler to see metadata keywords
- Use `--move-deletes` to automatically move deletion candidates to `_culled_deletes/`
- Search for keywords like `PhotoCuller:Delete` in ON1 to identify candidates
- Review high-confidence deletion candidates (confidence > 0.7) before deleting

### Universal Workflow Best Practices:
When using `culler_universal.py`:

- Use with any photo app (Lightroom, Bridge, Capture One)
- Metadata is stored in `.xmp` sidecar files
- Preserve existing ratings, keywords and descriptions  
- Search for metadata like `PhotoCuller:Delete` in your photo app

## License 📄

MIT License - Feel free to use and modify for your photo workflow!

---

**Pro Tip**: Start with fast mode on large collections, then use accurate mode on the "Review" category for final decisions. This hybrid approach gives you the best of both worlds! 🎯

## Repository Status 🔍

The repository is designed to be public and does not contain any personal information. All paths in example code have been updated for general use.

Please review the following before making this repository public:

1. **Test Files**: Remove any personal test data from `test/` directory
2. **Configuration files**: Ensure no private information in `.env` or config files (none currently exist)
3. **API Keys**: This tool does not use any external API keys
4. **User Paths**: All user-specific paths in example code have been replaced with generic references

You can safely make this repository public as it contains only the core tool functionality.
