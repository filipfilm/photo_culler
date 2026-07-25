# Ollama Vision Setup for Photo Culler 🦙

This guide shows how to set up Ollama as the vision backend for accurate photo analysis.

## Why Ollama? 

✅ **No PyTorch dependency conflicts**  
✅ **Works on both CPU and GPU**  
✅ **Local processing (privacy)**  
✅ **Multiple model options**  
✅ **Easy to install and use**  

## Installation Steps

### 1. Install Ollama

Visit https://ollama.ai and download Ollama for your platform, or:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai
```

### 2. Start Ollama Service

```bash
# Start Ollama (runs on localhost:11434 by default)
ollama serve
```

### 3. Install Vision Model

Choose a vision model. Gemma 4 is the default path in this repo:

```bash
# Recommended default
ollama pull gemma4:e4b

# Alternatives:
ollama pull gemma4        # Alias for the default Gemma 4 variant
ollama pull gemma4:26b    # Better quality, slower
ollama pull gemma4:31b    # Largest model, highest resource use
```

### 4. Verify Installation

Test that Ollama is working:

```bash
# Test with a simple query
ollama run gemma4:e4b
```

You should see a chat interface. Type `/bye` to exit.

## Using with Photo Culler

### Basic Usage

```bash
# Use Ollama in the ON1 workflow
python culler_on1.py /path/to/photos

# Specify which Ollama model to use
python culler_on1.py /path/to/photos --ollama-model gemma4:26b

# Combine with other options
python culler_universal.py /path/to/photos --cache-dir ~/.cache --ollama-model gemma4:e4b
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-ollama / --no-ollama` | Ollama on | Enable or disable Ollama vision analysis |
| `--ollama-model` | `gemma4:e4b` | Which Ollama model to use |

### Performance Tips

**Model Selection:**
- `gemma4:e4b` - Best default balance of speed and quality
- `gemma4:26b` - Better quality, more RAM/VRAM
- `gemma4:31b` - Highest quality, heaviest model

**Batch Size:**
- Smaller batch sizes (2-4) for CPU processing
- Larger batch sizes (8-16) if you have GPU with lots of VRAM

**Hardware Recommendations:**
- **CPU**: 16GB+ RAM, any modern processor
- **GPU**: More memory helps significantly for `gemma4:26b` and `gemma4:31b`

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if it's accessible
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# List available models
ollama list

# Pull the model if missing
ollama pull gemma4:e4b
```

### "Ollama query failed"
- Check Ollama logs: `ollama logs`
- Restart Ollama service
- Try a different model: `--ollama-model gemma4:26b`

### Slow Processing
```bash
# Use the ON1 workflow
python culler_on1.py /photos

# Use a larger model
python culler_on1.py /photos --ollama-model gemma4:26b
```

## Comparison: CLIP vs Ollama

| Feature | CLIP | Ollama |
|---------|------|--------|
| **Setup** | Complex (PyTorch deps) | Simple (single install) |
| **Speed** | Fast (~1s/image GPU) | Medium (~3s/image) |
| **Accuracy** | Good (85-90%) | Excellent (90-95%) |
| **Hardware** | GPU recommended | CPU/GPU both work |
| **Dependencies** | Heavy (PyTorch) | Light (HTTP requests) |
| **Privacy** | Local processing | Local processing |

## Example Workflow

```bash
# 1. Install Ollama and model
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull gemma4:e4b

# 2. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements_photo.txt

# 3. Run photo culler
python culler_universal.py ~/Photos/vacation_2024 \
  --cache-dir ~/.cache/photo_culler \
  --ollama-model gemma4:e4b
```

## Advanced Usage

### Custom Ollama Host
```python
# If running Ollama on different host/port
from ollama_vision import OllamaVisionAnalyzer

analyzer = OllamaVisionAnalyzer(
    model="gemma4:e4b",
    host="http://192.168.1.100:11434"
)
```

### Custom Prompts
Edit `ollama_vision.py` to customize the analysis prompts for your specific needs.

---

🎯 **Ready to cull photos with AI vision!** The Ollama setup gives you state-of-the-art photo analysis without the dependency headaches.
