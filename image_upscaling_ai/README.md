# Image Upscaling AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI-powered image upscaling service using OpenRouter and optimization_core for super-resolution image enhancement.

## Features

- **Multi-scale upscaling** — Support for 1.5x to 8x upscaling
- **AI enhancement** — Integration with OpenRouter for AI-powered image enhancement
- **optimization_core integration** — Uses optimization_core for advanced image processing
- **Quality modes** — Fast, balanced, high, and ultra quality modes
- **Batch processing** — Process multiple images in batch
- **REST API** — FastAPI-based REST API for easy integration

## Architecture

Based on the structure of `character_clothing_changer_ai` and adapted for OpenRouter integration similar to `color_grading_ai_truthgpt`.

### Components

- **Models**: `UpscalingModel` - Core upscaling logic with AI and optimization_core integration
- **Service**: `UpscalingService` - Main service for upscaling operations
- **API**: FastAPI endpoints for upscaling operations
- **Infrastructure**: OpenRouter client for AI enhancement

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export OPENROUTER_API_KEY="your-api-key"
export OPTIMIZATION_CORE_PATH="path/to/optimization_core"  # Optional
export UPSCALING_OUTPUT_DIR="./upscaled_images"
```

## Configuration

Configuration can be set via environment variables or using `UpscalingConfig`:

- `OPENROUTER_API_KEY`: OpenRouter API key
- `UPSCALING_SCALE_FACTOR`: Default scale factor (default: 2.0)
- `UPSCALING_QUALITY_MODE`: Quality mode - fast, balanced, high, ultra (default: high)
- `UPSCALING_USE_AI`: Enable AI enhancement (default: true)
- `UPSCALING_USE_OPTIMIZATION_CORE`: Enable optimization_core (default: true)
- `UPSCALING_OUTPUT_DIR`: Output directory (default: ./upscaled_images)
- `UPSCALING_API_PORT`: API port (default: 8003)

## Usage

### Python API

```python
from image_upscaling_ai.core.upscaling_service import UpscalingService
from image_upscaling_ai.config.upscaling_config import UpscalingConfig
from PIL import Image

# Initialize service
config = UpscalingConfig.from_env()
service = UpscalingService(config=config)

# Upscale image
image = Image.open("input.jpg")
result = await service.upscale_image(
    image=image,
    scale_factor=2.0,
    use_ai=True,
    use_optimization_core=True
)

print(f"Upscaled from {result['original_size']} to {result['upscaled_size']}")
print(f"Quality score: {result['quality_score']}")
```

### REST API

Start the server:

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn api.upscaling_api:app --host 0.0.0.0 --port 8003
```

#### Endpoints

- `POST /api/v1/upscale`: Upscale a single image
- `POST /api/v1/upscale-batch`: Upscale multiple images
- `GET /api/v1/model/info`: Get model information
- `GET /api/v1/health`: Health check
- `POST /api/v1/initialize`: Initialize model

#### Example Request

```bash
curl -X POST "http://localhost:8003/api/v1/upscale" \
  -F "image=@input.jpg" \
  -F "scale_factor=2.0" \
  -F "use_ai=true" \
  -F "use_optimization_core=true"
```

## Quality Modes

- **fast**: Basic upscaling, no enhancements
- **balanced**: Upscaling with sharpness and contrast enhancement
- **high**: Upscaling with AI enhancement and optimization_core
- **ultra**: Multi-pass upscaling with all enhancements

## optimization_core Integration

The service integrates with `optimization_core` from `Frontier-Model-run-polyglot/scripts/TruthGPT-main/optimization_core` for advanced image processing.

Set the path via environment variable:

```bash
export OPTIMIZATION_CORE_PATH="path/to/optimization_core"
```

Or configure in `UpscalingConfig`:

```python
config = UpscalingConfig(
    optimization_core_path="path/to/optimization_core",
    use_optimization_core=True
)
```

## OpenRouter Integration

The service uses OpenRouter for AI-powered image enhancement. Configure your API key:

```bash
export OPENROUTER_API_KEY="your-api-key"
```

The service will use OpenRouter to get AI recommendations for post-processing upscaled images to reduce pixelation and improve quality.

## Performance

- **Fast mode**: ~0.5-1s per image
- **Balanced mode**: ~1-2s per image
- **High mode**: ~2-5s per image (with AI)
- **Ultra mode**: ~5-10s per image (multi-pass with AI)

Times vary based on image size and scale factor.

## Requirements

- Python 3.8+
- PIL/Pillow
- FastAPI
- httpx
- numpy
- pydantic

See `requirements.txt` for full list.

---

[← Back to Main README](../README.md)
