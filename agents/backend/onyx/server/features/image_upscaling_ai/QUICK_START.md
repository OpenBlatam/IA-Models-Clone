# Quick Start Guide - Image Upscaling AI

## Prerequisites

- Python 3.8+
- OpenRouter API key (optional, for AI enhancement)
- optimization_core path (optional, for advanced processing)

## Installation

1. Install dependencies:

```bash
cd agents/backend/onyx/server/features/image_upscaling_ai
pip install -r requirements.txt
```

2. Set environment variables:

```bash
# Required for AI enhancement
export OPENROUTER_API_KEY="your-api-key"

# Optional: Path to optimization_core
export OPTIMIZATION_CORE_PATH="../Frontier-Model-run-polyglot/scripts/TruthGPT-main/optimization_core"

# Optional: Customize settings
export UPSCALING_SCALE_FACTOR=2.0
export UPSCALING_QUALITY_MODE=high
export UPSCALING_OUTPUT_DIR="./upscaled_images"
```

## Running the Service

### Option 1: Using main.py

```bash
python main.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn api.upscaling_api:app --host 0.0.0.0 --port 8003
```

## Using the API

### Upscale a Single Image

```bash
curl -X POST "http://localhost:8003/api/v1/upscale" \
  -F "image=@input.jpg" \
  -F "scale_factor=2.0" \
  -F "use_ai=true" \
  -F "use_optimization_core=true"
```

### Upscale Multiple Images

```bash
curl -X POST "http://localhost:8003/api/v1/upscale-batch" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "scale_factor=2.0"
```

### Check Health

```bash
curl http://localhost:8003/api/v1/health
```

### Get Model Info

```bash
curl http://localhost:8003/api/v1/model/info
```

## Python Usage

```python
import asyncio
from image_upscaling_ai.core.upscaling_service import UpscalingService
from image_upscaling_ai.config.upscaling_config import UpscalingConfig
from PIL import Image

async def main():
    # Initialize service
    config = UpscalingConfig.from_env()
    service = UpscalingService(config=config)
    
    # Load image
    image = Image.open("input.jpg")
    
    # Upscale
    result = await service.upscale_image(
        image=image,
        scale_factor=2.0,
        use_ai=True,
        use_optimization_core=True
    )
    
    print(f"Original: {result['original_size']}")
    print(f"Upscaled: {result['upscaled_size']}")
    print(f"Quality: {result['quality_score']}")
    print(f"Time: {result['processing_time']:.2f}s")
    
    # Cleanup
    await service.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Options

### Scale Factors

- Minimum: 1.5x
- Default: 2.0x
- Maximum: 8.0x

### Quality Modes

- `fast`: Basic upscaling, fastest
- `balanced`: Enhanced upscaling
- `high`: AI-enhanced upscaling (default)
- `ultra`: Multi-pass AI-enhanced upscaling, best quality

### Features

- `use_ai`: Enable AI enhancement via OpenRouter
- `use_optimization_core`: Enable optimization_core processing

## Troubleshooting

### optimization_core not found

If you see warnings about optimization_core not being found:

1. Set `OPTIMIZATION_CORE_PATH` environment variable
2. Or disable optimization_core: `export UPSCALING_USE_OPTIMIZATION_CORE=false`

### OpenRouter errors

If AI enhancement fails:

1. Check your `OPENROUTER_API_KEY` is set correctly
2. Or disable AI enhancement: `export UPSCALING_USE_AI=false`

### Memory issues

For large images or high scale factors:

1. Use `fast` or `balanced` quality mode
2. Reduce scale factor
3. Process images in smaller batches

## Next Steps

- See `README.md` for full documentation
- Check API documentation at `http://localhost:8003/docs`
- Explore different quality modes and scale factors


