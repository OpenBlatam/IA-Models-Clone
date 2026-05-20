# Real-ESRGAN Integration Guide

## Overview

Real-ESRGAN integration provides state-of-the-art super-resolution capabilities for image upscaling. This integration uses pre-trained models from the Real-ESRGAN project for superior quality results.

## Installation

### Option 1: Install Real-ESRGAN

```bash
pip install realesrgan basicsr
```

### Option 2: Install from source

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
pip install basicsr
```

## Available Models

### RealESRGAN_x4plus (Recommended)
- **Scale**: 4x
- **Best for**: General purpose upscaling
- **Quality**: Excellent
- **Speed**: Medium
- **Download**: Auto or manual

### RealESRGAN_x4plus_anime_6B
- **Scale**: 4x
- **Best for**: Anime, cartoons, illustrations
- **Quality**: Excellent for anime
- **Speed**: Fast
- **Download**: Auto or manual

### RealESRGAN_x2plus
- **Scale**: 2x
- **Best for**: 2x upscaling (faster than 4x)
- **Quality**: Excellent
- **Speed**: Fast
- **Download**: Auto or manual

### RealESRNet_x4plus
- **Scale**: 4x
- **Best for**: General purpose (no GAN, faster)
- **Quality**: Very good
- **Speed**: Fast
- **Download**: Auto or manual

## Usage

### Python API

```python
from image_upscaling_ai.models import RealESRGANUpscaler
from PIL import Image

# Initialize upscaler
upscaler = RealESRGANUpscaler(
    device="cuda",  # or "cpu"
    auto_download=False  # Set to True to auto-download models
)

# Upscale image
image = Image.open("input.jpg")
upscaled = upscaler.upscale(image, scale_factor=4.0)
upscaled.save("output.jpg")
```

### Using with Service

```python
from image_upscaling_ai.core.upscaling_service import UpscalingService
from image_upscaling_ai.config.upscaling_config import UpscalingConfig

# Configure to use Real-ESRGAN
config = UpscalingConfig.from_env()
config.use_realesrgan = True
config.realesrgan_model = "RealESRGAN_x4plus"
config.quality_mode = "high"  # High and ultra modes use Real-ESRGAN

service = UpscalingService(config=config)

# Upscale (will use Real-ESRGAN automatically)
result = await service.upscale_image(
    "input.jpg",
    scale_factor=4.0
)
```

### Direct Model Usage

```python
from image_upscaling_ai.models import RealESRGANWrapper

# Initialize specific model
model = RealESRGANWrapper(
    model_name="RealESRGAN_x4plus",
    device="cuda"
)

# Upscale
upscaled = model.upscale(image, outscale=4.0)
```

## Downloading Models

### Automatic Download

```python
from image_upscaling_ai.models import RealESRGANWrapper

# Download model
model_path = RealESRGANWrapper.download_model("RealESRGAN_x4plus")
print(f"Model downloaded to: {model_path}")
```

### Manual Download

Models are stored in `~/.cache/realesrgan/models/` by default.

Download URLs:
- RealESRGAN_x4plus: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- RealESRGAN_x4plus_anime_6B: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
- RealESRGAN_x2plus: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
- RealESRNet_x4plus: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth

## API Endpoints

### Check Availability

```bash
curl http://localhost:8003/api/v1/realesrgan/available
```

### List Models

```bash
curl http://localhost:8003/api/v1/realesrgan/models
```

### Download Model

```bash
curl -X POST http://localhost:8003/api/v1/realesrgan/download \
  -F "model_name=RealESRGAN_x4plus"
```

### Get Model Info

```bash
curl http://localhost:8003/api/v1/realesrgan/model/RealESRGAN_x4plus/info
```

## Configuration

### Environment Variables

```bash
# Enable Real-ESRGAN
export UPSCALING_USE_REALESRGAN=true

# Specify model
export REALESRGAN_MODEL=RealESRGAN_x4plus

# Model path (optional)
export REALESRGAN_MODEL_PATH=/path/to/model.pth

# Auto-download models
export REALESRGAN_AUTO_DOWNLOAD=false
```

### Quality Modes

- **high**: Uses Real-ESRGAN for 4x upscaling
- **ultra**: Uses Real-ESRGAN with multi-pass for >4x
- **balanced**: Uses OpenCV/Lanczos (faster)
- **fast**: Uses Lanczos (fastest)

## Performance

### Speed Comparison

| Method | 2x (512→1024) | 4x (512→2048) | Quality |
|--------|---------------|---------------|---------|
| Lanczos | ~0.1s | ~0.2s | Good |
| OpenCV | ~0.3s | ~0.5s | Very Good |
| Real-ESRGAN | ~1.0s | ~2.0s | Excellent |

### Quality Comparison

Real-ESRGAN provides:
- Better detail preservation
- Reduced artifacts
- Better edge handling
- Superior results for photos and artwork

## Troubleshooting

### Model Not Found

```python
# Download model first
from image_upscaling_ai.models import RealESRGANWrapper
RealESRGANWrapper.download_model("RealESRGAN_x4plus")
```

### CUDA Out of Memory

```python
# Use CPU or smaller tile size
model = RealESRGANWrapper(
    model_name="RealESRGAN_x4plus",
    device="cpu",  # Use CPU
    tile=512  # Smaller tiles for large images
)
```

### Import Errors

```bash
# Install dependencies
pip install realesrgan basicsr facexlib gfpgan
```

## Best Practices

1. **For Photos**: Use `RealESRGAN_x4plus`
2. **For Anime**: Use `RealESRGAN_x4plus_anime_6B`
3. **For Speed**: Use `RealESRNet_x4plus` (no GAN)
4. **For 2x**: Use `RealESRGAN_x2plus`
5. **Large Images**: Use tiling (`tile=512` or `tile=1024`)

## Integration with Existing Code

The integration is seamless - when Real-ESRGAN is available and enabled, it's automatically used for `high` and `ultra` quality modes:

```python
# Automatically uses Real-ESRGAN if available
result = await service.upscale_image(
    "photo.jpg",
    scale_factor=4.0,
    # quality_mode="high"  # Uses Real-ESRGAN
)
```

## References

- Real-ESRGAN GitHub: https://github.com/xinntao/Real-ESRGAN
- Paper: https://arxiv.org/abs/2107.10833
- Model Releases: https://github.com/xinntao/Real-ESRGAN/releases


