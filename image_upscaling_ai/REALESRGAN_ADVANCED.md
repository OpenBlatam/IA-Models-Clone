# Real-ESRGAN Advanced Features

## Overview

Advanced features for Real-ESRGAN integration including model management, automatic model selection, batch processing, and model comparison.

## Features

### 1. Model Manager (`RealESRGANModelManager`)

Intelligent model management with caching and optimization.

**Features:**
- Model caching (LRU cache)
- Automatic model selection based on image type
- Memory management
- Performance monitoring
- Batch processing optimization

**Usage:**

```python
from image_upscaling_ai.models import RealESRGANModelManager

# Initialize manager
manager = RealESRGANModelManager(
    max_cached_models=3,
    cache_ttl=3600.0,  # 1 hour
    auto_download=False,
    device="cuda"
)

# Smart upscaling with auto model selection
upscaled = await manager.upscale_async(
    image,
    scale_factor=4.0,
    auto_select=True  # Automatically selects best model
)

# Batch processing
images = [img1, img2, img3]
upscaled_images = await manager.batch_upscale_async(
    images,
    scale_factor=4.0,
    max_concurrent=2
)
```

### 2. Image Type Detection

Automatically detects image type (anime/photo/artwork) and selects appropriate model.

**Usage:**

```python
# Detect image type
image_type = manager.detect_image_type(image)
# Returns: 'anime', 'photo', or 'artwork'

# Select best model
best_model = manager.select_best_model(image, scale_factor=4.0)
```

**Detection Logic:**
- **Anime**: High saturation + high contrast
- **Photo**: Low saturation + low contrast
- **Artwork**: Everything else

### 3. Model Comparison (`ModelComparison`)

Compare different Real-ESRGAN models on the same image.

**Features:**
- Side-by-side comparison
- Quality metrics comparison
- Performance comparison
- Best model recommendation

**Usage:**

```python
from image_upscaling_ai.models import ModelComparison

comparison = ModelComparison()

# Compare models
results = comparison.compare_models(
    image,
    scale_factor=4.0,
    model_names=[
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRNet_x4plus"
    ]
)

# Get best model
best_model = results["best_model"]

# Create comparison grid
grid = comparison.create_comparison_grid(results, labels=True)
grid.save("comparison.png")
```

### 4. Advanced Presets

New presets specifically optimized for Real-ESRGAN:

- **realesrgan_photo**: 4x photo upscaling
- **realesrgan_anime**: 4x anime upscaling (uses anime model)
- **realesrgan_artwork**: 4x artwork upscaling
- **realesrgan_fast**: 2x fast upscaling (RealESRNet)

**Usage:**

```python
from image_upscaling_ai.models import PresetManager

# Apply Real-ESRGAN preset
config = PresetManager.apply_preset("realesrgan_anime")
```

## API Endpoints

### Smart Upscaling

```bash
POST /api/v1/realesrgan/advanced/upscale-smart
```

Automatically selects best model based on image type.

**Parameters:**
- `image`: Image file
- `scale_factor`: Scale factor
- `auto_select_model`: Auto-select best model (default: true)
- `model_name`: Specific model (optional)

### Compare Models

```bash
POST /api/v1/realesrgan/advanced/compare-models
```

Compare different models on the same image.

**Parameters:**
- `image`: Image file
- `scale_factor`: Scale factor (default: 4.0)
- `model_names`: Comma-separated model names (optional)

### Detect Image Type

```bash
POST /api/v1/realesrgan/advanced/detect-image-type
```

Detect image type and get model recommendation.

**Parameters:**
- `image`: Image file

**Response:**
```json
{
  "image_type": "anime",
  "recommended_model": "RealESRGAN_x4plus_anime_6B",
  "image_size": [512, 512]
}
```

### Batch Upscaling

```bash
POST /api/v1/realesrgan/advanced/batch-upscale
```

Upscale multiple images in parallel.

**Parameters:**
- `images`: List of image files
- `scale_factor`: Scale factor
- `max_concurrent`: Max concurrent upscales (default: 2)

### Manager Statistics

```bash
GET /api/v1/realesrgan/advanced/manager/stats
```

Get model manager statistics and cache information.

### Clear Cache

```bash
POST /api/v1/realesrgan/advanced/manager/clear-cache
```

Clear model cache.

## Performance Optimization

### Model Caching

Models are cached to avoid reloading:
- LRU (Least Recently Used) eviction
- Configurable TTL (time-to-live)
- Automatic memory management

### Batch Processing

Optimized batch processing:
- Parallel execution with semaphore
- Configurable concurrency
- Automatic model reuse

### Memory Management

- Automatic model eviction when cache is full
- Memory-efficient model loading
- GPU memory optimization

## Best Practices

1. **Use Model Manager for Multiple Images**
   ```python
   # Good: Reuses cached models
   manager = RealESRGANModelManager()
   for image in images:
       upscaled = await manager.upscale_async(image, 4.0)
   ```

2. **Auto-Select Model for Best Results**
   ```python
   # Automatically selects best model
   upscaled = await manager.upscale_async(
       image, 4.0, auto_select=True
   )
   ```

3. **Use Batch Processing for Multiple Images**
   ```python
   # Efficient batch processing
   results = await manager.batch_upscale_async(
       images, 4.0, max_concurrent=2
   )
   ```

4. **Compare Models for New Use Cases**
   ```python
   # Find best model for your images
   comparison = ModelComparison()
   results = comparison.compare_models(image, 4.0)
   best = results["best_model"]
   ```

## Statistics and Monitoring

### Manager Statistics

```python
stats = manager.get_stats()
# Returns:
# {
#   "models_loaded": 5,
#   "cache_hits": 20,
#   "cache_misses": 3,
#   "cache_hit_rate": 0.87,
#   "total_upscales": 23,
#   "cached_models": 2
# }
```

### Cache Information

```python
cache_info = manager.get_cache_info()
# Returns:
# {
#   "cached_models": 2,
#   "max_models": 3,
#   "cache_ttl": 3600.0,
#   "entries": [...]
# }
```

## Examples

### Example 1: Smart Upscaling

```python
from image_upscaling_ai.models import RealESRGANModelManager
from PIL import Image

manager = RealESRGANModelManager()

# Load image
image = Image.open("photo.jpg")

# Smart upscale (auto-selects best model)
upscaled = await manager.upscale_async(
    image,
    scale_factor=4.0,
    auto_select=True
)

upscaled.save("upscaled.jpg")
```

### Example 2: Batch Processing

```python
images = [Image.open(f"img_{i}.jpg") for i in range(10)]

# Batch upscale
upscaled = await manager.batch_upscale_async(
    images,
    scale_factor=4.0,
    max_concurrent=2
)

# Save results
for i, img in enumerate(upscaled):
    img.save(f"upscaled_{i}.jpg")
```

### Example 3: Model Comparison

```python
from image_upscaling_ai.models import ModelComparison

comparison = ModelComparison()

# Compare models
results = comparison.compare_models(
    image,
    scale_factor=4.0
)

# Get best model
print(f"Best model: {results['best_model']}")

# Create comparison grid
grid = comparison.create_comparison_grid(results)
grid.save("comparison.png")
```

## Integration with Service

The advanced features integrate seamlessly with the main service:

```python
from image_upscaling_ai.core.upscaling_service import UpscalingService
from image_upscaling_ai.config.upscaling_config import UpscalingConfig

config = UpscalingConfig.from_env()
config.use_realesrgan = True
config.quality_mode = "ultra"  # Uses Real-ESRGAN

service = UpscalingService(config=config)

# Service automatically uses Real-ESRGAN with smart model selection
result = await service.upscale_image("photo.jpg", scale_factor=4.0)
```

## Troubleshooting

### High Memory Usage

```python
# Reduce cache size
manager = RealESRGANModelManager(max_cached_models=1)

# Clear cache periodically
manager.clear_cache()
```

### Slow Performance

```python
# Use faster model for 2x
model_name = manager.select_best_model(image, 2.0)
# Returns: "RealESRGAN_x2plus"

# Or use RealESRNet (no GAN, faster)
model_name = "RealESRNet_x4plus"
```

### Model Not Found

```python
# Check available models
from image_upscaling_ai.models import RealESRGANWrapper
models = RealESRGANWrapper.list_available_models()

# Download missing model
RealESRGANWrapper.download_model("RealESRGAN_x4plus")
```


