# Optimization Guide

## Overview

This guide covers advanced optimizations for the image upscaling system, including memory management, performance tuning, and quality optimization.

## Smart Tiling

### When to Use

Use smart tiling for:
- Images larger than 2048x2048 pixels
- Limited GPU/CPU memory
- Batch processing large images
- High scale factors (>4x)

### Usage

```python
from image_upscaling_ai.models import SmartTiling

# Initialize with memory limit
tiler = SmartTiling(
    max_tile_size=512,
    overlap=32,
    memory_limit_mb=2048  # Auto-calculates tile size
)

# Process with tiling
upscaled = tiler.process_tiled(
    image,
    scale_factor=4.0,
    upscale_func=lambda img, scale: upscaler.upscale(img, scale)
)
```

### Auto-Tile Size Calculation

```python
# Automatically calculate optimal tile size
tile_size = SmartTiling.auto_tile_size(
    image_size=(4096, 4096),
    scale_factor=4.0,
    available_memory_mb=4096
)
```

## Hybrid Upscaling

### Methods

1. **Blend**: Combines two methods with weighted blending
2. **Two-Stage**: Coarse upscaling then refinement
3. **Adaptive**: Different methods for different regions

### Usage

```python
from image_upscaling_ai.models import HybridUpscaler

# Initialize hybrid upscaler
hybrid = HybridUpscaler(
    primary_method="realesrgan",
    secondary_method="lanczos",
    blend_ratio=0.7  # 70% primary, 30% secondary
)

# Upscale with hybrid method
upscaled = hybrid.upscale_hybrid(
    image,
    scale_factor=4.0,
    method="auto"  # or "blend", "two_stage", "adaptive"
)
```

## Advanced Image Detection

### Comprehensive Analysis

```python
from image_upscaling_ai.models import AdvancedImageDetector

detector = AdvancedImageDetector()
analysis = detector.analyze(image)

print(f"Type: {analysis.image_type}")
print(f"Quality: {analysis.quality_score}")
print(f"Complexity: {analysis.complexity}")
print(f"Recommended Model: {analysis.recommended_model}")
print(f"Confidence: {analysis.confidence}")
```

### Features

- **Image Type**: anime, photo, artwork, pixel_art, mixed
- **Quality Score**: 0.0-1.0
- **Complexity**: 0.0-1.0
- **Text Detection**: Boolean
- **Face Detection**: Boolean
- **Color Analysis**: Dominant colors
- **Model Recommendation**: Best model for image

## Memory Optimization

### Model Caching

```python
from image_upscaling_ai.models import RealESRGANModelManager

# Limit cached models
manager = RealESRGANModelManager(
    max_cached_models=2,  # Reduce for low memory
    cache_ttl=1800  # 30 minutes
)

# Clear cache when needed
manager.clear_cache()
```

### Batch Processing

```python
# Process in smaller batches
results = await manager.batch_upscale_async(
    images,
    scale_factor=4.0,
    max_concurrent=1  # Reduce for low memory
)
```

## Performance Tuning

### GPU vs CPU

```python
# Use CPU for small images or when GPU memory is limited
manager = RealESRGANModelManager(device="cpu")

# Use GPU for large batches
manager = RealESRGANModelManager(device="cuda")
```

### Tile Size Optimization

```python
# Larger tiles = faster but more memory
tiler = SmartTiling(max_tile_size=1024)  # Fast, high memory

# Smaller tiles = slower but less memory
tiler = SmartTiling(max_tile_size=256)  # Slow, low memory
```

### Method Selection

```python
# Fast methods
methods = ["lanczos", "bicubic_enhanced", "RealESRNet_x4plus"]

# Quality methods
methods = ["realesrgan", "opencv_edsr", "hybrid"]
```

## Quality Optimization

### Pre-processing

```python
# Enhance before upscaling
from image_upscaling_ai.models.advanced_upscaling import AdvancedUpscaling

# Denoise
denoised = AdvancedUpscaling.apply_denoising(image)

# Enhance contrast
enhanced = AdvancedUpscaling.enhance_contrast(image)
```

### Post-processing

```python
# Apply after upscaling
upscaled = AdvancedUpscaling.apply_anti_aliasing(upscaled)
upscaled = AdvancedUpscaling.enhance_edges(upscaled)
upscaled = AdvancedUpscaling.reduce_artifacts(upscaled)
```

## Best Practices

### 1. Image Size Considerations

```python
# Small images (<512x512): Use high-quality methods
if image.size[0] < 512:
    method = "realesrgan"

# Medium images (512-2048): Use balanced methods
elif image.size[0] < 2048:
    method = "hybrid"

# Large images (>2048): Use tiling
else:
    use_tiling = True
```

### 2. Scale Factor Optimization

```python
# For high scale factors (>4x), use multi-stage
if scale_factor > 4.0:
    # Two-stage: 2x then remaining
    result = hybrid._two_stage_upscale(image, scale_factor)
else:
    # Single stage
    result = upscaler.upscale(image, scale_factor)
```

### 3. Batch Processing

```python
# Optimal batch size depends on:
# - Available memory
# - Image sizes
# - Scale factor

# Small images: Larger batches
batch_size = 8 if image_size < 512 else 4

# Large images: Smaller batches
batch_size = 2 if image_size > 2048 else 4
```

### 4. Model Selection

```python
# Use detection to select model
detector = AdvancedImageDetector()
analysis = detector.analyze(image)

# Use recommended model
model = analysis.recommended_model
upscaled = manager.upscale_async(image, 4.0, model_name=model)
```

## Monitoring

### Statistics

```python
# Get manager statistics
stats = manager.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Total upscales: {stats['total_upscales']}")
print(f"Models loaded: {stats['models_loaded']}")
```

### Performance Metrics

```python
# Track processing time
import time

start = time.time()
result = await manager.upscale_async(image, 4.0)
elapsed = time.time() - start

print(f"Processing time: {elapsed:.2f}s")
print(f"Pixels per second: {image.size[0] * image.size[1] / elapsed:.0f}")
```

## Troubleshooting

### Out of Memory

```python
# Solution 1: Reduce tile size
tiler = SmartTiling(max_tile_size=256, memory_limit_mb=1024)

# Solution 2: Use CPU
manager = RealESRGANModelManager(device="cpu")

# Solution 3: Reduce cache
manager = RealESRGANModelManager(max_cached_models=1)

# Solution 4: Process sequentially
results = await manager.batch_upscale_async(
    images, 4.0, max_concurrent=1
)
```

### Slow Performance

```python
# Solution 1: Use faster model
model = "RealESRNet_x4plus"  # No GAN, faster

# Solution 2: Reduce scale factor
# Use 2x then 2x again instead of 4x directly

# Solution 3: Use hybrid with fast secondary
hybrid = HybridUpscaler(
    primary_method="realesrgan",
    secondary_method="lanczos",  # Fast
    blend_ratio=0.5  # More secondary = faster
)
```

### Quality Issues

```python
# Solution 1: Use higher quality model
model = "RealESRGAN_x4plus"  # Best quality

# Solution 2: Add post-processing
upscaled = AdvancedUpscaling.enhance_edges(upscaled)
upscaled = AdvancedUpscaling.reduce_artifacts(upscaled)

# Solution 3: Use hybrid blending
hybrid = HybridUpscaler(blend_ratio=0.8)  # More primary = better quality
```

## Examples

### Example 1: Large Image with Memory Limit

```python
from image_upscaling_ai.models import SmartTiling, RealESRGANModelManager

# Setup
manager = RealESRGANModelManager(device="cuda")
tiler = SmartTiling(memory_limit_mb=2048)

# Process
upscaled = tiler.process_tiled(
    large_image,
    scale_factor=4.0,
    upscale_func=lambda img, scale: manager.upscale_async(img, scale)
)
```

### Example 2: Quality vs Speed Trade-off

```python
from image_upscaling_ai.models import HybridUpscaler

# Fast mode
fast_hybrid = HybridUpscaler(
    primary_method="lanczos",
    secondary_method="bicubic_enhanced",
    blend_ratio=0.3
)

# Quality mode
quality_hybrid = HybridUpscaler(
    primary_method="realesrgan",
    secondary_method="opencv_edsr",
    blend_ratio=0.8
)
```

### Example 3: Adaptive Processing

```python
from image_upscaling_ai.models import AdvancedImageDetector, RealESRGANModelManager

detector = AdvancedImageDetector()
manager = RealESRGANModelManager()

# Analyze
analysis = detector.analyze(image)

# Process with recommended settings
if analysis.complexity > 0.7:
    # High complexity: use tiling
    tiler = SmartTiling(max_tile_size=512)
    result = tiler.process_tiled(image, 4.0, upscale_func)
else:
    # Low complexity: direct processing
    result = await manager.upscale_async(image, 4.0)
```


