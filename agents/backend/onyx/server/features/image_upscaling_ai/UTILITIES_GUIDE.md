# Utilities Guide

## Available Utilities

### 1. Image Utils (`utils/image_utils.py`)

Image manipulation utilities.

```python
from image_upscaling_ai.utils import ImageUtils

# Ensure RGB
image = ImageUtils.ensure_rgb(image)

# Get info
info = ImageUtils.get_image_info(image)

# Resize maintaining aspect
resized = ImageUtils.resize_maintain_aspect(image, (1024, 1024))

# Pad to size
padded = ImageUtils.pad_to_size(image, (512, 512))

# Crop to aspect
cropped = ImageUtils.crop_to_aspect(image, 16/9)

# Enhance
enhanced = ImageUtils.enhance_image(
    image,
    brightness=1.1,
    contrast=1.2,
    saturation=1.1,
    sharpness=1.15
)

# Compare images
comparison = ImageUtils.compare_images(image1, image2)
print(f"Similarity: {comparison['similarity']:.2f}")

# Create thumbnail
thumbnail = ImageUtils.create_thumbnail(image, (256, 256))

# Split into tiles
tiles = ImageUtils.split_into_tiles(image, (512, 512), overlap=32)

# Combine tiles
combined = ImageUtils.combine_tiles(tiles, (2048, 2048))
```

### 2. Config Validator (`utils/config_validator.py`)

Configuration validation and optimization.

```python
from image_upscaling_ai.utils import ConfigValidator

# Validate config
result = ConfigValidator.validate_config(config)
if not result["valid"]:
    print("Issues:", result["issues"])
    print("Suggestions:", result["suggestions"])

# Optimize config
optimized = ConfigValidator.optimize_config(config)

# Get recommended config
recommended = ConfigValidator.get_recommended_config("production")
```

### 3. Error Recovery (`utils/error_recovery.py`)

Advanced error recovery mechanisms.

```python
from image_upscaling_ai.utils import retry_with_backoff, fallback_on_error, ErrorRecovery

# Retry decorator
@retry_with_backoff(max_attempts=3, initial_delay=1.0)
async def upscale_image(image):
    return await upscaler.upscale(image, 4.0)

# Fallback decorator
def fallback_upscale(image):
    return image.resize((image.width * 2, image.height * 2))

@fallback_on_error(fallback_upscale)
async def upscale_with_fallback(image):
    return await upscaler.upscale(image, 4.0)

# Error recovery manager
recovery = ErrorRecovery()

def recovery_strategy(error, context):
    # Custom recovery logic
    return fallback_result

recovery.register_strategy(ValueError, recovery_strategy)
result = await recovery.recover(error, context)
```

### 4. Performance Profiler (`utils/performance_profiler.py`)

Performance profiling and analysis.

```python
from image_upscaling_ai.utils import profile_context, profile_function, PerformanceProfiler

# Context manager
with profile_context("profile.stats"):
    result = await upscale_image(image)

# Function decorator
@profile_function("upscale_profile.stats")
async def upscale_image(image):
    return await upscaler.upscale(image, 4.0)

# Profiler class
profiler = PerformanceProfiler()
results = profiler.profile(upscale_func, image)
bottlenecks = profiler.get_bottlenecks(results)
```

## Usage Examples

### Example 1: Image Processing

```python
from image_upscaling_ai.utils import ImageUtils
from PIL import Image

# Load and process
image = Image.open("input.jpg")

# Get information
info = ImageUtils.get_image_info(image)
print(f"Size: {info['size']}, Mode: {info['mode']}")

# Enhance before upscaling
enhanced = ImageUtils.enhance_image(
    image,
    brightness=1.05,
    contrast=1.1,
    sharpness=1.1
)

# Upscale (using service)
# ... upscale ...

# Compare results
comparison = ImageUtils.compare_images(original, upscaled)
print(f"Similarity: {comparison['similarity']:.2f}")
```

### Example 2: Configuration

```python
from image_upscaling_ai.utils import ConfigValidator
from image_upscaling_ai.config.upscaling_config import UpscalingConfig

# Get config
config = UpscalingConfig.from_env()

# Validate
validation = ConfigValidator.validate_config(config.dict())
if validation["warnings"]:
    for warning in validation["warnings"]:
        print(f"⚠️ {warning}")

# Optimize
optimized_dict = ConfigValidator.optimize_config(config.dict())

# Get recommended
recommended = ConfigValidator.get_recommended_config("production")
```

### Example 3: Error Handling

```python
from image_upscaling_ai.utils import retry_with_backoff, ErrorRecovery

# Retry with backoff
@retry_with_backoff(max_attempts=3, initial_delay=1.0)
async def robust_upscale(image):
    return await upscaler.upscale(image, 4.0)

# Error recovery
recovery = ErrorRecovery()

def fallback_strategy(error, context):
    # Use simpler method
    return simple_upscale(context["image"])

recovery.register_strategy(ValueError, fallback_strategy)

try:
    result = await upscaler.upscale(image, 4.0)
except Exception as e:
    result = await recovery.recover(e, {"image": image})
```

### Example 4: Performance Profiling

```python
from image_upscaling_ai.utils import profile_context

# Profile operation
with profile_context("upscale_profile.stats"):
    result = await service.upscale_image_enhanced(image, 4.0)

# Analyze profile
# Use: python -m pstats upscale_profile.stats
```

## Best Practices

1. **Use ImageUtils** for common image operations
2. **Validate config** before initialization
3. **Use error recovery** for robust operations
4. **Profile** to identify bottlenecks
5. **Optimize config** based on system resources

## Summary

Utilities provide:
- ✅ Image manipulation helpers
- ✅ Configuration validation
- ✅ Error recovery mechanisms
- ✅ Performance profiling

Use these utilities to enhance your upscaling workflows!


