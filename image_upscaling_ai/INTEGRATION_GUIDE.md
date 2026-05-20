# Integration Guide

## Overview

This guide covers integrating the image upscaling system with other applications and services.

## Integration Helper

### Format Conversion

```python
from image_upscaling_ai.core.integration_helper import IntegrationHelper

# Convert to base64
base64_str = IntegrationHelper.image_to_base64(image)

# Convert from base64
image = IntegrationHelper.base64_to_image(base64_str)

# Convert to bytes
image_bytes = IntegrationHelper.image_to_bytes(image)

# Convert from bytes
image = IntegrationHelper.bytes_to_image(image_bytes)
```

### Batch Preparation

```python
# Prepare batch request
requests = IntegrationHelper.prepare_batch_request(
    image_paths,
    scale_factor=4.0,
    use_ai=True
)

# Process batch
for request in requests:
    result = await service.upscale_image_enhanced(**request)
```

### Response Formatting

```python
# Format response for API
formatted = IntegrationHelper.format_response(
    result,
    include_image=True,
    image_format="base64"  # or "bytes", "path"
)
```

## Format Converter

### Convert Between Formats

```python
from image_upscaling_ai.utils import FormatConverter

# PIL to NumPy
numpy_array = FormatConverter.pil_to_numpy(image)

# NumPy to PIL
image = FormatConverter.numpy_to_pil(numpy_array)

# PIL to OpenCV
cv_image = FormatConverter.pil_to_cv2(image)

# OpenCV to PIL
image = FormatConverter.cv2_to_pil(cv_image)

# Universal converter
result = FormatConverter.convert_format(
    image,
    target_format="numpy"  # or "pil", "cv2", "base64", "bytes"
)
```

## Batch Helper

### Prepare Images

```python
from image_upscaling_ai.utils import BatchHelper

# Prepare images from paths
images = BatchHelper.prepare_images(
    image_paths,
    validate=True
)
```

### Process with Progress

```python
# Process batch with progress
results = await BatchHelper.process_batch_with_progress(
    images,
    upscale_func,
    progress_callback=lambda current, total: print(f"{current}/{total}"),
    max_concurrent=2
)

# Aggregate results
stats = BatchHelper.aggregate_results(results)
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Integration Examples

### Example 1: Web Application

```python
from fastapi import FastAPI, File, UploadFile
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
from image_upscaling_ai.core.integration_helper import IntegrationHelper

app = FastAPI()
service = EnhancedUpscalingService()

@app.post("/upscale")
async def upscale_endpoint(file: UploadFile = File(...)):
    # Load image
    image_bytes = await file.read()
    image = IntegrationHelper.bytes_to_image(image_bytes)
    
    # Upscale
    result = await service.upscale_image_enhanced(image, 4.0)
    
    # Format response
    return IntegrationHelper.format_response(
        result,
        include_image=True,
        image_format="base64"
    )
```

### Example 2: Microservice

```python
import requests
from image_upscaling_ai.core.integration_helper import IntegrationHelper

def upscale_via_service(image_path: str, service_url: str):
    """Upscale image via microservice."""
    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Send request
    response = requests.post(
        f"{service_url}/api/v1/enhanced/upscale",
        files={"image": image_bytes},
        data={"scale_factor": 4.0}
    )
    
    result = response.json()
    
    # Decode image
    if result["success"]:
        image = IntegrationHelper.base64_to_image(
            result["upscaled_image_base64"]
        )
        return image
    
    return None
```

### Example 3: Batch Processing Service

```python
from image_upscaling_ai.utils import BatchHelper
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService

service = EnhancedUpscalingService()

async def process_batch_service(image_paths: List[str]):
    """Process batch of images."""
    # Prepare images
    images = BatchHelper.prepare_images(image_paths)
    
    # Process function
    async def upscale(img):
        result = await service.upscale_image_enhanced(img, 4.0)
        return result["upscaled_image"] if result["success"] else None
    
    # Process with progress
    results = await BatchHelper.process_batch_with_progress(
        images,
        upscale,
        progress_callback=update_progress,
        max_concurrent=2
    )
    
    # Aggregate
    stats = BatchHelper.aggregate_results(results)
    return results, stats
```

### Example 4: Format Conversion Pipeline

```python
from image_upscaling_ai.utils import FormatConverter

# Input: NumPy array
numpy_image = np.array(...)

# Convert to PIL
pil_image = FormatConverter.convert_format(numpy_image, "pil")

# Upscale (using service)
upscaled = await service.upscale_image_enhanced(pil_image, 4.0)

# Convert back to NumPy
upscaled_numpy = FormatConverter.convert_format(
    upscaled["upscaled_image"],
    "numpy"
)
```

## Best Practices

1. **Use IntegrationHelper** for format conversions
2. **Use FormatConverter** for complex conversions
3. **Use BatchHelper** for batch operations
4. **Format responses** consistently
5. **Handle errors** gracefully

## Summary

Integration tools provide:
- ✅ Format conversion utilities
- ✅ Batch processing helpers
- ✅ Response formatting
- ✅ Easy integration with other systems

Use these tools for seamless integration!


