# Examples Guide

## Available Examples

### 1. Simple Example (`examples/simple_example.py`)

Basic upscaling with minimal code.

```python
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService

service = EnhancedUpscalingService()
result = await service.upscale_image_enhanced("image.jpg", 4.0)
result["upscaled_image"].save("output.jpg")
```

**Run:**
```bash
python examples/simple_example.py
```

### 2. Complete Example (`examples/complete_example.py`)

Comprehensive example showing all features:
- Basic upscaling
- Enhanced upscaling
- Real-ESRGAN
- Smart recommendations
- Batch processing
- Quality validation
- Real-time monitoring
- System metrics
- Complete workflow

**Run:**
```bash
python examples/complete_example.py
```

### 3. Batch Example (`examples/batch_example.py`)

Batch processing multiple images.

```python
from image_upscaling_ai.models import BatchOptimizer, RealESRGANModelManager

manager = RealESRGANModelManager()
optimizer = BatchOptimizer()

result = await optimizer.process_batch_optimized(images, upscale_func)
```

**Run:**
```bash
python examples/batch_example.py
```

### 4. API Example (`examples/api_example.py`)

Using the REST API.

```python
import requests

# Upscale via API
response = requests.post(
    "http://localhost:8003/api/v1/enhanced/upscale",
    files={"image": open("input.jpg", "rb")},
    data={"scale_factor": 4.0}
)
```

**Run:**
```bash
# Start server first
uvicorn image_upscaling_ai.api.upscaling_api:app --host 0.0.0.0 --port 8003

# Then run example
python examples/api_example.py
```

## Quick Start Examples

### Example 1: Basic Upscaling

```python
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
import asyncio

async def main():
    service = EnhancedUpscalingService()
    result = await service.upscale_image_enhanced("input.jpg", 4.0)
    result["upscaled_image"].save("output.jpg")

asyncio.run(main())
```

### Example 2: With Recommendations

```python
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService

service = EnhancedUpscalingService()
result = await service.upscale_image_enhanced(
    "input.jpg",
    use_recommendations=True
)

print(f"Method: {result['recommendation']['method']}")
print(f"Expected quality: {result['recommendation']['expected_quality']:.2f}")
```

### Example 3: Batch Processing

```python
from image_upscaling_ai.models import BatchOptimizer, RealESRGANModelManager

manager = RealESRGANModelManager()
optimizer = BatchOptimizer()

images = [Image.open(f"img_{i}.jpg") for i in range(10)]

async def upscale(img):
    return await manager.upscale_async(img, 4.0)

result = await optimizer.process_batch_optimized(images, upscale)
```

### Example 4: Quality Validation

```python
from image_upscaling_ai.models import QualityValidator

validator = QualityValidator(min_score=0.7)
report = validator.validate(upscaled, original, 4.0)

if report.passed:
    print(f"✅ Quality: {report.overall_score:.2f}")
else:
    print(f"❌ Issues: {report.issues}")
```

### Example 5: Real-Time Monitoring

```python
from image_upscaling_ai.models import RealtimeAnalyzer

analyzer = RealtimeAnalyzer()

def on_progress(metrics):
    print(f"{metrics.stage}: {metrics.progress:.1%}")

analyzer.add_progress_callback(on_progress)
analyzer.start_operation()
# ... process ...
summary = analyzer.finish_operation()
```

## API Examples

### cURL Examples

```bash
# Upscale image
curl -X POST "http://localhost:8003/api/v1/enhanced/upscale" \
  -F "image=@input.jpg" \
  -F "scale_factor=4.0"

# Get recommendations
curl -X POST "http://localhost:8003/api/v1/enhanced/recommendations" \
  -F "image=@input.jpg" \
  -F "target_scale=4.0"

# Get status
curl "http://localhost:8003/api/v1/enhanced/status"
```

### Python Requests

```python
import requests

# Upscale
with open("input.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8003/api/v1/enhanced/upscale",
        files={"image": f},
        data={"scale_factor": 4.0}
    )
    result = response.json()
```

## Best Practices

1. **Always use Enhanced Service** for production
2. **Enable recommendations** for best results
3. **Validate quality** for critical operations
4. **Use batch processing** for multiple images
5. **Monitor performance** with metrics
6. **Collect feedback** for improvement

## Troubleshooting

### Import Errors

```bash
# Install dependencies
pip install -r requirements.txt

# Install Real-ESRGAN (optional)
pip install realesrgan basicsr
```

### File Not Found

```python
# Check if file exists
from pathlib import Path
if not Path("input.jpg").exists():
    print("File not found!")
```

### API Connection Error

```bash
# Start server
uvicorn image_upscaling_ai.api.upscaling_api:app --host 0.0.0.0 --port 8003
```

## Summary

All examples are ready to use:
- ✅ Simple examples for quick start
- ✅ Complete examples for all features
- ✅ Batch processing examples
- ✅ API usage examples
- ✅ Best practices included

Start with `simple_example.py` for quick testing!


