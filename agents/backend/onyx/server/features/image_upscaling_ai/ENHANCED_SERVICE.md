# Enhanced Service Guide

## Overview

The Enhanced Upscaling Service integrates all intelligent features into a single, easy-to-use service.

## Features

### Integrated Components

- ✅ Smart Recommendations
- ✅ Adaptive Learning
- ✅ Real-Time Monitoring
- ✅ Quality Validation
- ✅ Performance Optimization
- ✅ Feedback Collection
- ✅ Advanced Metrics
- ✅ Intelligent Caching

## Usage

### Basic Usage

```python
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService

service = EnhancedUpscalingService()

# Upscale with all features
result = await service.upscale_image_enhanced(
    "image.jpg",
    scale_factor=4.0
)

print(f"Quality: {result['quality_score']:.2f}")
print(f"Method: {result['method_used']}")
print(f"Time: {result['processing_time']:.2f}s")
```

### With Recommendations

```python
# Automatically gets best method
result = await service.upscale_image_enhanced(
    "image.jpg",
    use_recommendations=True
)

print(f"Recommended method: {result['recommendation']['method']}")
print(f"Expected quality: {result['recommendation']['expected_quality']:.2f}")
```

### With Feedback

```python
# Upscale
result = await service.upscale_image_enhanced("image.jpg", 4.0)

# Submit feedback
service.submit_feedback(
    operation_id=result["operation_id"],
    satisfaction=0.9,
    quality_rating=0.95,
    speed_rating=0.8,
    comments="Excellent quality!"
)
```

## API Endpoints

### Enhanced Upscaling

```bash
POST /api/v1/enhanced/upscale
```

**Parameters:**
- `image`: Image file
- `scale_factor`: Scale factor (optional)
- `use_recommendations`: Use smart recommendations (default: true)
- `validate_quality`: Validate output quality (default: true)

**Response:**
```json
{
  "success": true,
  "operation_id": "op_123",
  "upscaled_image_base64": "...",
  "original_size": [512, 512],
  "upscaled_size": [2048, 2048],
  "scale_factor": 4.0,
  "method_used": "RealESRGAN_x4plus",
  "processing_time": 2.1,
  "quality_score": 0.92,
  "quality_passed": true,
  "recommendation": {
    "method": "RealESRGAN_x4plus",
    "expected_quality": 0.90,
    "expected_time": 2.0,
    "confidence": 0.85
  },
  "analysis": {
    "image_type": "photo",
    "quality": 0.75,
    "complexity": 0.6
  }
}
```

### Submit Feedback

```bash
POST /api/v1/enhanced/feedback
```

**Parameters:**
- `operation_id`: Operation ID
- `satisfaction`: Satisfaction (0.0-1.0)
- `quality_rating`: Quality rating (0.0-1.0)
- `speed_rating`: Speed rating (0.0-1.0)
- `comments`: Optional comments

### Get Status

```bash
GET /api/v1/enhanced/status
```

**Response:**
```json
{
  "system_metrics": {
    "total_operations": 150,
    "success_rate": 0.98,
    "avg_quality": 0.87,
    "throughput": 1.8,
    "cache_hit_rate": 0.65
  },
  "performance": {
    "avg_time": 2.1,
    "optimal_batch_size": 2,
    "optimal_tile_size": 512
  },
  "cache": {
    "entries": 45,
    "hit_rate": 0.65,
    "total_size_mb": 850
  },
  "feedback": {
    "total_feedback": 30,
    "avg_satisfaction": 0.88,
    "avg_quality": 0.90
  }
}
```

### Get Recommendations

```bash
POST /api/v1/enhanced/recommendations
```

**Parameters:**
- `image`: Image file
- `target_scale`: Target scale factor
- `prioritize_speed`: Prioritize speed

**Response:**
```json
{
  "method": "RealESRGAN_x4plus",
  "scale_factor": 4.0,
  "preprocessing_mode": "auto",
  "postprocessing_mode": "auto",
  "expected_quality": 0.90,
  "expected_time": 2.0,
  "confidence": 0.85,
  "reasoning": [
    "Image type: photo (confidence: 0.82)",
    "Complexity: 0.60, Quality: 0.75",
    "Learned recommendation: RealESRGAN_x4plus (confidence: 0.85)"
  ]
}
```

## Complete Example

```python
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService

# Initialize
service = EnhancedUpscalingService()

# Upscale with all features
result = await service.upscale_image_enhanced(
    "photo.jpg",
    use_recommendations=True,
    validate_quality=True
)

# Check result
if result["success"]:
    print(f"✅ Success!")
    print(f"Method: {result['method_used']}")
    print(f"Quality: {result['quality_score']:.2f}")
    print(f"Time: {result['processing_time']:.2f}s")
    
    # Save result
    result["upscaled_image"].save("upscaled.jpg")
    
    # Submit feedback
    service.submit_feedback(
        result["operation_id"],
        satisfaction=0.95,
        quality_rating=0.92,
        speed_rating=0.85
    )
else:
    print(f"❌ Error: {result['error']}")

# Get system status
status = service.get_system_status()
print(f"System throughput: {status['system_metrics']['throughput']:.2f} ops/s")
print(f"Cache hit rate: {status['cache']['hit_rate']:.2%}")
```

## Benefits

1. **Single Interface**: All features in one service
2. **Automatic Optimization**: Learns and improves
3. **Quality Assurance**: Validates every result
4. **Performance Tracking**: Monitors and optimizes
5. **User Feedback**: Continuous improvement

## Integration

The enhanced service can be used as a drop-in replacement:

```python
# Old way
from image_upscaling_ai.core.upscaling_service import UpscalingService
service = UpscalingService()
result = await service.upscale_image("image.jpg", 4.0)

# New way (with all features)
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
service = EnhancedUpscalingService()
result = await service.upscale_image_enhanced("image.jpg", 4.0)
```

## Summary

The Enhanced Service provides:
- ✅ All intelligent features integrated
- ✅ Easy-to-use interface
- ✅ Automatic optimization
- ✅ Quality assurance
- ✅ Performance monitoring
- ✅ Feedback collection

Use this for production deployments with maximum quality and efficiency!


