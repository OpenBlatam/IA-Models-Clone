# Pydantic Guide for Video-OpusClip System

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Model Definitions](#model-definitions)
5. [Validation](#validation)
6. [API Integration](#api-integration)
7. [Error Handling](#error-handling)
8. [Serialization](#serialization)
9. [Configuration](#configuration)
10. [Best Practices](#best-practices)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)
13. [Examples](#examples)

## Overview

Pydantic is a data validation library using Python type annotations. In the Video-OpusClip system, we use Pydantic for:

- **Input Validation**: Ensuring all incoming data meets our requirements
- **Response Schemas**: Consistent API response formats
- **Configuration Management**: Type-safe configuration handling
- **Data Serialization**: Efficient JSON serialization/deserialization
- **Error Handling**: Structured error messages and validation feedback

### Key Benefits

- **Type Safety**: Catch errors at validation time, not runtime
- **Performance**: Fast validation with Rust-powered core
- **Developer Experience**: Excellent IDE support and error messages
- **Documentation**: Automatic API documentation generation
- **Integration**: Seamless FastAPI integration

## Installation

Pydantic is already included in the Video-OpusClip requirements:

```bash
# Check if Pydantic is installed
pip show pydantic

# Install if needed
pip install pydantic>=2.5.0
```

### Version Requirements

- **Pydantic**: >=2.5.0 (for performance and features)
- **Python**: >=3.8 (for type annotations)

## Core Concepts

### Base Models

All Video-OpusClip Pydantic models inherit from `VideoOpusClipBaseModel`:

```python
from .pydantic_models import VideoOpusClipBaseModel, Field

class MyModel(VideoOpusClipBaseModel):
    name: str = Field(..., description="Model name")
    value: int = Field(default=0, ge=0, description="Numeric value")
```

### Field Validation

Use `Field` for validation and metadata:

```python
from pydantic import Field, field_validator

class VideoRequest(VideoOpusClipBaseModel):
    youtube_url: str = Field(
        ..., 
        min_length=1,
        max_length=2048,
        description="YouTube URL to process"
    )
    duration: int = Field(
        default=60,
        ge=3,
        le=600,
        description="Clip duration in seconds"
    )
    
    @field_validator('youtube_url')
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        # Custom validation logic
        if not v.startswith('https://www.youtube.com/'):
            raise ValueError('Invalid YouTube URL')
        return v
```

### Model Validation

Use `model_validator` for cross-field validation:

```python
from pydantic import model_validator

class ClipRequest(VideoOpusClipBaseModel):
    min_duration: int = Field(ge=3, le=600)
    max_duration: int = Field(ge=3, le=600)
    
    @model_validator(mode='after')
    def validate_duration_logic(self) -> 'ClipRequest':
        if self.min_duration >= self.max_duration:
            raise ValueError("min_duration must be less than max_duration")
        return self
```

## Model Definitions

### Request Models

#### VideoClipRequest

```python
from .pydantic_models import VideoClipRequest, VideoQuality, VideoFormat

# Create a video clip request
request = VideoClipRequest(
    youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language="en",
    max_clip_length=60,
    quality=VideoQuality.HIGH,
    format=VideoFormat.MP4
)

# Access computed fields
print(f"Video ID: {request.video_id}")
print(f"Request Hash: {request.request_hash}")
```

#### ViralVideoRequest

```python
from .pydantic_models import ViralVideoRequest, EngagementType, ContentType

# Create a viral video request
viral_request = ViralVideoRequest(
    youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    n_variants=5,
    use_langchain=True,
    viral_optimization=True,
    engagement_focus=EngagementType.VIRAL_POTENTIAL,
    content_type=ContentType.ENTERTAINMENT
)
```

#### BatchVideoRequest

```python
from .pydantic_models import BatchVideoRequest, ProcessingPriority

# Create batch request
requests = [
    VideoClipRequest(youtube_url="https://youtube.com/watch?v=1", language="en"),
    VideoClipRequest(youtube_url="https://youtube.com/watch?v=2", language="es")
]

batch_request = BatchVideoRequest(
    requests=requests,
    priority=ProcessingPriority.HIGH,
    max_workers=8
)

print(f"Total requests: {batch_request.total_requests}")
print(f"Estimated time: {batch_request.estimated_processing_time}")
```

### Response Models

#### VideoClipResponse

```python
from .pydantic_models import VideoClipResponse, VideoStatus

# Create response
response = VideoClipResponse(
    success=True,
    clip_id="clip_123",
    title="Processed Video",
    duration=45.5,
    file_size=10485760,  # 10MB
    processing_time=12.3,
    status=VideoStatus.COMPLETED
)

# Access computed fields
print(f"File size (MB): {response.file_size_mb}")
print(f"Is successful: {response.is_successful}")
print(f"Has warnings: {response.has_warnings}")
```

#### ViralVideoBatchResponse

```python
from .pydantic_models import ViralVideoBatchResponse, ViralVideoVariant

# Create viral response
variants = [
    ViralVideoVariant(
        variant_id="var_1",
        title="Viral Variant 1",
        viral_score=0.85,
        engagement_prediction=0.78,
        retention_score=0.92,
        duration=45.0
    )
]

viral_response = ViralVideoBatchResponse(
    success=True,
    original_clip_id="clip_123",
    variants=variants,
    total_variants_generated=1,
    successful_variants=1
)

# Access computed fields
print(f"Success rate: {viral_response.success_rate}")
print(f"Best variant: {viral_response.best_variant.variant_id if viral_response.best_variant else 'None'}")
print(f"High performing: {len(viral_response.high_performing_variants)}")
```

## Validation

### Built-in Validation

Pydantic provides extensive built-in validation:

```python
from .pydantic_models import VideoClipRequest

# Type validation
request = VideoClipRequest(
    youtube_url="https://youtube.com/watch?v=123",
    language="en",
    max_clip_length=60  # Must be int, between 3-600
)

# Enum validation
from .pydantic_models import VideoQuality
request.quality = VideoQuality.HIGH  # Must be valid enum value
```

### Custom Validation

```python
from pydantic import field_validator, ValidationError

class CustomVideoRequest(VideoOpusClipBaseModel):
    youtube_url: str
    
    @field_validator('youtube_url')
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        if not v.startswith('https://www.youtube.com/'):
            raise ValueError('Must be a YouTube URL')
        return v

# Usage
try:
    request = CustomVideoRequest(youtube_url="https://youtube.com/watch?v=123")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Validation Utilities

```python
from .pydantic_models import validate_video_request, validate_batch_request

# Validate single request
request = VideoClipRequest(youtube_url="https://youtube.com/watch?v=123", language="en")
validation_result = validate_video_request(request)

print(f"Valid: {validation_result.is_valid}")
print(f"Score: {validation_result.overall_score}")
print(f"Errors: {validation_result.errors}")

# Validate batch request
batch_request = BatchVideoRequest(requests=[request])
batch_validation = validate_batch_request(batch_request)

print(f"Batch valid: {batch_validation.is_valid}")
print(f"Valid videos: {batch_validation.valid_videos}")
```

### YouTube URL Validation

```python
from .pydantic_models import YouTubeUrlValidator

# Validate URL
try:
    valid_url = YouTubeUrlValidator.validate_youtube_url(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    video_id = YouTubeUrlValidator.extract_video_id(valid_url)
    print(f"Video ID: {video_id}")
except ValueError as e:
    print(f"Invalid URL: {e}")
```

## API Integration

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from .pydantic_models import VideoClipRequest, VideoClipResponse
from .pydantic_integration import VideoOpusClipPydanticIntegration

app = FastAPI()
integration = VideoOpusClipPydanticIntegration()

@app.post("/api/v1/video/process", response_model=VideoClipResponse)
async def process_video(request: VideoClipRequest):
    """Process video with Pydantic validation."""
    try:
        # Process request
        result = await integration.process_request(
            request.model_dump(), 
            "video_clip"
        )
        return VideoClipResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Request Processing

```python
from .pydantic_integration import PydanticAPIIntegrator

api_integrator = PydanticAPIIntegrator()

# Process video clip
request_data = {
    "youtube_url": "https://youtube.com/watch?v=123",
    "language": "en",
    "max_clip_length": 60
}

response = await api_integrator.process_video_clip_request(request_data)
print(f"Success: {response.success}")
print(f"Clip ID: {response.clip_id}")

# Process viral video
viral_data = {
    "youtube_url": "https://youtube.com/watch?v=123",
    "n_variants": 5,
    "use_langchain": True
}

viral_response = await api_integrator.process_viral_video_request(viral_data)
print(f"Variants: {viral_response.total_variants_generated}")
```

### Validation Integration

```python
from .pydantic_integration import PydanticValidationIntegrator

validator = PydanticValidationIntegrator()

# Validate and convert request
request_data = {"youtube_url": "https://youtube.com/watch?v=123", "language": "en"}
pydantic_request = await validator.validate_and_convert_request(request_data, "video_clip")

# Additional validation
validation_result = await validator.validate_request_with_legacy_system(pydantic_request)
if not validation_result.is_valid:
    print(f"Validation failed: {validation_result.errors}")
```

## Error Handling

### Pydantic Validation Errors

```python
from pydantic import ValidationError
from .pydantic_integration import PydanticErrorIntegrator

error_integrator = PydanticErrorIntegrator()

try:
    request = VideoClipRequest(
        youtube_url="invalid-url",
        language="invalid_lang",
        max_clip_length=-10
    )
except ValidationError as e:
    # Convert to ErrorInfo
    error_info = error_integrator.convert_pydantic_error(e)
    print(f"Error: {error_info.error_message}")
    print(f"Field: {error_info.field_name}")
```

### Custom Error Handling

```python
from .pydantic_models import ErrorInfo

# Create error info
error_info = ErrorInfo(
    error_code="CUSTOM_ERROR",
    error_message="This is a custom error",
    error_type="validation_error",
    field_name="youtube_url",
    field_value="invalid-url",
    request_id="req_123",
    additional_context={"source": "example"}
)

print(f"Error: {error_info.error_message}")
print(f"Code: {error_info.error_code}")
print(f"Context: {error_info.additional_context}")
```

### API Error Responses

```python
from fastapi import HTTPException
from .pydantic_integration import PydanticErrorIntegrator

error_integrator = PydanticErrorIntegrator()

@app.post("/api/v1/video/process")
async def process_video(request_data: dict):
    try:
        # Process request
        pass
    except ValidationError as e:
        error_response = error_integrator.handle_pydantic_error(e, "request_processing")
        raise HTTPException(status_code=400, detail=error_response)
```

## Serialization

### Basic Serialization

```python
from .pydantic_models import VideoClipResponse
from .pydantic_integration import PydanticSerializationIntegrator

serializer = PydanticSerializationIntegrator()

# Create response
response = VideoClipResponse(
    success=True,
    clip_id="clip_123",
    duration=45.5,
    processing_time=12.3
)

# Serialize for API
api_data = serializer.serialize_for_api(response)
print(json.dumps(api_data, indent=2))

# Serialize for cache
cache_data = serializer.serialize_for_cache(response)
print(f"Cache data: {cache_data[:100]}...")

# Deserialize from cache
deserialized = serializer.deserialize_from_cache(cache_data, VideoClipResponse)
if deserialized:
    print(f"Deserialized: {deserialized.clip_id}")
```

### Model Serialization

```python
# Model to dict
request = VideoClipRequest(youtube_url="https://youtube.com/watch?v=123", language="en")
data = request.model_dump()
print(f"Dict: {data}")

# Model to JSON
json_data = request.model_dump_json()
print(f"JSON: {json_data}")

# Dict to model
new_request = VideoClipRequest.model_validate(data)
print(f"Recreated: {new_request.youtube_url}")
```

## Configuration

### Processing Configuration

```python
from .pydantic_models import VideoProcessingConfig, ViralProcessingConfig

# Basic config
config = VideoProcessingConfig(
    target_quality=VideoQuality.ULTRA,
    target_format=VideoFormat.MP4,
    target_resolution="1920x1080",
    max_workers=16,
    use_gpu=True
)

print(f"Quality: {config.target_quality}")
print(f"Workers: {config.max_workers}")

# Viral config
viral_config = ViralProcessingConfig(
    viral_optimization_enabled=True,
    use_langchain=True,
    langchain_model="gpt-4",
    min_viral_score=0.7,
    max_variants=15
)
```

### Configuration Integration

```python
from .pydantic_integration import PydanticConfigIntegrator

config_integrator = PydanticConfigIntegrator()

# Create config from dict
config_data = {
    "target_quality": "ultra",
    "max_workers": 16,
    "use_gpu": True
}
config = config_integrator.create_config_from_dict(config_data)

# Merge configs
override_data = {"max_workers": 32}
merged_config = config_integrator.merge_configs(config, override_data)

# Validate config
is_valid = config_integrator.validate_config(merged_config)
print(f"Config valid: {is_valid}")
```

## Best Practices

### Model Design

1. **Use descriptive field names**
```python
# Good
class VideoRequest(VideoOpusClipBaseModel):
    youtube_url: str = Field(..., description="YouTube video URL")
    max_duration: int = Field(..., description="Maximum clip duration in seconds")

# Avoid
class VideoRequest(VideoOpusClipBaseModel):
    url: str
    max_len: int
```

2. **Provide field descriptions**
```python
class VideoRequest(VideoOpusClipBaseModel):
    youtube_url: str = Field(
        ..., 
        description="YouTube video URL to process",
        example="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
```

3. **Use appropriate field types**
```python
from typing import Optional, List, Dict, Any

class VideoRequest(VideoOpusClipBaseModel):
    required_field: str  # Required
    optional_field: Optional[str] = None  # Optional
    list_field: List[str] = Field(default_factory=list)  # List with default
    dict_field: Dict[str, Any] = Field(default_factory=dict)  # Dict with default
```

### Validation

1. **Use built-in validators when possible**
```python
class VideoRequest(VideoOpusClipBaseModel):
    duration: int = Field(ge=3, le=600)  # Built-in range validation
    quality: VideoQuality  # Enum validation
    tags: List[str] = Field(max_length=50)  # List length validation
```

2. **Create reusable validators**
```python
from pydantic import field_validator

def validate_youtube_url(url: str) -> str:
    """Reusable YouTube URL validator."""
    if not url.startswith('https://www.youtube.com/'):
        raise ValueError('Invalid YouTube URL')
    return url

class VideoRequest(VideoOpusClipBaseModel):
    youtube_url: str
    
    @field_validator('youtube_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_youtube_url(v)
```

3. **Use model validators for cross-field validation**
```python
@model_validator(mode='after')
def validate_duration_logic(self) -> 'VideoRequest':
    if self.min_duration >= self.max_duration:
        raise ValueError("min_duration must be less than max_duration")
    return self
```

### Error Handling

1. **Catch specific exceptions**
```python
try:
    request = VideoClipRequest(**data)
except ValidationError as e:
    # Handle validation errors
    handle_validation_error(e)
except Exception as e:
    # Handle other errors
    handle_general_error(e)
```

2. **Provide helpful error messages**
```python
@field_validator('youtube_url')
@classmethod
def validate_youtube_url(cls, v: str) -> str:
    if not v.startswith('https://www.youtube.com/'):
        raise ValueError(
            'Invalid YouTube URL. Must start with https://www.youtube.com/'
        )
    return v
```

3. **Use error codes for programmatic handling**
```python
from .pydantic_models import ErrorInfo

error_info = ErrorInfo(
    error_code="INVALID_YOUTUBE_URL",
    error_message="Invalid YouTube URL format",
    error_type="validation_error"
)
```

### Performance

1. **Use computed fields for derived values**
```python
@computed_field
@property
def file_size_mb(self) -> Optional[float]:
    """Get file size in megabytes."""
    if self.file_size is not None:
        return round(self.file_size / (1024 * 1024), 2)
    return None
```

2. **Cache validation results**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def validate_youtube_url_cached(url: str) -> str:
    return validate_youtube_url(url)
```

3. **Use efficient serialization**
```python
# Use model_dump() for dict conversion
data = model.model_dump()

# Use model_dump_json() for JSON serialization
json_data = model.model_dump_json()

# Use model_validate() for dict to model conversion
model = ModelClass.model_validate(data)
```

## Performance Optimization

### Validation Performance

```python
import time
from .pydantic_models import VideoClipRequest

# Benchmark validation
start_time = time.time()
for i in range(1000):
    request = VideoClipRequest(
        youtube_url=f"https://youtube.com/watch?v=test{i}",
        language="en",
        max_clip_length=60
    )
validation_time = time.time() - start_time
print(f"Validated 1000 requests in {validation_time:.3f} seconds")
```

### Serialization Performance

```python
# Benchmark serialization
response = VideoClipResponse(
    success=True,
    clip_id="clip_123",
    duration=45.5,
    processing_time=12.3
)

start_time = time.time()
for i in range(1000):
    data = response.model_dump()
serialization_time = time.time() - start_time
print(f"Serialized 1000 responses in {serialization_time:.3f} seconds")
```

### Memory Optimization

```python
# Use slots for better memory usage
class OptimizedModel(VideoOpusClipBaseModel):
    model_config = ConfigDict(
        validate_assignment=False,  # Disable assignment validation
        copy_on_model_validation=False  # Disable copying
    )
```

## Troubleshooting

### Common Issues

1. **Validation Errors**
```python
# Problem: ValidationError not caught
try:
    request = VideoClipRequest(**data)
except Exception as e:  # Too broad
    print(f"Error: {e}")

# Solution: Catch specific exception
try:
    request = VideoClipRequest(**data)
except ValidationError as e:
    print(f"Validation error: {e}")
```

2. **Field Type Errors**
```python
# Problem: Wrong field type
class Model(VideoOpusClipBaseModel):
    value: int = Field(default="string")  # Wrong type

# Solution: Correct field type
class Model(VideoOpusClipBaseModel):
    value: int = Field(default=0)  # Correct type
```

3. **Circular Imports**
```python
# Problem: Circular import
from .models import VideoClipRequest
from .api import process_video

# Solution: Use forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models import VideoClipRequest
```

### Debugging

1. **Enable debug mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Use model validation debugging**
```python
# Print validation errors
try:
    request = VideoClipRequest(**data)
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Type: {error['type']}")
```

3. **Validate individual fields**
```python
# Validate specific field
from pydantic import field_validator

@field_validator('youtube_url')
@classmethod
def validate_youtube_url(cls, v: str) -> str:
    print(f"Validating URL: {v}")  # Debug print
    # Validation logic
    return v
```

## Examples

### Complete API Example

```python
from fastapi import FastAPI, HTTPException
from .pydantic_models import VideoClipRequest, VideoClipResponse
from .pydantic_integration import VideoOpusClipPydanticIntegration

app = FastAPI()
integration = VideoOpusClipPydanticIntegration()

@app.post("/api/v1/video/process", response_model=VideoClipResponse)
async def process_video(request: VideoClipRequest):
    """Process video with full Pydantic integration."""
    try:
        # Process request
        result = await integration.process_request(
            request.model_dump(), 
            "video_clip"
        )
        
        # Return response
        return VideoClipResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=400, 
            detail={"error": "Validation failed", "details": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={"error": "Processing failed", "details": str(e)}
        )

@app.post("/api/v1/viral/process", response_model=ViralVideoBatchResponse)
async def process_viral_video(request: ViralVideoRequest):
    """Process viral video with Pydantic validation."""
    try:
        result = await integration.process_request(
            request.model_dump(), 
            "viral"
        )
        return ViralVideoBatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Configuration Example

```python
from .pydantic_models import VideoProcessingConfig, ViralProcessingConfig

# Load configuration
config = VideoProcessingConfig(
    target_quality=VideoQuality.ULTRA,
    target_format=VideoFormat.MP4,
    max_workers=16,
    use_gpu=True,
    optimize_for_web=True
)

# Viral configuration
viral_config = ViralProcessingConfig(
    viral_optimization_enabled=True,
    use_langchain=True,
    langchain_model="gpt-4",
    min_viral_score=0.7,
    max_variants=15,
    cache_enabled=True
)

# Validate configurations
if not config_integrator.validate_config(config):
    raise ValueError("Invalid processing configuration")

if not config_integrator.validate_config(viral_config):
    raise ValueError("Invalid viral configuration")
```

### Error Handling Example

```python
from .pydantic_integration import PydanticErrorIntegrator

error_integrator = PydanticErrorIntegrator()

def process_user_request(request_data: dict):
    """Process user request with comprehensive error handling."""
    try:
        # Validate request
        request = VideoClipRequest(**request_data)
        
        # Process request
        result = process_video(request)
        
        return {"success": True, "result": result}
        
    except ValidationError as e:
        # Handle validation errors
        error_response = error_integrator.handle_pydantic_error(e, "user_request")
        return error_response
        
    except Exception as e:
        # Handle other errors
        return {
            "success": False,
            "error": {
                "code": "PROCESSING_ERROR",
                "message": str(e),
                "type": "processing_error"
            }
        }
```

This comprehensive guide covers all aspects of using Pydantic with the Video-OpusClip system, from basic usage to advanced optimization techniques. 