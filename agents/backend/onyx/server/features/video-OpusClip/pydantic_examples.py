"""
Pydantic Usage Examples for Video-OpusClip System

Comprehensive examples demonstrating Pydantic model usage, validation,
API integration, error handling, and best practices.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import structlog

from .pydantic_models import (
    VideoClipRequest,
    VideoClipResponse,
    ViralVideoRequest,
    ViralVideoBatchResponse,
    BatchVideoRequest,
    BatchVideoResponse,
    VideoValidationResult,
    BatchValidationResult,
    VideoProcessingConfig,
    ViralProcessingConfig,
    ProcessingMetrics,
    ErrorInfo,
    VideoStatus,
    VideoQuality,
    VideoFormat,
    ProcessingPriority,
    ContentType,
    EngagementType,
    validate_video_request,
    validate_batch_request,
    create_video_clip_request,
    create_viral_video_request,
    create_batch_request,
    create_processing_config,
    YouTubeUrlValidator,
    VideoOpusClipConfig
)
from .pydantic_integration import (
    VideoOpusClipPydanticIntegration,
    PydanticValidationIntegrator,
    PydanticAPIIntegrator,
    PydanticConfigIntegrator,
    PydanticErrorIntegrator,
    PydanticSerializationIntegrator
)

logger = structlog.get_logger()

# =============================================================================
# BASIC MODEL USAGE EXAMPLES
# =============================================================================

def example_basic_model_creation():
    """Example: Basic model creation and validation."""
    print("=== Basic Model Creation ===")
    
    # Create a video clip request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60,
        quality=VideoQuality.HIGH
    )
    
    print(f"Request created: {request.youtube_url}")
    print(f"Video ID: {request.video_id}")
    print(f"Request hash: {request.request_hash}")
    print(f"Quality: {request.quality}")
    print(f"Format: {request.format}")
    print()

def example_model_validation():
    """Example: Model validation with error handling."""
    print("=== Model Validation ===")
    
    # Valid request
    try:
        valid_request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language="en",
            max_clip_length=60
        )
        print("✅ Valid request created successfully")
    except Exception as e:
        print(f"❌ Validation error: {e}")
    
    # Invalid request - wrong URL
    try:
        invalid_request = VideoClipRequest(
            youtube_url="https://invalid-url.com",
            language="en",
            max_clip_length=60
        )
        print("✅ Invalid request should have failed")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
    
    # Invalid request - unsupported language
    try:
        invalid_request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language="invalid_lang",
            max_clip_length=60
        )
        print("✅ Invalid language should have failed")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
    
    # Invalid request - duration constraints
    try:
        invalid_request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language="en",
            min_clip_length=100,
            max_clip_length=50  # min > max
        )
        print("✅ Invalid duration should have failed")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
    
    print()

def example_computed_fields():
    """Example: Using computed fields."""
    print("=== Computed Fields ===")
    
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60
    )
    
    print(f"Video ID: {request.video_id}")
    print(f"Request Hash: {request.request_hash}")
    
    # Response with computed fields
    response = VideoClipResponse(
        success=True,
        clip_id="clip_123",
        file_size=10485760,  # 10MB
        processing_time=45.5
    )
    
    print(f"File size (MB): {response.file_size_mb}")
    print(f"Is successful: {response.is_successful}")
    print(f"Has warnings: {response.has_warnings}")
    print()

def example_viral_video_models():
    """Example: Viral video models and variants."""
    print("=== Viral Video Models ===")
    
    # Create viral video request
    viral_request = ViralVideoRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        n_variants=3,
        use_langchain=True,
        viral_optimization=True,
        engagement_focus=EngagementType.VIRAL_POTENTIAL,
        content_type=ContentType.ENTERTAINMENT
    )
    
    print(f"Viral request: {viral_request.n_variants} variants")
    print(f"Engagement focus: {viral_request.engagement_focus}")
    print(f"Content type: {viral_request.content_type}")
    
    # Create viral variant
    from .pydantic_models import ViralVideoVariant
    
    variant = ViralVideoVariant(
        variant_id="variant_1",
        title="Amazing Viral Video",
        description="This will go viral!",
        viral_score=0.85,
        engagement_prediction=0.78,
        retention_score=0.92,
        duration=45.0,
        content_type=ContentType.ENTERTAINMENT,
        engagement_type=EngagementType.VIRAL_POTENTIAL,
        target_audience=["18-24", "25-34"],
        viral_hooks=["Shocking reveal", "Unexpected twist"],
        trending_elements=["#viral", "#trending"],
        hashtags=["#amazing", "#viral", "#trending"]
    )
    
    print(f"Variant overall score: {variant.overall_score}")
    print(f"Is high performing: {variant.is_high_performing}")
    print()

# =============================================================================
# VALIDATION EXAMPLES
# =============================================================================

def example_validation_utilities():
    """Example: Using validation utilities."""
    print("=== Validation Utilities ===")
    
    # Create a request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60
    )
    
    # Validate request
    validation_result = validate_video_request(request)
    
    print(f"Validation valid: {validation_result.is_valid}")
    print(f"Overall score: {validation_result.overall_score}")
    print(f"Quality score: {validation_result.quality_score}")
    print(f"Duration score: {validation_result.duration_score}")
    print(f"Format score: {validation_result.format_score}")
    
    if validation_result.has_errors:
        print("Errors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    if validation_result.has_warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    if validation_result.suggestions:
        print("Suggestions:")
        for suggestion in validation_result.suggestions:
            print(f"  - {suggestion}")
    
    print()

def example_batch_validation():
    """Example: Batch request validation."""
    print("=== Batch Validation ===")
    
    # Create batch request
    requests = [
        VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language="en",
            max_clip_length=60
        ),
        VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=9bZkp7q19f0",
            language="es",
            max_clip_length=45
        )
    ]
    
    batch_request = BatchVideoRequest(
        requests=requests,
        priority=ProcessingPriority.HIGH,
        max_workers=4
    )
    
    # Validate batch
    validation_result = validate_batch_request(batch_request)
    
    print(f"Batch validation valid: {validation_result.is_valid}")
    print(f"Valid videos: {validation_result.valid_videos}")
    print(f"Invalid videos: {validation_result.invalid_videos}")
    print(f"Validation rate: {validation_result.validation_rate}")
    print(f"Overall score: {validation_result.overall_score}")
    print()

def example_youtube_url_validation():
    """Example: YouTube URL validation."""
    print("=== YouTube URL Validation ===")
    
    # Valid URLs
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ"
    ]
    
    for url in valid_urls:
        try:
            validated_url = YouTubeUrlValidator.validate_youtube_url(url)
            video_id = YouTubeUrlValidator.extract_video_id(url)
            print(f"✅ Valid URL: {url}")
            print(f"   Video ID: {video_id}")
        except Exception as e:
            print(f"❌ Invalid URL: {url} - {e}")
    
    # Invalid URLs
    invalid_urls = [
        "https://invalid-url.com",
        "javascript:alert('xss')",
        "https://youtube.com/invalid",
        ""
    ]
    
    for url in invalid_urls:
        try:
            YouTubeUrlValidator.validate_youtube_url(url)
            print(f"❌ Should have failed: {url}")
        except Exception as e:
            print(f"✅ Correctly rejected: {url} - {e}")
    
    print()

# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

def example_configuration_models():
    """Example: Configuration models."""
    print("=== Configuration Models ===")
    
    # Basic processing config
    config = VideoProcessingConfig(
        target_quality=VideoQuality.ULTRA,
        target_format=VideoFormat.MP4,
        target_resolution="1920x1080",
        target_fps=60.0,
        target_bitrate=8000,
        max_workers=16,
        use_gpu=True,
        optimize_for_web=True
    )
    
    print(f"Target quality: {config.target_quality}")
    print(f"Target format: {config.target_format}")
    print(f"Target resolution: {config.target_resolution}")
    print(f"Max workers: {config.max_workers}")
    print(f"Use GPU: {config.use_gpu}")
    
    # Viral processing config
    viral_config = ViralProcessingConfig(
        viral_optimization_enabled=True,
        use_langchain=True,
        langchain_model="gpt-4",
        langchain_temperature=0.8,
        min_viral_score=0.7,
        max_variants=15,
        variant_diversity=0.9,
        cache_enabled=True,
        parallel_generation=True
    )
    
    print(f"Viral optimization: {viral_config.viral_optimization_enabled}")
    print(f"LangChain model: {viral_config.langchain_model}")
    print(f"Min viral score: {viral_config.min_viral_score}")
    print(f"Max variants: {viral_config.max_variants}")
    print()

# =============================================================================
# API INTEGRATION EXAMPLES
# =============================================================================

async def example_api_integration():
    """Example: API integration with Pydantic models."""
    print("=== API Integration ===")
    
    # Create integration instance
    integration = VideoOpusClipPydanticIntegration()
    
    # Example request data
    request_data = {
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "language": "en",
        "max_clip_length": 60,
        "quality": "high",
        "format": "mp4"
    }
    
    # Process video clip request
    print("Processing video clip request...")
    result = await integration.process_request(request_data, "video_clip")
    
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Clip ID: {result.get('clip_id')}")
        print(f"Duration: {result.get('duration')}")
        print(f"Processing time: {result.get('processing_time')}")
    else:
        print(f"Error: {result.get('error')}")
    
    # Process viral video request
    viral_data = {
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "language": "en",
        "n_variants": 3,
        "use_langchain": True,
        "viral_optimization": True
    }
    
    print("\nProcessing viral video request...")
    viral_result = await integration.process_request(viral_data, "viral")
    
    print(f"Success: {viral_result.get('success')}")
    if viral_result.get('success'):
        print(f"Variants generated: {viral_result.get('total_variants_generated')}")
        print(f"Average viral score: {viral_result.get('average_viral_score')}")
        print(f"Best viral score: {viral_result.get('best_viral_score')}")
    else:
        print(f"Error: {viral_result.get('error')}")
    
    print()

# =============================================================================
# ERROR HANDLING EXAMPLES
# =============================================================================

def example_error_handling():
    """Example: Error handling with Pydantic models."""
    print("=== Error Handling ===")
    
    error_integrator = PydanticErrorIntegrator()
    
    # Example validation errors
    try:
        invalid_request = VideoClipRequest(
            youtube_url="invalid-url",
            language="invalid_lang",
            max_clip_length=-10
        )
    except Exception as e:
        print(f"Validation error caught: {e}")
        
        # Convert to ErrorInfo
        if hasattr(e, '__class__') and 'ValidationError' in str(e.__class__):
            error_info = error_integrator.convert_pydantic_error(e)
            print(f"Error code: {error_info.error_code}")
            print(f"Error message: {error_info.error_message}")
            print(f"Error type: {error_info.error_type}")
    
    # Create error info manually
    error_info = ErrorInfo(
        error_code="CUSTOM_ERROR",
        error_message="This is a custom error message",
        error_type="custom_error",
        field_name="youtube_url",
        field_value="invalid-url",
        request_id="req_123",
        user_id="user_456",
        additional_context={
            "source": "example",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    print(f"Custom error: {error_info.error_message}")
    print(f"Field: {error_info.field_name}")
    print(f"Context: {error_info.additional_context}")
    print()

# =============================================================================
# SERIALIZATION EXAMPLES
# =============================================================================

def example_serialization():
    """Example: Serialization with Pydantic models."""
    print("=== Serialization ===")
    
    serializer = PydanticSerializationIntegrator()
    
    # Create a response
    response = VideoClipResponse(
        success=True,
        clip_id="clip_123",
        title="Amazing Video",
        description="This is an amazing video",
        duration=45.5,
        file_size=10485760,
        processing_time=12.3,
        quality=VideoQuality.HIGH,
        format=VideoFormat.MP4,
        status=VideoStatus.COMPLETED
    )
    
    # Serialize for API
    api_data = serializer.serialize_for_api(response)
    print("API serialization:")
    print(json.dumps(api_data, indent=2))
    
    # Serialize for cache
    cache_data = serializer.serialize_for_cache(response)
    print(f"\nCache serialization (length: {len(cache_data)}):")
    print(cache_data[:100] + "..." if len(cache_data) > 100 else cache_data)
    
    # Deserialize from cache
    deserialized = serializer.deserialize_from_cache(cache_data, VideoClipResponse)
    if deserialized:
        print(f"\nDeserialized successfully: {deserialized.clip_id}")
    
    print()

# =============================================================================
# FACTORY FUNCTION EXAMPLES
# =============================================================================

def example_factory_functions():
    """Example: Using factory functions."""
    print("=== Factory Functions ===")
    
    # Create video clip request
    request = create_video_clip_request(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60,
        quality=VideoQuality.HIGH
    )
    print(f"Created request: {request.youtube_url}")
    
    # Create viral video request
    viral_request = create_viral_video_request(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        n_variants=5,
        use_langchain=True
    )
    print(f"Created viral request: {viral_request.n_variants} variants")
    
    # Create batch request
    requests = [
        create_video_clip_request("https://youtube.com/watch?v=1", "en", 30),
        create_video_clip_request("https://youtube.com/watch?v=2", "es", 45)
    ]
    batch_request = create_batch_request(requests, ProcessingPriority.HIGH)
    print(f"Created batch request: {batch_request.total_requests} requests")
    
    # Create processing config
    config = create_processing_config(
        quality=VideoQuality.ULTRA,
        format=VideoFormat.MP4,
        max_workers=16
    )
    print(f"Created config: {config.target_quality} quality, {config.max_workers} workers")
    print()

# =============================================================================
# ADVANCED USAGE EXAMPLES
# =============================================================================

def example_advanced_usage():
    """Example: Advanced Pydantic model usage."""
    print("=== Advanced Usage ===")
    
    # Create processing metrics
    metrics = ProcessingMetrics(
        start_time=datetime.now(),
        memory_usage_mb=512.5,
        cpu_usage_percent=75.2,
        gpu_usage_percent=85.0,
        throughput=100.5,
        efficiency=0.92,
        cache_hit_rate=0.78
    )
    
    print(f"Processing metrics:")
    print(f"  Memory usage: {metrics.memory_usage_mb} MB")
    print(f"  CPU usage: {metrics.cpu_usage_percent}%")
    print(f"  GPU usage: {metrics.gpu_usage_percent}%")
    print(f"  Throughput: {metrics.throughput}")
    print(f"  Efficiency: {metrics.efficiency}")
    print(f"  Cache hit rate: {metrics.cache_hit_rate}")
    
    # Create viral batch response
    from .pydantic_models import ViralVideoVariant
    
    variants = [
        ViralVideoVariant(
            variant_id="var_1",
            title="Variant 1",
            description="First variant",
            viral_score=0.85,
            engagement_prediction=0.78,
            retention_score=0.92,
            duration=45.0
        ),
        ViralVideoVariant(
            variant_id="var_2",
            title="Variant 2",
            description="Second variant",
            viral_score=0.92,
            engagement_prediction=0.85,
            retention_score=0.88,
            duration=52.0
        )
    ]
    
    viral_response = ViralVideoBatchResponse(
        success=True,
        original_clip_id="clip_123",
        variants=variants,
        total_variants_generated=2,
        successful_variants=2,
        processing_time=45.5,
        average_viral_score=0.885,
        best_viral_score=0.92
    )
    
    print(f"\nViral batch response:")
    print(f"  Success: {viral_response.success}")
    print(f"  Variants: {viral_response.total_variants_generated}")
    print(f"  Success rate: {viral_response.success_rate}")
    print(f"  Best variant: {viral_response.best_variant.variant_id if viral_response.best_variant else 'None'}")
    print(f"  High performing variants: {len(viral_response.high_performing_variants)}")
    print()

# =============================================================================
# PERFORMANCE EXAMPLES
# =============================================================================

def example_performance_optimization():
    """Example: Performance optimization with Pydantic."""
    print("=== Performance Optimization ===")
    
    import time
    
    # Benchmark model creation
    start_time = time.time()
    for i in range(1000):
        request = VideoClipRequest(
            youtube_url=f"https://www.youtube.com/watch?v=test{i}",
            language="en",
            max_clip_length=60
        )
    creation_time = time.time() - start_time
    print(f"Created 1000 models in {creation_time:.3f} seconds")
    
    # Benchmark validation
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60
    )
    
    start_time = time.time()
    for i in range(1000):
        validation_result = validate_video_request(request)
    validation_time = time.time() - start_time
    print(f"Validated 1000 requests in {validation_time:.3f} seconds")
    
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
    
    print()

# =============================================================================
# MAIN EXAMPLE RUNNER
# =============================================================================

def run_all_examples():
    """Run all Pydantic examples."""
    print("🚀 Video-OpusClip Pydantic Examples")
    print("=" * 50)
    
    # Basic examples
    example_basic_model_creation()
    example_model_validation()
    example_computed_fields()
    example_viral_video_models()
    
    # Validation examples
    example_validation_utilities()
    example_batch_validation()
    example_youtube_url_validation()
    
    # Configuration examples
    example_configuration_models()
    
    # Error handling examples
    example_error_handling()
    
    # Serialization examples
    example_serialization()
    
    # Factory function examples
    example_factory_functions()
    
    # Advanced usage examples
    example_advanced_usage()
    
    # Performance examples
    example_performance_optimization()
    
    print("✅ All examples completed successfully!")

async def run_async_examples():
    """Run async examples."""
    print("🔄 Running Async Examples")
    print("=" * 30)
    
    await example_api_integration()
    
    print("✅ Async examples completed successfully!")

if __name__ == "__main__":
    # Run synchronous examples
    run_all_examples()
    
    # Run asynchronous examples
    asyncio.run(run_async_examples()) 