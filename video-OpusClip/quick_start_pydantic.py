#!/usr/bin/env python3
"""
Quick Start Guide for Pydantic in Video-OpusClip System

This script demonstrates the basic usage of Pydantic models for input/output
validation and response schemas in the Video-OpusClip system.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

def main():
    """Main function demonstrating Pydantic usage."""
    print("🚀 Video-OpusClip Pydantic Quick Start")
    print("=" * 50)
    
    # Import Pydantic models
    try:
        from pydantic_models import (
            VideoClipRequest,
            VideoClipResponse,
            ViralVideoRequest,
            ViralVideoBatchResponse,
            BatchVideoRequest,
            VideoQuality,
            VideoFormat,
            ProcessingPriority,
            ContentType,
            EngagementType,
            validate_video_request,
            create_video_clip_request,
            create_viral_video_request,
            YouTubeUrlValidator
        )
        print("✅ Pydantic models imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the video-OpusClip directory")
        return
    
    print("\n1. Basic Model Creation")
    print("-" * 30)
    
    # Create a basic video clip request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60,
        quality=VideoQuality.HIGH,
        format=VideoFormat.MP4
    )
    
    print(f"✅ Created request for: {request.youtube_url}")
    print(f"   Video ID: {request.video_id}")
    print(f"   Request Hash: {request.request_hash}")
    print(f"   Quality: {request.quality}")
    print(f"   Format: {request.format}")
    
    print("\n2. Model Validation")
    print("-" * 30)
    
    # Validate the request
    validation_result = validate_video_request(request)
    
    print(f"✅ Validation result:")
    print(f"   Valid: {validation_result.is_valid}")
    print(f"   Overall score: {validation_result.overall_score}")
    print(f"   Quality score: {validation_result.quality_score}")
    print(f"   Duration score: {validation_result.duration_score}")
    
    if validation_result.has_warnings:
        print(f"   Warnings: {validation_result.warnings}")
    
    print("\n3. Error Handling")
    print("-" * 30)
    
    # Demonstrate error handling
    try:
        invalid_request = VideoClipRequest(
            youtube_url="invalid-url",
            language="invalid_lang",
            max_clip_length=-10
        )
        print("❌ Should have failed validation")
    except Exception as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    print("\n4. YouTube URL Validation")
    print("-" * 30)
    
    # Test YouTube URL validation
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://invalid-url.com"
    ]
    
    for url in test_urls:
        try:
            validated_url = YouTubeUrlValidator.validate_youtube_url(url)
            video_id = YouTubeUrlValidator.extract_video_id(url)
            print(f"✅ Valid URL: {url}")
            print(f"   Video ID: {video_id}")
        except Exception as e:
            print(f"❌ Invalid URL: {url} - {e}")
    
    print("\n5. Factory Functions")
    print("-" * 30)
    
    # Use factory functions
    factory_request = create_video_clip_request(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60,
        quality=VideoQuality.HIGH
    )
    
    print(f"✅ Created with factory: {factory_request.youtube_url}")
    print(f"   Default quality: {factory_request.quality}")
    print(f"   Default format: {factory_request.format}")
    
    print("\n6. Viral Video Models")
    print("-" * 30)
    
    # Create viral video request
    viral_request = create_viral_video_request(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        n_variants=3,
        use_langchain=True
    )
    
    print(f"✅ Created viral request:")
    print(f"   Variants: {viral_request.n_variants}")
    print(f"   Use LangChain: {viral_request.use_langchain}")
    print(f"   Viral optimization: {viral_request.viral_optimization}")
    
    print("\n7. Response Models")
    print("-" * 30)
    
    # Create a response
    response = VideoClipResponse(
        success=True,
        clip_id="clip_123",
        title="Processed Video",
        description="Video processed successfully",
        duration=45.5,
        language="en",
        file_path="/path/to/video.mp4",
        file_size=10485760,  # 10MB
        resolution="1920x1080",
        fps=30.0,
        bitrate=5000,
        processing_time=12.3,
        quality=VideoQuality.HIGH,
        format=VideoFormat.MP4
    )
    
    print(f"✅ Created response:")
    print(f"   Success: {response.success}")
    print(f"   Clip ID: {response.clip_id}")
    print(f"   Duration: {response.duration}s")
    print(f"   File size: {response.file_size_mb} MB")
    print(f"   Processing time: {response.processing_time}s")
    print(f"   Is successful: {response.is_successful}")
    print(f"   Has warnings: {response.has_warnings}")
    
    print("\n8. Serialization")
    print("-" * 30)
    
    # Serialize to dict
    response_dict = response.model_dump()
    print(f"✅ Serialized to dict: {len(response_dict)} fields")
    
    # Serialize to JSON
    response_json = response.model_dump_json()
    print(f"✅ Serialized to JSON: {len(response_json)} characters")
    
    # Deserialize from dict
    recreated_response = VideoClipResponse.model_validate(response_dict)
    print(f"✅ Deserialized: {recreated_response.clip_id}")
    
    print("\n9. Batch Processing")
    print("-" * 30)
    
    # Create batch request
    requests = [
        create_video_clip_request("https://youtube.com/watch?v=1", "en", 30),
        create_video_clip_request("https://youtube.com/watch?v=2", "es", 45),
        create_video_clip_request("https://youtube.com/watch?v=3", "fr", 60)
    ]
    
    batch_request = BatchVideoRequest(
        requests=requests,
        priority=ProcessingPriority.HIGH,
        max_workers=4
    )
    
    print(f"✅ Created batch request:")
    print(f"   Total requests: {batch_request.total_requests}")
    print(f"   Priority: {batch_request.priority}")
    print(f"   Max workers: {batch_request.max_workers}")
    print(f"   Estimated time: {batch_request.estimated_processing_time:.1f}s")
    
    print("\n10. Configuration Models")
    print("-" * 30)
    
    from pydantic_models import VideoProcessingConfig, ViralProcessingConfig
    
    # Create processing config
    config = VideoProcessingConfig(
        target_quality=VideoQuality.ULTRA,
        target_format=VideoFormat.MP4,
        target_resolution="1920x1080",
        target_fps=60.0,
        max_workers=16,
        use_gpu=True,
        optimize_for_web=True
    )
    
    print(f"✅ Created processing config:")
    print(f"   Quality: {config.target_quality}")
    print(f"   Format: {config.target_format}")
    print(f"   Resolution: {config.target_resolution}")
    print(f"   FPS: {config.target_fps}")
    print(f"   Workers: {config.max_workers}")
    print(f"   Use GPU: {config.use_gpu}")
    
    # Create viral config
    viral_config = ViralProcessingConfig(
        viral_optimization_enabled=True,
        use_langchain=True,
        langchain_model="gpt-4",
        min_viral_score=0.7,
        max_variants=15,
        cache_enabled=True
    )
    
    print(f"✅ Created viral config:")
    print(f"   Viral optimization: {viral_config.viral_optimization_enabled}")
    print(f"   LangChain model: {viral_config.langchain_model}")
    print(f"   Min viral score: {viral_config.min_viral_score}")
    print(f"   Max variants: {viral_config.max_variants}")
    
    print("\n11. Integration Example")
    print("-" * 30)
    
    try:
        from pydantic_integration import VideoOpusClipPydanticIntegration
        
        integration = VideoOpusClipPydanticIntegration()
        
        # Example request data
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en",
            "max_clip_length": 60,
            "quality": "high"
        }
        
        print("✅ Integration created successfully")
        print("   (Async processing would happen here)")
        
    except ImportError as e:
        print(f"⚠️  Integration not available: {e}")
    
    print("\n12. Performance Test")
    print("-" * 30)
    
    import time
    
    # Benchmark model creation
    start_time = time.time()
    for i in range(100):
        request = VideoClipRequest(
            youtube_url=f"https://youtube.com/watch?v=test{i}",
            language="en",
            max_clip_length=60
        )
    creation_time = time.time() - start_time
    
    print(f"✅ Created 100 models in {creation_time:.3f} seconds")
    print(f"   Average: {creation_time/100*1000:.2f} ms per model")
    
    # Benchmark validation
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language="en",
        max_clip_length=60
    )
    
    start_time = time.time()
    for i in range(100):
        validation_result = validate_video_request(request)
    validation_time = time.time() - start_time
    
    print(f"✅ Validated 100 requests in {validation_time:.3f} seconds")
    print(f"   Average: {validation_time/100*1000:.2f} ms per validation")
    
    # Benchmark serialization
    response = VideoClipResponse(
        success=True,
        clip_id="clip_123",
        duration=45.5,
        processing_time=12.3
    )
    
    start_time = time.time()
    for i in range(100):
        data = response.model_dump()
    serialization_time = time.time() - start_time
    
    print(f"✅ Serialized 100 responses in {serialization_time:.3f} seconds")
    print(f"   Average: {serialization_time/100*1000:.2f} ms per serialization")
    
    print("\n🎉 Pydantic Quick Start Completed Successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Read the PYDANTIC_GUIDE.md for detailed documentation")
    print("2. Check pydantic_examples.py for comprehensive examples")
    print("3. Integrate Pydantic models into your API endpoints")
    print("4. Use validation utilities for robust error handling")
    print("5. Leverage factory functions for common use cases")

async def async_example():
    """Demonstrate async Pydantic integration."""
    print("\n🔄 Async Integration Example")
    print("-" * 30)
    
    try:
        from pydantic_integration import VideoOpusClipPydanticIntegration
        
        integration = VideoOpusClipPydanticIntegration()
        
        # Example request data
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en",
            "max_clip_length": 60
        }
        
        print("Processing video clip request...")
        result = await integration.process_request(request_data, "video_clip")
        
        print(f"✅ Async processing result:")
        print(f"   Success: {result.get('success')}")
        if result.get('success'):
            print(f"   Clip ID: {result.get('clip_id')}")
            print(f"   Duration: {result.get('duration')}")
            print(f"   Processing time: {result.get('processing_time')}")
        else:
            print(f"   Error: {result.get('error')}")
        
        # Viral video example
        viral_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "n_variants": 3,
            "use_langchain": True
        }
        
        print("\nProcessing viral video request...")
        viral_result = await integration.process_request(viral_data, "viral")
        
        print(f"✅ Viral processing result:")
        print(f"   Success: {viral_result.get('success')}")
        if viral_result.get('success'):
            print(f"   Variants: {viral_result.get('total_variants_generated')}")
            print(f"   Average score: {viral_result.get('average_viral_score')}")
        else:
            print(f"   Error: {viral_result.get('error')}")
            
    except ImportError as e:
        print(f"⚠️  Async integration not available: {e}")
    except Exception as e:
        print(f"❌ Async example failed: {e}")

if __name__ == "__main__":
    # Run main examples
    main()
    
    # Run async examples
    asyncio.run(async_example()) 