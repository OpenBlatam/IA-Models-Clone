"""
Serialization Examples

Practical examples of using the enhanced serialization system with high-performance libraries.
"""

import time
import asyncio
from typing import List, Dict, Any
import structlog

from ..models.video_models import (
    VideoClipRequest,
    VideoClipResponse,
    VideoBatchRequest,
    VideoQuality,
    VideoFormat,
    video_serializer,
    video_batch_serializer,
    video_benchmark,
    create_video_request,
    create_video_batch
)
from ..models.viral_models import (
    ViralVideoVariant,
    ViralVideoBatchResponse,
    CaptionSegment,
    ScreenDivision,
    Transition,
    VideoEffect,
    TransitionType,
    ScreenDivisionType,
    CaptionStyle,
    VideoEffect as VideoEffectEnum,
    serializer,
    batch_serializer,
    benchmark,
    create_default_caption_config,
    create_split_screen_layout,
    create_viral_transition
)

logger = structlog.get_logger()

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_video_requests(count: int = 10) -> List[VideoClipRequest]:
    """Generate sample video requests."""
    return [
        VideoClipRequest(
            youtube_url=f"https://youtube.com/watch?v=video_{i}",
            language="en" if i % 2 == 0 else "es",
            max_clip_length=60 + (i % 30),
            quality=VideoQuality.HIGH,
            format=VideoFormat.MP4
        )
        for i in range(count)
    ]

def generate_viral_variants(count: int = 5) -> List[ViralVideoVariant]:
    """Generate sample viral variants."""
    variants = []
    for i in range(count):
        # Create captions
        captions = [
            CaptionSegment(
                text=f"🔥 Viral Caption {i+1}! 🔥",
                start_time=1.0,
                end_time=4.0,
                font_size=28,
                styles=[CaptionStyle.BOLD, CaptionStyle.SHADOW],
                animation="fade_in"
            )
        ]
        
        # Create screen division
        screen_division = create_split_screen_layout(
            ScreenDivisionType.SPLIT_HORIZONTAL if i % 2 == 0 else ScreenDivisionType.SPLIT_VERTICAL
        )
        
        # Create transitions
        transitions = [
            create_viral_transition(TransitionType.FADE),
            create_viral_transition(TransitionType.SLIDE)
        ]
        
        # Create effects
        effects = [
            VideoEffect(
                effect_type=VideoEffectEnum.NEON,
                intensity=0.7,
                duration=3.0
            )
        ]
        
        variant = ViralVideoVariant(
            variant_id=f"viral_variant_{i}",
            title=f"Viral Video {i+1}",
            description=f"Amazing viral content {i+1}",
            viral_score=0.7 + (i * 0.1),
            engagement_prediction=0.6 + (i * 0.1),
            captions=captions,
            screen_division=screen_division,
            transitions=transitions,
            effects=effects,
            total_duration=60.0,
            estimated_views=10000 + (i * 5000),
            estimated_likes=5000 + (i * 2500),
            estimated_shares=2000 + (i * 1000),
            estimated_comments=500 + (i * 250),
            tags=["viral", "trending", "amazing"],
            hashtags=["#viral", "#trending", "#amazing"],
            target_audience=["young_adults", "social_media"]
        )
        variants.append(variant)
    
    return variants

# =============================================================================
# EXAMPLE 1: BASIC SERIALIZATION
# =============================================================================

def example_basic_serialization():
    """Example of basic serialization with different formats."""
    print("=== Example 1: Basic Serialization ===")
    
    # Create sample data
    video_request = create_video_request(
        "https://youtube.com/watch?v=example",
        language="en",
        max_clip_length=60
    )
    
    viral_variant = generate_viral_variants(1)[0]
    
    print(f"Serializing video request and viral variant...")
    
    # Test different serialization formats
    formats = {
        "MsgPack": lambda obj: (serializer.to_msgpack(obj), serializer.from_msgpack),
        "JSON": lambda obj: (serializer.to_json(obj), serializer.from_json),
        "Dict": lambda obj: (serializer.to_dict(obj), serializer.from_dict)
    }
    
    for format_name, (serialize_func, deserialize_func) in formats.items():
        try:
            # Serialize
            start_time = time.perf_counter()
            serialized_data = serialize_func(video_request)
            serialize_time = time.perf_counter() - start_time
            
            # Deserialize
            start_time = time.perf_counter()
            if format_name == "JSON":
                deserialized_obj = deserialize_func(serialized_data, VideoClipRequest)
            else:
                deserialized_obj = deserialize_func(serialized_data, VideoClipRequest)
            deserialize_time = time.perf_counter() - start_time
            
            # Verify
            is_valid = (
                deserialized_obj.youtube_url == video_request.youtube_url and
                deserialized_obj.language == video_request.language and
                deserialized_obj.max_clip_length == video_request.max_clip_length
            )
            
            print(f"  {format_name}:")
            print(f"    Serialize: {serialize_time:.6f}s")
            print(f"    Deserialize: {deserialize_time:.6f}s")
            print(f"    Valid: {is_valid}")
            print(f"    Data size: {len(str(serialized_data))} chars")
            
        except Exception as e:
            print(f"  {format_name}: Failed - {str(e)}")
    
    print()

# =============================================================================
# EXAMPLE 2: BATCH SERIALIZATION
# =============================================================================

def example_batch_serialization():
    """Example of batch serialization for large datasets."""
    print("=== Example 2: Batch Serialization ===")
    
    # Generate large datasets
    video_requests = generate_video_requests(100)
    viral_variants = generate_viral_variants(50)
    
    print(f"Serializing {len(video_requests)} video requests and {len(viral_variants)} viral variants...")
    
    # Test batch serialization
    batch_formats = {
        "MsgPack": lambda objs: (batch_serializer.batch_to_msgpack(objs), batch_serializer.batch_from_msgpack),
        "JSON": lambda objs: (batch_serializer.batch_to_json(objs), batch_serializer.batch_from_json)
    }
    
    for format_name, (serialize_func, deserialize_func) in batch_formats.items():
        try:
            # Video requests batch
            start_time = time.perf_counter()
            serialized_videos = serialize_func(video_requests)
            video_serialize_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            deserialized_videos = deserialize_func(serialized_videos, VideoClipRequest)
            video_deserialize_time = time.perf_counter() - start_time
            
            # Viral variants batch
            start_time = time.perf_counter()
            serialized_variants = serialize_func(viral_variants)
            variant_serialize_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            deserialized_variants = deserialize_func(serialized_variants, ViralVideoVariant)
            variant_deserialize_time = time.perf_counter() - start_time
            
            # Verify
            videos_valid = len(deserialized_videos) == len(video_requests)
            variants_valid = len(deserialized_variants) == len(viral_variants)
            
            print(f"  {format_name}:")
            print(f"    Videos - Serialize: {video_serialize_time:.6f}s, Deserialize: {video_deserialize_time:.6f}s")
            print(f"    Variants - Serialize: {variant_serialize_time:.6f}s, Deserialize: {variant_deserialize_time:.6f}s")
            print(f"    Videos valid: {videos_valid}, Variants valid: {variants_valid}")
            print(f"    Video data size: {len(serialized_videos)} bytes")
            print(f"    Variant data size: {len(serialized_variants)} bytes")
            
        except Exception as e:
            print(f"  {format_name}: Failed - {str(e)}")
    
    print()

# =============================================================================
# EXAMPLE 3: PERFORMANCE BENCHMARKING
# =============================================================================

def example_performance_benchmarking():
    """Example of performance benchmarking for serialization."""
    print("=== Example 3: Performance Benchmarking ===")
    
    # Create test objects
    video_request = create_video_request("https://youtube.com/watch?v=benchmark")
    viral_variant = generate_viral_variants(1)[0]
    video_requests = generate_video_requests(50)
    viral_variants = generate_viral_variants(25)
    
    print("Benchmarking individual object serialization...")
    
    # Benchmark individual objects
    objects_to_test = [
        ("Video Request", video_request),
        ("Viral Variant", viral_variant)
    ]
    
    for obj_name, obj in objects_to_test:
        print(f"  {obj_name}:")
        results = benchmark.benchmark_serialization(obj, iterations=1000)
        
        for format_name, time_taken in results.items():
            ops_per_sec = 1000 / time_taken
            print(f"    {format_name}: {time_taken:.6f}s ({ops_per_sec:.0f} ops/sec)")
    
    print("\nBenchmarking batch serialization...")
    
    # Benchmark batch objects
    batches_to_test = [
        ("Video Requests", video_requests),
        ("Viral Variants", viral_variants)
    ]
    
    for batch_name, batch_objs in batches_to_test:
        print(f"  {batch_name}:")
        results = benchmark.benchmark_batch_serialization(batch_objs, iterations=100)
        
        for format_name, time_taken in results.items():
            ops_per_sec = 100 / time_taken
            print(f"    {format_name}: {time_taken:.6f}s ({ops_per_sec:.0f} ops/sec)")
    
    print()

# =============================================================================
# EXAMPLE 4: CUSTOM SERIALIZATION
# =============================================================================

def example_custom_serialization():
    """Example of custom serialization with specific requirements."""
    print("=== Example 4: Custom Serialization ===")
    
    # Create complex viral variant
    viral_variant = generate_viral_variants(1)[0]
    
    print("Testing custom serialization scenarios...")
    
    # 1. Serialize to different formats
    print("  1. Format conversion:")
    
    # Start with MsgPack (fastest)
    msgpack_data = serializer.to_msgpack(viral_variant)
    print(f"    MsgPack size: {len(msgpack_data)} bytes")
    
    # Convert to JSON for API
    json_data = serializer.to_json(viral_variant)
    print(f"    JSON size: {len(json_data)} chars")
    
    # Convert to dict for processing
    dict_data = serializer.to_dict(viral_variant)
    print(f"    Dict keys: {list(dict_data.keys())}")
    
    # 2. Partial serialization
    print("  2. Partial serialization:")
    
    # Serialize only captions
    captions_data = serializer.to_msgpack(viral_variant.captions)
    print(f"    Captions only: {len(captions_data)} bytes")
    
    # Serialize only metadata
    metadata = {
        "variant_id": viral_variant.variant_id,
        "viral_score": viral_variant.viral_score,
        "estimated_views": viral_variant.estimated_views
    }
    metadata_data = serializer.to_msgpack(metadata)
    print(f"    Metadata only: {len(metadata_data)} bytes")
    
    # 3. Validation with pydantic
    print("  3. Pydantic validation:")
    
    is_valid = serializer.validate_with_pydantic(viral_variant, ViralVideoVariant)
    print(f"    Variant valid: {is_valid}")
    
    # Test with invalid data
    invalid_data = {"invalid": "data"}
    is_valid_invalid = serializer.validate_with_pydantic(invalid_data, ViralVideoVariant)
    print(f"    Invalid data valid: {is_valid_invalid}")
    
    print()

# =============================================================================
# EXAMPLE 5: STREAMING SERIALIZATION
# =============================================================================

def example_streaming_serialization():
    """Example of streaming serialization for large datasets."""
    print("=== Example 5: Streaming Serialization ===")
    
    # Generate large dataset
    viral_variants = generate_viral_variants(1000)
    
    print(f"Streaming serialization of {len(viral_variants)} viral variants...")
    
    # Test streaming serialization
    formats = ["msgpack", "json"]
    
    for format_name in formats:
        try:
            start_time = time.perf_counter()
            stream_data = batch_serializer.stream_serialize(viral_variants, format=format_name)
            stream_time = time.perf_counter() - start_time
            
            print(f"  {format_name.upper()}:")
            print(f"    Time: {stream_time:.6f}s")
            print(f"    Data size: {len(stream_data)} bytes")
            print(f"    Throughput: {len(stream_data) / stream_time / 1024 / 1024:.2f} MB/s")
            
        except Exception as e:
            print(f"  {format_name.upper()}: Failed - {str(e)}")
    
    print()

# =============================================================================
# EXAMPLE 6: ERROR HANDLING
# =============================================================================

def example_error_handling():
    """Example of error handling in serialization."""
    print("=== Example 6: Error Handling ===")
    
    print("Testing error handling scenarios...")
    
    # 1. Invalid data type
    print("  1. Invalid data type:")
    try:
        invalid_obj = {"invalid": "object", "with": "non_serializable": lambda x: x}
        serializer.to_msgpack(invalid_obj)
    except Exception as e:
        print(f"    Caught error: {type(e).__name__}")
    
    # 2. Corrupted data
    print("  2. Corrupted data:")
    try:
        corrupted_data = b"invalid msgpack data"
        serializer.from_msgpack(corrupted_data, ViralVideoVariant)
    except Exception as e:
        print(f"    Caught error: {type(e).__name__}")
    
    # 3. Type mismatch
    print("  3. Type mismatch:")
    try:
        video_request = create_video_request("https://youtube.com/watch?v=test")
        json_data = serializer.to_json(video_request)
        # Try to deserialize as wrong type
        serializer.from_json(json_data, ViralVideoVariant)
    except Exception as e:
        print(f"    Caught error: {type(e).__name__}")
    
    # 4. Large object handling
    print("  4. Large object handling:")
    try:
        # Create very large object
        large_variants = generate_viral_variants(10000)
        start_time = time.perf_counter()
        large_data = batch_serializer.batch_to_msgpack(large_variants)
        large_time = time.perf_counter() - start_time
        print(f"    Large object serialized: {len(large_data)} bytes in {large_time:.3f}s")
    except Exception as e:
        print(f"    Large object failed: {type(e).__name__}")
    
    print()

# =============================================================================
# EXAMPLE 7: COMPARISON WITH STANDARD LIBRARIES
# =============================================================================

def example_library_comparison():
    """Example comparing with standard Python libraries."""
    print("=== Example 7: Library Comparison ===")
    
    import json
    import pickle
    import marshal
    
    # Create test object
    viral_variant = generate_viral_variants(1)[0]
    
    print("Comparing with standard Python libraries...")
    
    # Test different libraries
    libraries = {
        "msgspec": lambda obj: (serializer.to_msgpack(obj), lambda data: serializer.from_msgpack(data, type(obj))),
        "orjson": lambda obj: (serializer.to_json(obj), lambda data: serializer.from_json(data, type(obj))),
        "json": lambda obj: (json.dumps(serializer.to_dict(obj)), lambda data: serializer.from_dict(json.loads(data), type(obj))),
        "pickle": lambda obj: (pickle.dumps(obj), lambda data: pickle.loads(data)),
        "marshal": lambda obj: (marshal.dumps(serializer.to_dict(obj)), lambda data: serializer.from_dict(marshal.loads(data), type(obj)))
    }
    
    for lib_name, (serialize_func, deserialize_func) in libraries.items():
        try:
            # Serialize
            start_time = time.perf_counter()
            serialized_data = serialize_func(viral_variant)
            serialize_time = time.perf_counter() - start_time
            
            # Deserialize
            start_time = time.perf_counter()
            deserialized_obj = deserialize_func(serialized_data)
            deserialize_time = time.perf_counter() - start_time
            
            # Verify
            is_valid = deserialized_obj.variant_id == viral_variant.variant_id
            
            print(f"  {lib_name}:")
            print(f"    Serialize: {serialize_time:.6f}s")
            print(f"    Deserialize: {deserialize_time:.6f}s")
            print(f"    Valid: {is_valid}")
            print(f"    Data size: {len(serialized_data)} bytes")
            
        except Exception as e:
            print(f"  {lib_name}: Failed - {str(e)}")
    
    print()

# =============================================================================
# EXAMPLE 8: REAL-WORLD USAGE SCENARIOS
# =============================================================================

def example_real_world_usage():
    """Example of real-world serialization usage scenarios."""
    print("=== Example 8: Real-World Usage Scenarios ===")
    
    # Scenario 1: API Response
    print("Scenario 1: API Response")
    viral_variant = generate_viral_variants(1)[0]
    
    # Serialize for API response
    api_response = {
        "success": True,
        "data": serializer.to_dict(viral_variant),
        "timestamp": datetime.now().isoformat()
    }
    
    api_json = serializer.to_json(api_response)
    print(f"  API response size: {len(api_json)} chars")
    
    # Scenario 2: Database Storage
    print("Scenario 2: Database Storage")
    
    # Serialize for database (binary format)
    db_data = serializer.to_msgpack(viral_variant)
    print(f"  Database storage size: {len(db_data)} bytes")
    
    # Scenario 3: Cache Storage
    print("Scenario 3: Cache Storage")
    
    # Serialize for cache (fast access)
    cache_data = serializer.to_msgpack(viral_variant)
    cache_key = f"viral_variant:{viral_variant.variant_id}"
    print(f"  Cache key: {cache_key}")
    print(f"  Cache data size: {len(cache_data)} bytes")
    
    # Scenario 4: Batch Processing
    print("Scenario 4: Batch Processing")
    
    batch_variants = generate_viral_variants(100)
    
    # Serialize batch for processing
    batch_data = batch_serializer.batch_to_msgpack(batch_variants)
    print(f"  Batch size: {len(batch_data)} bytes")
    print(f"  Variants per MB: {len(batch_variants) / (len(batch_data) / 1024 / 1024):.1f}")
    
    # Scenario 5: File Storage
    print("Scenario 5: File Storage")
    
    # Serialize for file storage
    file_data = serializer.to_json(viral_variant)
    print(f"  File size: {len(file_data)} chars")
    print(f"  Human readable: Yes")
    
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_serialization_examples():
    """Run all serialization examples."""
    print("🚀 High-Performance Serialization Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_serialization()
        example_batch_serialization()
        example_performance_benchmarking()
        example_custom_serialization()
        example_streaming_serialization()
        example_error_handling()
        example_library_comparison()
        example_real_world_usage()
        
        print("🎉 All serialization examples completed successfully!")
        
    except Exception as e:
        logger.error("Serialization example execution failed", error=str(e))
        print(f"❌ Serialization example execution failed: {e}")

if __name__ == "__main__":
    run_all_serialization_examples() 