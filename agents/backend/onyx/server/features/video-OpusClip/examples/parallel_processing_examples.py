"""
Parallel Processing Examples

Practical examples of using the advanced parallel processing system for video operations.
"""

import asyncio
import time
from typing import List, Dict
import structlog

from ..models.video_models import VideoClipRequest, VideoClip, VideoClipResponse
from ..models.viral_models import ViralVideoBatchResponse
from ..processors.video_processor import (
    VideoClipProcessor,
    create_high_performance_processor,
    create_video_processor
)
from ..processors.viral_processor import (
    ViralVideoProcessor,
    create_high_performance_viral_processor
)
from ..utils.parallel_utils import (
    HybridParallelProcessor,
    parallel_map,
    BackendType,
    ParallelConfig,
    setup_async_loop,
    estimate_processing_time
)

logger = structlog.get_logger()

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_video_requests(count: int = 100) -> List[VideoClipRequest]:
    """Generate sample video requests for testing."""
    base_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
        "https://www.youtube.com/watch?v=ZZ5LpwO-An4",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
    
    requests = []
    for i in range(count):
        url = base_urls[i % len(base_urls)]
        requests.append(VideoClipRequest(
            youtube_url=f"{url}&t={i}",
            language="en" if i % 2 == 0 else "es",
            max_clip_length=60 + (i % 30),
            min_clip_length=15 + (i % 10)
        ))
    
    return requests

# =============================================================================
# EXAMPLE 1: BASIC PARALLEL PROCESSING
# =============================================================================

def example_basic_parallel_processing():
    """Example of basic parallel processing with auto-backend selection."""
    print("=== Example 1: Basic Parallel Processing ===")
    
    # Generate sample data
    requests = generate_sample_video_requests(50)
    print(f"Processing {len(requests)} video requests...")
    
    # Create processor
    processor = create_high_performance_processor()
    
    # Process with auto-backend selection
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(requests)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Completed in {duration:.2f}s")
    print(f"📊 Items/second: {len(requests) / duration:.2f}")
    print(f"📈 Backend stats: {processor.get_processing_stats()}")
    print()

# =============================================================================
# EXAMPLE 2: BACKEND COMPARISON
# =============================================================================

def example_backend_comparison():
    """Compare performance of different parallel processing backends."""
    print("=== Example 2: Backend Comparison ===")
    
    requests = generate_sample_video_requests(20)
    processor = create_high_performance_processor()
    
    backends = [
        BackendType.THREAD,
        BackendType.PROCESS,
        BackendType.JOBLIB,
        BackendType.DASK,
        BackendType.ASYNC
    ]
    
    results = {}
    
    for backend in backends:
        try:
            print(f"Testing {backend.value} backend...")
            start_time = time.perf_counter()
            
            if backend == BackendType.ASYNC:
                results_async = asyncio.run(processor.process_batch_async(requests))
                duration = time.perf_counter() - start_time
            else:
                results_sync = processor.process_batch_parallel(requests, backend=backend)
                duration = time.perf_counter() - start_time
            
            results[backend.value] = {
                'duration': duration,
                'items_per_second': len(requests) / duration
            }
            
            print(f"  ✅ {backend.value}: {duration:.3f}s ({len(requests) / duration:.1f} items/s)")
            
        except Exception as e:
            print(f"  ❌ {backend.value}: Failed - {str(e)}")
            results[backend.value] = {'error': str(e)}
    
    # Find fastest backend
    fastest = min(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['duration']
    )
    print(f"\n🏆 Fastest backend: {fastest[0]} ({fastest[1]['items_per_second']:.1f} items/s)")
    print()

# =============================================================================
# EXAMPLE 3: VIRAL CONTENT PROCESSING
# =============================================================================

def example_viral_processing():
    """Example of viral content processing with parallel variants generation."""
    print("=== Example 3: Viral Content Processing ===")
    
    requests = generate_sample_video_requests(10)
    processor = create_high_performance_viral_processor()
    
    print(f"Generating viral variants for {len(requests)} videos...")
    
    start_time = time.perf_counter()
    viral_results = processor.process_batch_parallel(
        requests,
        n_variants=5,
        audience_profile={'age': '18-35', 'interests': ['tech', 'gaming']},
        experiment_id="viral_test_001"
    )
    duration = time.perf_counter() - start_time
    
    total_variants = sum(len(result.variants) for result in viral_results)
    avg_score = sum(
        sum(v.viral_score for v in result.variants) / len(result.variants)
        for result in viral_results if result.variants
    ) / len(viral_results)
    
    print(f"✅ Generated {total_variants} viral variants in {duration:.2f}s")
    print(f"📊 Average viral score: {avg_score:.3f}")
    print(f"🚀 Variants per second: {total_variants / duration:.1f}")
    print()

# =============================================================================
# EXAMPLE 4: BATCH VALIDATION
# =============================================================================

def example_batch_validation():
    """Example of parallel batch validation."""
    print("=== Example 4: Batch Validation ===")
    
    requests = generate_sample_video_requests(100)
    processor = create_high_performance_processor()
    
    print(f"Validating {len(requests)} video requests...")
    
    start_time = time.perf_counter()
    validation_results = processor.validate_batch(requests)
    duration = time.perf_counter() - start_time
    
    valid_count = sum(validation_results)
    invalid_count = len(validation_results) - valid_count
    
    print(f"✅ Validation completed in {duration:.3f}s")
    print(f"📊 Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"🚀 Validations per second: {len(requests) / duration:.1f}")
    print()

# =============================================================================
# EXAMPLE 5: SPECIALIZED PROCESSORS
# =============================================================================

def example_specialized_processors():
    """Example using specialized processors for specific operations."""
    print("=== Example 5: Specialized Processors ===")
    
    from ..processors.video_processor import VideoEncodingProcessor, VideoValidationProcessor
    from ..processors.viral_processor import ViralVariantProcessor, ViralAnalyticsProcessor
    
    # Generate sample data
    requests = generate_sample_video_requests(20)
    video_processor = create_high_performance_processor()
    viral_processor = create_high_performance_viral_processor()
    
    # Process videos
    print("Processing videos...")
    video_results = video_processor.process_batch_parallel(requests)
    
    # Process viral variants
    print("Generating viral variants...")
    viral_results = viral_processor.process_batch_parallel(requests, n_variants=3)
    
    # Use specialized processors
    encoding_processor = VideoEncodingProcessor()
    validation_processor = VideoValidationProcessor()
    variant_processor = ViralVariantProcessor()
    analytics_processor = ViralAnalyticsProcessor()
    
    # Batch encoding
    print("Batch encoding videos...")
    start_time = time.perf_counter()
    encoded_data = encoding_processor.batch_encode_videos(video_results)
    encoding_duration = time.perf_counter() - start_time
    
    # Batch validation
    print("Batch validating videos...")
    start_time = time.perf_counter()
    validation_results = validation_processor.batch_validate_videos(video_results)
    validation_duration = time.perf_counter() - start_time
    
    # Batch variant extraction
    print("Extracting viral variants...")
    start_time = time.perf_counter()
    all_variants = variant_processor.batch_generate_variants(viral_results)
    variant_duration = time.perf_counter() - start_time
    
    # Batch analytics
    print("Analyzing viral performance...")
    start_time = time.perf_counter()
    analytics_results = analytics_processor.batch_analyze_performance(viral_results)
    analytics_duration = time.perf_counter() - start_time
    
    print(f"✅ Encoding: {encoding_duration:.3f}s")
    print(f"✅ Validation: {validation_duration:.3f}s")
    print(f"✅ Variant extraction: {variant_duration:.3f}s")
    print(f"✅ Analytics: {analytics_duration:.3f}s")
    print(f"📊 Total variants extracted: {sum(len(v) for v in all_variants)}")
    print()

# =============================================================================
# EXAMPLE 6: PERFORMANCE MONITORING
# =============================================================================

def example_performance_monitoring():
    """Example of performance monitoring and optimization."""
    print("=== Example 6: Performance Monitoring ===")
    
    requests = generate_sample_video_requests(50)
    processor = create_high_performance_processor()
    
    # Estimate processing time
    estimated_time = estimate_processing_time(requests, processor.process, sample_size=5)
    print(f"⏱️ Estimated processing time: {estimated_time:.2f}s")
    
    # Monitor actual performance
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(requests)
    actual_time = time.perf_counter() - start_time
    
    print(f"⏱️ Actual processing time: {actual_time:.2f}s")
    print(f"📈 Estimation accuracy: {((estimated_time - actual_time) / actual_time * 100):.1f}%")
    
    # Get detailed stats
    stats = processor.get_processing_stats()
    print(f"📊 Processing statistics:")
    for backend, metrics in stats.items():
        print(f"  {backend}: {metrics['items_per_second']:.1f} items/s")
    print()

# =============================================================================
# EXAMPLE 7: ASYNC PROCESSING
# =============================================================================

async def example_async_processing():
    """Example of async processing for I/O-intensive operations."""
    print("=== Example 7: Async Processing ===")
    
    requests = generate_sample_video_requests(30)
    processor = create_high_performance_processor()
    
    print(f"Processing {len(requests)} requests asynchronously...")
    
    start_time = time.perf_counter()
    results = await processor.process_batch_async(requests)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Async processing completed in {duration:.2f}s")
    print(f"🚀 Items per second: {len(requests) / duration:.1f}")
    print()

# =============================================================================
# EXAMPLE 8: CUSTOM PARALLEL CONFIGURATION
# =============================================================================

def example_custom_configuration():
    """Example of custom parallel processing configuration."""
    print("=== Example 8: Custom Configuration ===")
    
    # Custom configuration for specific use case
    config = ParallelConfig(
        max_workers=8,
        chunk_size=500,
        timeout=30.0,
        backend="auto",
        use_uvloop=True,
        use_numba=True
    )
    
    processor = VideoClipProcessor(config)
    requests = generate_sample_video_requests(25)
    
    print(f"Processing with custom config: {config.max_workers} workers, chunk_size={config.chunk_size}")
    
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(requests)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Custom config completed in {duration:.2f}s")
    print(f"📊 Performance stats: {processor.get_processing_stats()}")
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all parallel processing examples."""
    print("🚀 Video Processing Parallel Examples")
    print("=" * 50)
    
    # Setup async loop
    setup_async_loop()
    
    try:
        # Run examples
        example_basic_parallel_processing()
        example_backend_comparison()
        example_viral_processing()
        example_batch_validation()
        example_specialized_processors()
        example_performance_monitoring()
        example_custom_configuration()
        
        # Run async example
        asyncio.run(example_async_processing())
        
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error("Example execution failed", error=str(e))
        print(f"❌ Example execution failed: {e}")

if __name__ == "__main__":
    run_all_examples() 