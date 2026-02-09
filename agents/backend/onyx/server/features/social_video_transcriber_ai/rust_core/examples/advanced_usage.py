"""
Advanced Usage Examples for Transcriber Core

This file demonstrates advanced usage patterns and optimizations.
"""

from transcriber_core import (
    TextProcessor, CacheService, CompressionService,
    BatchProcessor, Profiler, HealthChecker, Config
)

def example_profiling():
    """Example: Using the profiler for performance analysis"""
    profiler = Profiler()
    
    # Profile text processing
    processor = TextProcessor()
    profiler.start_timer("text_analysis")
    stats = processor.analyze("Sample text for analysis")
    profiler.record_time("text_analysis", 10.5)
    
    # Profile compression
    compressor = CompressionService()
    data = b"Sample data for compression" * 1000
    profiler.start_timer("compression")
    compressed = compressor.compress_lz4(data)
    profiler.record_time("compression", 5.2)
    
    # Get statistics
    stats = profiler.get_stats()
    print("Performance Stats:", stats)
    
    # Export report
    report = profiler.export_report()
    print("JSON Report:", report)

def example_health_monitoring():
    """Example: Health checking and monitoring"""
    health = HealthChecker()
    
    # Simulate requests
    for _ in range(100):
        health.record_request()
        if _ % 10 == 0:
            health.record_error()
    
    # Get health status
    status = health.get_health()
    print("Health Status:", status)
    
    # Get metrics
    metrics = health.get_metrics()
    print("Metrics:", metrics)

def example_configuration():
    """Example: Using configuration"""
    # Default config
    config = Config()
    print("Default cache size:", config.get_max_cache_size())
    
    # Custom config
    custom_config = Config.with_options(
        max_cache_size=50_000,
        num_workers=8,
        enable_simd=True,
        compression_level=6
    )
    print("Custom config:", custom_config.to_dict())

def example_batch_with_cache():
    """Example: Batch processing with caching"""
    cache = CacheService(max_size=1000, default_ttl=3600)
    processor = BatchProcessor(num_workers=4, batch_size=10)
    
    # Process items with caching
    items = [f"item_{i}" for i in range(100)]
    
    for item in items:
        if not cache.get(item):
            # Process item
            processed = item.upper()
            cache.set(item, processed, ttl=None)
        else:
            # Use cached value
            processed = cache.get(item)
    
    print("Cache stats:", cache.get_stats())

def example_optimized_pipeline():
    """Example: Optimized processing pipeline"""
    profiler = Profiler()
    config = Config.with_options(
        max_cache_size=10_000,
        num_workers=8,
        enable_simd=True
    )
    
    # Text processing
    processor = TextProcessor()
    text = "Sample transcription text from video"
    
    profiler.start_timer("pipeline")
    
    # Analyze
    stats = processor.analyze(text)
    profiler.increment("analyses")
    
    # Extract keywords
    keywords = processor.extract_keywords(text, top_n=10)
    profiler.increment("keyword_extractions")
    
    # Compress
    compressor = CompressionService()
    compressed = compressor.compress_lz4(text.encode())
    profiler.increment("compressions")
    
    profiler.record_time("pipeline", 15.0)
    
    # Get results
    pipeline_stats = profiler.get_stats()
    print("Pipeline Stats:", pipeline_stats)

if __name__ == "__main__":
    print("=== Profiling Example ===")
    example_profiling()
    
    print("\n=== Health Monitoring Example ===")
    example_health_monitoring()
    
    print("\n=== Configuration Example ===")
    example_configuration()
    
    print("\n=== Batch with Cache Example ===")
    example_batch_with_cache()
    
    print("\n=== Optimized Pipeline Example ===")
    example_optimized_pipeline()












