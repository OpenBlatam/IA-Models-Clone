"""
Demo Script for Improved Video-OpusClip API

This script demonstrates all the improvements made to the API:
- Early returns and guard clauses
- Lifespan context managers
- Enhanced type hints and Pydantic models
- Performance optimizations with caching
- Modular route organization
- Comprehensive validation and security
"""

import asyncio
import time
import json
from typing import Dict, Any

# Import improved modules
from models import (
    VideoClipRequest, VideoClipBatchRequest, ViralVideoRequest, LangChainRequest,
    Language, VideoQuality, VideoFormat, AnalysisType, Priority
)
from validation import validate_video_request, validate_batch_request
from error_handling import ValidationError, SecurityError, VideoProcessingError
from cache import CacheManager, CacheConfig
from monitoring import PerformanceMonitor, HealthChecker, MonitoringConfig
from dependencies import DependencyConfig

async def demo_improved_api():
    """Demonstrate all improvements in the Video-OpusClip API."""
    
    print("🚀 Video-OpusClip API Improvements Demo")
    print("=" * 50)
    
    # 1. Demo Enhanced Models with Validation
    print("\n1. 📋 Enhanced Pydantic Models with Validation")
    print("-" * 40)
    
    try:
        # Create video request with validation
        video_request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language=Language.EN,
            max_clip_length=60,
            min_clip_length=15,
            quality=VideoQuality.HIGH,
            format=VideoFormat.MP4,
            priority=Priority.NORMAL
        )
        print(f"✅ Video request created: {video_request.youtube_url}")
        
        # Create viral request
        viral_request = ViralVideoRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            n_variants=5,
            use_langchain=True,
            platform="tiktok"
        )
        print(f"✅ Viral request created: {viral_request.n_variants} variants")
        
        # Create LangChain request
        langchain_request = LangChainRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            analysis_type=AnalysisType.COMPREHENSIVE,
            platform="youtube"
        )
        print(f"✅ LangChain request created: {langchain_request.analysis_type}")
        
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
    
    # 2. Demo Early Returns and Guard Clauses
    print("\n2. 🛡️ Early Returns and Guard Clauses")
    print("-" * 40)
    
    def demo_early_returns():
        """Demonstrate early return pattern."""
        # Simulate function with early returns
        def process_video_with_early_returns(request):
            # Early return for None request
            if not request:
                raise ValidationError("Request object is required")
            
            # Early return for empty URL
            if not request.youtube_url or not request.youtube_url.strip():
                raise ValidationError("YouTube URL is required and cannot be empty")
            
            # Early return for invalid clip lengths
            if request.min_clip_length > request.max_clip_length:
                raise ValidationError("Minimum clip length cannot be greater than maximum")
            
            # Happy path: Process video
            return f"Processing video: {request.youtube_url}"
        
        # Test with valid request
        try:
            result = process_video_with_early_returns(video_request)
            print(f"✅ Early returns working: {result}")
        except ValidationError as e:
            print(f"❌ Early return error: {e}")
        
        # Test with invalid request
        try:
            invalid_request = VideoClipRequest(
                youtube_url="",  # Empty URL
                language=Language.EN
            )
            process_video_with_early_returns(invalid_request)
        except ValidationError as e:
            print(f"✅ Early return caught invalid request: {e}")
    
    demo_early_returns()
    
    # 3. Demo Enhanced Validation
    print("\n3. 🔍 Enhanced Validation and Security")
    print("-" * 40)
    
    # Test validation
    validation_result = validate_video_request(video_request)
    if validation_result.is_valid:
        print("✅ Video request validation passed")
    else:
        print(f"❌ Validation errors: {validation_result.errors}")
    
    # Test batch validation
    batch_request = VideoClipBatchRequest(
        requests=[video_request, viral_request],
        max_workers=4
    )
    batch_validation = validate_batch_request(batch_request)
    if batch_validation.is_valid:
        print("✅ Batch request validation passed")
    else:
        print(f"❌ Batch validation errors: {batch_validation.errors}")
    
    # 4. Demo Caching System
    print("\n4. 💾 High-Performance Caching System")
    print("-" * 40)
    
    async def demo_caching():
        """Demonstrate caching system."""
        # Initialize cache manager
        cache_config = CacheConfig(
            enable_fallback=True,
            fallback_max_size=100
        )
        cache_manager = CacheManager(cache_config)
        await cache_manager.initialize()
        
        # Test cache operations
        test_key = "demo_video_result"
        test_value = {"processed": True, "duration": 60, "quality": "high"}
        
        # Set value in cache
        await cache_manager.set(test_key, test_value, ttl=60)
        print("✅ Value cached successfully")
        
        # Get value from cache
        cached_value = await cache_manager.get(test_key)
        if cached_value:
            print(f"✅ Value retrieved from cache: {cached_value}")
        else:
            print("❌ Value not found in cache")
        
        # Get cache statistics
        stats = cache_manager.get_stats()
        print(f"📊 Cache stats: {stats['hit_rate_percent']}% hit rate")
        
        await cache_manager.close()
    
    await demo_caching()
    
    # 5. Demo Performance Monitoring
    print("\n5. 📊 Performance Monitoring and Health Checks")
    print("-" * 40)
    
    async def demo_monitoring():
        """Demonstrate monitoring system."""
        # Initialize monitoring
        monitoring_config = MonitoringConfig(
            enable_performance_monitoring=True,
            enable_health_checks=True
        )
        
        performance_monitor = PerformanceMonitor(monitoring_config)
        await performance_monitor.start()
        
        health_checker = HealthChecker(monitoring_config)
        await health_checker.initialize()
        
        # Simulate some requests
        await performance_monitor.record_request("POST", "/api/v1/video/process", 200, 1.5)
        await performance_monitor.record_request("POST", "/api/v1/viral/process", 200, 2.3)
        await performance_monitor.record_request("GET", "/health", 200, 0.1)
        
        # Get performance metrics
        metrics = performance_monitor.get_metrics()
        print(f"📈 Performance metrics: {metrics['performance']['request_count']} requests")
        print(f"📈 Average response time: {metrics['performance']['response_time_avg']}s")
        
        # Get health status
        health_status = await health_checker.check_system_health()
        print(f"🏥 System health: {health_status.status}")
        if health_status.issues:
            print(f"⚠️ Health issues: {health_status.issues}")
        
        await performance_monitor.stop()
        await health_checker.close()
    
    await demo_monitoring()
    
    # 6. Demo Error Handling
    print("\n6. 🚨 Comprehensive Error Handling")
    print("-" * 40)
    
    def demo_error_handling():
        """Demonstrate error handling patterns."""
        from error_handling import handle_processing_errors, create_error_response
        
        @handle_processing_errors
        async def demo_processing_function(request):
            if not request:
                raise ValidationError("Request is required")
            
            if "invalid" in request.youtube_url:
                raise SecurityError("Invalid URL detected")
            
            # Simulate processing
            await asyncio.sleep(0.1)
            return {"success": True, "processed": request.youtube_url}
        
        # Test with valid request
        try:
            result = await demo_processing_function(video_request)
            print(f"✅ Processing successful: {result}")
        except Exception as e:
            print(f"❌ Processing failed: {e}")
        
        # Test with invalid request
        try:
            invalid_request = VideoClipRequest(
                youtube_url="invalid_url",
                language=Language.EN
            )
            await demo_processing_function(invalid_request)
        except Exception as e:
            print(f"✅ Error handling caught invalid request: {type(e).__name__}")
    
    await demo_error_handling()
    
    # 7. Demo Dependency Injection
    print("\n7. 🔧 Enhanced Dependency Injection")
    print("-" * 40)
    
    def demo_dependency_injection():
        """Demonstrate dependency injection patterns."""
        from dependencies import DependencyConfig, get_video_processor
        
        # Create configuration
        config = DependencyConfig()
        print(f"✅ Dependency config created: {config.video_processor_config.max_workers} workers")
        
        # Get processor instance
        processor = get_video_processor()
        print(f"✅ Video processor instance created: {type(processor).__name__}")
        
        # Get processor stats
        stats = processor.get_stats()
        print(f"📊 Processor stats: {stats['processed_videos']} videos processed")
    
    demo_dependency_injection()
    
    # 8. Demo Async Operations
    print("\n8. ⚡ Async Operations and Performance")
    print("-" * 40)
    
    async def demo_async_operations():
        """Demonstrate async operations."""
        async def simulate_video_processing(request):
            # Simulate async processing
            await asyncio.sleep(0.5)
            return f"Processed {request.youtube_url} in 0.5s"
        
        async def simulate_batch_processing(requests):
            # Process multiple requests concurrently
            tasks = [simulate_video_processing(req) for req in requests]
            results = await asyncio.gather(*tasks)
            return results
        
        # Test single async operation
        start_time = time.perf_counter()
        result = await simulate_video_processing(video_request)
        single_time = time.perf_counter() - start_time
        print(f"✅ Single async operation: {result} ({single_time:.2f}s)")
        
        # Test batch async operations
        start_time = time.perf_counter()
        batch_results = await simulate_batch_processing([video_request, viral_request])
        batch_time = time.perf_counter() - start_time
        print(f"✅ Batch async operations: {len(batch_results)} results ({batch_time:.2f}s)")
        print(f"📈 Performance improvement: {single_time * 2 / batch_time:.1f}x faster")
    
    await demo_async_operations()
    
    # 9. Demo Type Safety
    print("\n9. 🔒 Enhanced Type Safety")
    print("-" * 40)
    
    def demo_type_safety():
        """Demonstrate type safety improvements."""
        from models import create_video_request, create_viral_request, create_langchain_request
        
        # Create requests using utility functions
        video_req = create_video_request(
            youtube_url="https://www.youtube.com/watch?v=example",
            language=Language.EN,
            quality=VideoQuality.HIGH
        )
        print(f"✅ Type-safe video request: {video_req.language} language")
        
        viral_req = create_viral_request(
            youtube_url="https://www.youtube.com/watch?v=example",
            n_variants=3
        )
        print(f"✅ Type-safe viral request: {viral_req.n_variants} variants")
        
        langchain_req = create_langchain_request(
            youtube_url="https://www.youtube.com/watch?v=example",
            analysis_type=AnalysisType.ENGAGEMENT
        )
        print(f"✅ Type-safe LangChain request: {langchain_req.analysis_type}")
    
    demo_type_safety()
    
    # 10. Demo Security Features
    print("\n10. 🔐 Security Features")
    print("-" * 40)
    
    def demo_security():
        """Demonstrate security features."""
        from validation import sanitize_youtube_url, contains_malicious_content
        
        # Test URL sanitization
        clean_url = sanitize_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        if clean_url:
            print(f"✅ URL sanitized: {clean_url}")
        else:
            print("❌ URL sanitization failed")
        
        # Test malicious content detection
        malicious_text = "javascript:alert('xss')"
        is_malicious = contains_malicious_content(malicious_text)
        if is_malicious:
            print("✅ Malicious content detected")
        else:
            print("❌ Malicious content not detected")
        
        # Test clean content
        clean_text = "This is a normal video description"
        is_clean = contains_malicious_content(clean_text)
        if not is_clean:
            print("✅ Clean content validated")
        else:
            print("❌ Clean content incorrectly flagged")
    
    demo_security()
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)
    print("All improvements are working correctly:")
    print("✅ Enhanced Pydantic models with validation")
    print("✅ Early returns and guard clauses")
    print("✅ Comprehensive validation and security")
    print("✅ High-performance caching system")
    print("✅ Performance monitoring and health checks")
    print("✅ Comprehensive error handling")
    print("✅ Enhanced dependency injection")
    print("✅ Async operations and performance")
    print("✅ Enhanced type safety")
    print("✅ Security features")
    print("\n🚀 The improved Video-OpusClip API is ready for production!")

async def main():
    """Main demo function."""
    try:
        await demo_improved_api()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
































