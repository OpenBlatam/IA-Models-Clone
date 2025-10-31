"""
Integration Test for Improved Video-OpusClip API

Comprehensive integration tests to ensure all components work together:
- API endpoints integration
- Processor integration
- Cache integration
- Monitoring integration
- Error handling integration
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from fastapi.testclient import TestClient

# Import improved modules
from improved_api import app
from models import (
    VideoClipRequest, VideoClipBatchRequest, ViralVideoRequest, LangChainRequest,
    Language, VideoQuality, VideoFormat, AnalysisType, Priority
)
from processors import (
    VideoProcessor, VideoProcessorConfig,
    ViralVideoProcessor, ViralProcessorConfig,
    LangChainVideoProcessor, LangChainConfig,
    BatchVideoProcessor, BatchProcessorConfig
)
from cache import CacheManager, CacheConfig
from monitoring import PerformanceMonitor, HealthChecker, MonitoringConfig
from dependencies import DependencyConfig

# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)

@pytest.fixture
async def integrated_processors():
    """Create integrated processor instances."""
    # Video processor
    video_config = VideoProcessorConfig()
    video_processor = VideoProcessor(video_config)
    await video_processor.initialize()
    
    # Viral processor
    viral_config = ViralProcessorConfig()
    viral_processor = ViralVideoProcessor(viral_config)
    await viral_processor.initialize()
    
    # LangChain processor
    langchain_config = LangChainConfig()
    langchain_processor = LangChainVideoProcessor(langchain_config)
    await langchain_processor.initialize()
    
    # Batch processor
    batch_config = BatchProcessorConfig()
    batch_processor = BatchVideoProcessor(batch_config)
    await batch_processor.initialize()
    
    yield {
        'video': video_processor,
        'viral': viral_processor,
        'langchain': langchain_processor,
        'batch': batch_processor
    }
    
    # Cleanup
    await video_processor.close()
    await viral_processor.close()
    await langchain_processor.close()
    await batch_processor.close()

@pytest.fixture
async def integrated_cache():
    """Create integrated cache manager."""
    config = CacheConfig(enable_fallback=True, fallback_max_size=100)
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    yield cache_manager
    await cache_manager.close()

@pytest.fixture
async def integrated_monitoring():
    """Create integrated monitoring system."""
    config = MonitoringConfig(
        enable_performance_monitoring=True,
        enable_health_checks=True
    )
    
    performance_monitor = PerformanceMonitor(config)
    await performance_monitor.start()
    
    health_checker = HealthChecker(config)
    await health_checker.initialize()
    
    yield {
        'performance': performance_monitor,
        'health': health_checker
    }
    
    await performance_monitor.stop()
    await health_checker.close()

# =============================================================================
# API INTEGRATION TESTS
# =============================================================================

class TestAPIIntegration:
    """Test API endpoint integration."""
    
    def test_health_endpoint(self, client):
        """Test health endpoint integration."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "system_metrics" in data
        assert "gpu_metrics" in data
        assert "timestamp" in data
    
    def test_video_processing_endpoint(self, client):
        """Test video processing endpoint integration."""
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en",
            "max_clip_length": 60,
            "quality": "high",
            "format": "mp4"
        }
        
        response = client.post("/api/v1/video/process", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["youtube_url"] == request_data["youtube_url"]
        assert "processing_time" in data
    
    def test_batch_processing_endpoint(self, client):
        """Test batch processing endpoint integration."""
        request_data = {
            "requests": [
                {
                    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "language": "en",
                    "max_clip_length": 60
                },
                {
                    "youtube_url": "https://www.youtube.com/watch?v=example2",
                    "language": "es",
                    "max_clip_length": 45
                }
            ],
            "max_workers": 4
        }
        
        response = client.post("/api/v1/video/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total_requests"] == 2
        assert len(data["results"]) == 2
    
    def test_viral_processing_endpoint(self, client):
        """Test viral processing endpoint integration."""
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "n_variants": 3,
            "use_langchain": True,
            "platform": "tiktok"
        }
        
        response = client.post("/api/v1/viral/process", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["youtube_url"] == request_data["youtube_url"]
        assert len(data["variants"]) == 3
    
    def test_langchain_analysis_endpoint(self, client):
        """Test LangChain analysis endpoint integration."""
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "analysis_type": "comprehensive",
            "platform": "youtube"
        }
        
        response = client.post("/api/v1/langchain/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["youtube_url"] == request_data["youtube_url"]
        assert data["analysis_type"] == "comprehensive"
    
    def test_error_handling_integration(self, client):
        """Test error handling integration."""
        # Test invalid URL
        request_data = {
            "youtube_url": "invalid_url",
            "language": "en"
        }
        
        response = client.post("/api/v1/video/process", json=request_data)
        assert response.status_code == 422  # Validation error
        
        # Test empty batch
        request_data = {
            "requests": [],
            "max_workers": 4
        }
        
        response = client.post("/api/v1/video/batch", json=request_data)
        assert response.status_code == 422  # Validation error

# =============================================================================
# PROCESSOR INTEGRATION TESTS
# =============================================================================

class TestProcessorIntegration:
    """Test processor integration."""
    
    @pytest.mark.asyncio
    async def test_video_processor_integration(self, integrated_processors):
        """Test video processor integration."""
        processor = integrated_processors['video']
        
        request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language=Language.EN,
            max_clip_length=60,
            quality=VideoQuality.HIGH
        )
        
        response = await processor.process_video_async(request)
        assert response.success is True
        assert response.youtube_url == request.youtube_url
    
    @pytest.mark.asyncio
    async def test_viral_processor_integration(self, integrated_processors):
        """Test viral processor integration."""
        processor = integrated_processors['viral']
        
        request = ViralVideoRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            n_variants=3,
            use_langchain=True,
            platform="tiktok"
        )
        
        response = await processor.process_viral_variants_async(request)
        assert response.success is True
        assert response.youtube_url == request.youtube_url
        assert len(response.variants) == 3
    
    @pytest.mark.asyncio
    async def test_langchain_processor_integration(self, integrated_processors):
        """Test LangChain processor integration."""
        processor = integrated_processors['langchain']
        
        request = LangChainRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            analysis_type=AnalysisType.COMPREHENSIVE,
            platform="youtube"
        )
        
        response = await processor.analyze_content_async(request)
        assert response.success is True
        assert response.youtube_url == request.youtube_url
        assert response.analysis_type == AnalysisType.COMPREHENSIVE
    
    @pytest.mark.asyncio
    async def test_batch_processor_integration(self, integrated_processors):
        """Test batch processor integration."""
        processor = integrated_processors['batch']
        
        requests = [
            VideoClipRequest(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language=Language.EN,
                max_clip_length=60
            ),
            VideoClipRequest(
                youtube_url="https://www.youtube.com/watch?v=example2",
                language=Language.ES,
                max_clip_length=45
            )
        ]
        
        responses = await processor.process_batch_async(requests)
        assert len(responses) == 2
        assert all(r.success for r in responses)

# =============================================================================
# CACHE INTEGRATION TESTS
# =============================================================================

class TestCacheIntegration:
    """Test cache integration."""
    
    @pytest.mark.asyncio
    async def test_cache_with_processors(self, integrated_cache, integrated_processors):
        """Test cache integration with processors."""
        video_processor = integrated_processors['video']
        
        request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language=Language.EN,
            max_clip_length=60
        )
        
        # First request - should be cache miss
        cache_key = f"video:{request.youtube_url}:{request.language}"
        cached_result = await integrated_cache.get(cache_key)
        assert cached_result is None
        
        # Process video
        response = await video_processor.process_video_async(request)
        assert response.success is True
        
        # Cache result
        await integrated_cache.set(cache_key, response.dict(), ttl=60)
        
        # Second request - should be cache hit
        cached_result = await integrated_cache.get(cache_key)
        assert cached_result is not None
        assert cached_result["youtube_url"] == request.youtube_url
    
    @pytest.mark.asyncio
    async def test_cache_fallback_integration(self, integrated_cache):
        """Test cache fallback integration."""
        # Test fallback cache
        key = "fallback_test"
        value = {"test": "data"}
        
        # Set in fallback cache
        await integrated_cache.set(key, value, ttl=60)
        
        # Get from fallback cache
        result = await integrated_cache.get(key)
        assert result == value
        
        # Test cache statistics
        stats = integrated_cache.get_stats()
        assert stats['fallback_hits'] >= 0
        assert stats['fallback_misses'] >= 0

# =============================================================================
# MONITORING INTEGRATION TESTS
# =============================================================================

class TestMonitoringIntegration:
    """Test monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_monitoring_with_processors(self, integrated_monitoring, integrated_processors):
        """Test monitoring integration with processors."""
        performance_monitor = integrated_monitoring['performance']
        health_checker = integrated_monitoring['health']
        
        # Record some performance metrics
        await performance_monitor.record_request("POST", "/api/v1/video/process", 200, 1.5)
        await performance_monitor.record_request("POST", "/api/v1/viral/process", 200, 2.3)
        
        # Get performance metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 2
        assert metrics['performance']['response_time_avg'] > 0
        
        # Check health status
        health_status = await health_checker.check_system_health()
        assert health_status is not None
        assert hasattr(health_status, 'is_healthy')
    
    @pytest.mark.asyncio
    async def test_monitoring_with_cache(self, integrated_monitoring, integrated_cache):
        """Test monitoring integration with cache."""
        performance_monitor = integrated_monitoring['performance']
        
        # Simulate cache operations
        await integrated_cache.set("test_key", "test_value", ttl=60)
        await integrated_cache.get("test_key")
        await integrated_cache.get("non_existent_key")
        
        # Record cache performance
        await performance_monitor.record_request("GET", "/cache/hit", 200, 0.001)
        await performance_monitor.record_request("GET", "/cache/miss", 200, 0.002)
        
        # Verify metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] >= 2

# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_video_processing_workflow(self, integrated_processors, integrated_cache, integrated_monitoring):
        """Test complete video processing workflow."""
        video_processor = integrated_processors['video']
        performance_monitor = integrated_monitoring['performance']
        
        # Create request
        request = VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            language=Language.EN,
            max_clip_length=60,
            quality=VideoQuality.HIGH
        )
        
        # Check cache first
        cache_key = f"video:{request.youtube_url}:{request.language}"
        cached_result = await integrated_cache.get(cache_key)
        assert cached_result is None  # Should be cache miss
        
        # Process video
        start_time = time.perf_counter()
        response = await video_processor.process_video_async(request)
        processing_time = time.perf_counter() - start_time
        
        # Verify response
        assert response.success is True
        assert response.youtube_url == request.youtube_url
        
        # Cache result
        await integrated_cache.set(cache_key, response.dict(), ttl=60)
        
        # Record performance
        await performance_monitor.record_request("POST", "/api/v1/video/process", 200, processing_time)
        
        # Verify cache hit
        cached_result = await integrated_cache.get(cache_key)
        assert cached_result is not None
        assert cached_result["youtube_url"] == request.youtube_url
        
        # Verify performance metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 1
        assert metrics['performance']['response_time_avg'] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_monitoring(self, integrated_processors, integrated_monitoring):
        """Test batch processing with monitoring integration."""
        batch_processor = integrated_processors['batch']
        performance_monitor = integrated_monitoring['performance']
        
        # Create batch request
        requests = [
            VideoClipRequest(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language=Language.EN,
                max_clip_length=60
            ),
            VideoClipRequest(
                youtube_url="https://www.youtube.com/watch?v=example2",
                language=Language.ES,
                max_clip_length=45
            )
        ]
        
        # Process batch
        start_time = time.perf_counter()
        responses = await batch_processor.process_batch_async(requests)
        processing_time = time.perf_counter() - start_time
        
        # Verify responses
        assert len(responses) == 2
        assert all(r.success for r in responses)
        
        # Record performance
        await performance_monitor.record_request("POST", "/api/v1/video/batch", 200, processing_time)
        
        # Verify performance metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 1
        assert metrics['performance']['response_time_avg'] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, integrated_processors, integrated_monitoring):
        """Test error recovery workflow integration."""
        video_processor = integrated_processors['video']
        performance_monitor = integrated_monitoring['performance']
        
        # Test with invalid request
        invalid_request = VideoClipRequest(
            youtube_url="invalid_url",
            language=Language.EN
        )
        
        # This should raise an exception
        with pytest.raises(Exception):
            await video_processor.process_video_async(invalid_request)
        
        # Record error performance
        await performance_monitor.record_request("POST", "/api/v1/video/process", 422, 0.1)
        
        # Verify error metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 1
        assert metrics['performance']['error_count'] == 1

# =============================================================================
# PERFORMANCE INTEGRATION TESTS
# =============================================================================

class TestPerformanceIntegration:
    """Test performance integration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_integration(self, integrated_processors):
        """Test concurrent processing integration."""
        video_processor = integrated_processors['video']
        
        # Create multiple requests
        requests = [
            VideoClipRequest(
                youtube_url=f"https://www.youtube.com/watch?v=test{i}",
                language=Language.EN,
                max_clip_length=60
            )
            for i in range(5)
        ]
        
        # Process concurrently
        start_time = time.perf_counter()
        tasks = [video_processor.process_video_async(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        processing_time = time.perf_counter() - start_time
        
        # Verify all responses
        assert len(responses) == 5
        assert all(r.success for r in responses)
        
        # Verify concurrent processing was faster than sequential
        assert processing_time < 5.0  # Should be much faster than sequential
    
    @pytest.mark.asyncio
    async def test_cache_performance_integration(self, integrated_cache):
        """Test cache performance integration."""
        # Test cache performance
        start_time = time.perf_counter()
        
        # Perform many cache operations
        for i in range(100):
            key = f"perf_test_{i}"
            value = f"value_{i}"
            await integrated_cache.set(key, value, ttl=60)
            await integrated_cache.get(key)
        
        cache_time = time.perf_counter() - start_time
        
        # Verify cache performance
        assert cache_time < 1.0  # Should be very fast
        
        # Check cache statistics
        stats = integrated_cache.get_stats()
        assert stats['total_requests'] == 200  # 100 sets + 100 gets
        assert stats['hits'] == 100  # All gets should be hits

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])






























