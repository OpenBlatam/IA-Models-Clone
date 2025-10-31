"""
Test Suite for Improved Video-OpusClip API

Comprehensive tests for all improvements:
- Model validation tests
- Error handling tests
- Performance tests
- Security tests
- Integration tests
"""

import pytest
import asyncio
import time
from typing import Dict, Any

# Import improved modules
from models import (
    VideoClipRequest, VideoClipBatchRequest, ViralVideoRequest, LangChainRequest,
    Language, VideoQuality, VideoFormat, AnalysisType, Priority,
    ValidationResult
)
from validation import (
    validate_video_request, validate_batch_request, validate_viral_request,
    validate_langchain_request, sanitize_youtube_url, contains_malicious_content
)
from error_handling import (
    ValidationError, SecurityError, VideoProcessingError, ResourceError,
    handle_processing_errors, create_error_response
)
from cache import CacheManager, CacheConfig
from monitoring import PerformanceMonitor, HealthChecker, MonitoringConfig
from dependencies import DependencyConfig

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_video_request():
    """Create a valid video request for testing."""
    return VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        language=Language.EN,
        max_clip_length=60,
        min_clip_length=15,
        quality=VideoQuality.HIGH,
        format=VideoFormat.MP4,
        priority=Priority.NORMAL
    )

@pytest.fixture
def valid_viral_request():
    """Create a valid viral request for testing."""
    return ViralVideoRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        n_variants=5,
        use_langchain=True,
        platform="tiktok"
    )

@pytest.fixture
def valid_langchain_request():
    """Create a valid LangChain request for testing."""
    return LangChainRequest(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        analysis_type=AnalysisType.COMPREHENSIVE,
        platform="youtube"
    )

@pytest.fixture
async def cache_manager():
    """Create a cache manager for testing."""
    config = CacheConfig(enable_fallback=True, fallback_max_size=100)
    manager = CacheManager(config)
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.fixture
async def performance_monitor():
    """Create a performance monitor for testing."""
    config = MonitoringConfig(enable_performance_monitoring=True)
    monitor = PerformanceMonitor(config)
    await monitor.start()
    yield monitor
    await monitor.stop()

@pytest.fixture
async def health_checker():
    """Create a health checker for testing."""
    config = MonitoringConfig(enable_health_checks=True)
    checker = HealthChecker(config)
    await checker.initialize()
    yield checker
    await checker.close()

# =============================================================================
# MODEL VALIDATION TESTS
# =============================================================================

class TestModelValidation:
    """Test enhanced Pydantic models and validation."""
    
    def test_valid_video_request(self, valid_video_request):
        """Test valid video request creation."""
        assert valid_video_request.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert valid_video_request.language == Language.EN
        assert valid_video_request.max_clip_length == 60
        assert valid_video_request.quality == VideoQuality.HIGH
    
    def test_invalid_youtube_url(self):
        """Test invalid YouTube URL validation."""
        with pytest.raises(ValidationError):
            VideoClipRequest(
                youtube_url="invalid_url",
                language=Language.EN
            )
    
    def test_invalid_clip_lengths(self):
        """Test invalid clip length validation."""
        with pytest.raises(ValidationError):
            VideoClipRequest(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language=Language.EN,
                min_clip_length=60,
                max_clip_length=30  # min > max
            )
    
    def test_malicious_url_detection(self):
        """Test malicious URL detection."""
        with pytest.raises(ValidationError):
            VideoClipRequest(
                youtube_url="javascript:alert('xss')",
                language=Language.EN
            )
    
    def test_viral_request_validation(self, valid_viral_request):
        """Test viral request validation."""
        assert valid_viral_request.n_variants == 5
        assert valid_viral_request.platform == "tiktok"
        assert valid_viral_request.use_langchain is True
    
    def test_invalid_variant_count(self):
        """Test invalid variant count validation."""
        with pytest.raises(ValidationError):
            ViralVideoRequest(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                n_variants=100  # Too many variants
            )
    
    def test_langchain_request_validation(self, valid_langchain_request):
        """Test LangChain request validation."""
        assert valid_langchain_request.analysis_type == AnalysisType.COMPREHENSIVE
        assert valid_langchain_request.platform == "youtube"

# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test comprehensive validation functions."""
    
    def test_validate_video_request_success(self, valid_video_request):
        """Test successful video request validation."""
        result = validate_video_request(valid_video_request)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_video_request_failure(self):
        """Test failed video request validation."""
        invalid_request = VideoClipRequest(
            youtube_url="",  # Empty URL
            language=Language.EN
        )
        result = validate_video_request(invalid_request)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_batch_request_success(self, valid_video_request):
        """Test successful batch request validation."""
        batch_request = VideoClipBatchRequest(
            requests=[valid_video_request],
            max_workers=4
        )
        result = validate_batch_request(batch_request)
        assert result.is_valid is True
    
    def test_validate_batch_request_empty(self):
        """Test empty batch request validation."""
        batch_request = VideoClipBatchRequest(requests=[])
        result = validate_batch_request(batch_request)
        assert result.is_valid is False
        assert "empty" in str(result.errors[0]).lower()
    
    def test_validate_batch_request_too_large(self, valid_video_request):
        """Test batch request with too many items."""
        batch_request = VideoClipBatchRequest(
            requests=[valid_video_request] * 101,  # Too many requests
            max_workers=4
        )
        result = validate_batch_request(batch_request)
        assert result.is_valid is False
    
    def test_sanitize_youtube_url_valid(self):
        """Test valid YouTube URL sanitization."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        sanitized = sanitize_youtube_url(url)
        assert sanitized == url
    
    def test_sanitize_youtube_url_invalid(self):
        """Test invalid YouTube URL sanitization."""
        url = "javascript:alert('xss')"
        sanitized = sanitize_youtube_url(url)
        assert sanitized is None
    
    def test_contains_malicious_content_detection(self):
        """Test malicious content detection."""
        malicious_texts = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "eval(malicious_code)",
            "<script>alert('xss')</script>"
        ]
        
        for text in malicious_texts:
            assert contains_malicious_content(text) is True
    
    def test_contains_malicious_content_clean(self):
        """Test clean content validation."""
        clean_texts = [
            "This is a normal video description",
            "Learn how to create amazing videos",
            "Subscribe for more content",
            "Check out this tutorial"
        ]
        
        for text in clean_texts:
            assert contains_malicious_content(text) is False

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_validation_error_creation(self):
        """Test validation error creation."""
        error = ValidationError("Test validation error", field="test_field")
        assert error.message == "Test validation error"
        assert error.field == "test_field"
        assert error.error_code == "VALIDATION_ERROR"
    
    def test_security_error_creation(self):
        """Test security error creation."""
        error = SecurityError("Test security error", threat_type="xss")
        assert error.message == "Test security error"
        assert error.threat_type == "xss"
        assert error.error_code == "SECURITY_ERROR"
    
    def test_video_processing_error_creation(self):
        """Test video processing error creation."""
        original_error = Exception("Original error")
        error = VideoProcessingError(
            "Processing failed",
            original_error=original_error
        )
        assert error.message == "Processing failed"
        assert error.original_error == original_error
        assert error.error_code == "PROCESSING_ERROR"
    
    def test_create_error_response(self):
        """Test error response creation."""
        response = create_error_response(
            error_code="TEST_ERROR",
            message="Test error message",
            request_id="test-123"
        )
        assert response["error"]["code"] == "TEST_ERROR"
        assert response["error"]["message"] == "Test error message"
        assert response["error"]["request_id"] == "test-123"
    
    def test_handle_processing_errors_decorator(self):
        """Test error handling decorator."""
        @handle_processing_errors
        async def test_function(should_fail=False):
            if should_fail:
                raise ValidationError("Test error")
            return "success"
        
        # Test successful execution
        result = asyncio.run(test_function(False))
        assert result == "success"
        
        # Test error handling
        with pytest.raises(ValidationError):
            asyncio.run(test_function(True))

# =============================================================================
# CACHING TESTS
# =============================================================================

class TestCaching:
    """Test caching system."""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager):
        """Test cache set and get operations."""
        key = "test_key"
        value = {"test": "data", "number": 42}
        
        # Set value
        success = await cache_manager.set(key, value, ttl=60)
        assert success is True
        
        # Get value
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value == value
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_manager):
        """Test cache miss scenario."""
        non_existent_key = "non_existent_key"
        value = await cache_manager.get(non_existent_key)
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_manager):
        """Test cache delete operation."""
        key = "delete_test_key"
        value = {"delete": "me"}
        
        # Set value
        await cache_manager.set(key, value, ttl=60)
        
        # Verify it exists
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value == value
        
        # Delete value
        success = await cache_manager.delete(key)
        assert success is True
        
        # Verify it's deleted
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        # Perform some operations
        await cache_manager.set("key1", "value1", ttl=60)
        await cache_manager.get("key1")  # Hit
        await cache_manager.get("key2")  # Miss
        
        stats = cache_manager.get_stats()
        assert stats['total_requests'] == 2
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate_percent'] == 50.0

# =============================================================================
# MONITORING TESTS
# =============================================================================

class TestMonitoring:
    """Test monitoring and health checking."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, performance_monitor):
        """Test performance monitoring."""
        # Record some requests
        await performance_monitor.record_request("POST", "/api/v1/video/process", 200, 1.5)
        await performance_monitor.record_request("POST", "/api/v1/viral/process", 200, 2.3)
        await performance_monitor.record_request("GET", "/health", 200, 0.1)
        
        # Get metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 3
        assert metrics['performance']['response_time_avg'] > 0
        assert len(metrics['endpoints']) == 3
    
    @pytest.mark.asyncio
    async def test_health_checking(self, health_checker):
        """Test health checking."""
        health_status = await health_checker.check_system_health()
        assert health_status is not None
        assert hasattr(health_status, 'is_healthy')
        assert hasattr(health_status, 'status')
        assert hasattr(health_status, 'system_metrics')
        assert hasattr(health_status, 'gpu_metrics')
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self):
        """Test health checker initialization."""
        config = MonitoringConfig(enable_health_checks=True)
        checker = HealthChecker(config)
        await checker.initialize()
        
        # Verify initialization
        assert checker._running is True
        
        await checker.close()

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration between components."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, valid_video_request, cache_manager, performance_monitor):
        """Test full workflow integration."""
        # Simulate video processing workflow
        start_time = time.perf_counter()
        
        # 1. Validate request
        validation_result = validate_video_request(valid_video_request)
        assert validation_result.is_valid is True
        
        # 2. Check cache
        cache_key = f"video:{valid_video_request.youtube_url}"
        cached_result = await cache_manager.get(cache_key)
        assert cached_result is None  # Should be cache miss
        
        # 3. Simulate processing
        await asyncio.sleep(0.1)  # Simulate processing time
        processing_time = time.perf_counter() - start_time
        
        # 4. Cache result
        result = {"processed": True, "duration": processing_time}
        await cache_manager.set(cache_key, result, ttl=60)
        
        # 5. Record performance
        await performance_monitor.record_request(
            "POST", "/api/v1/video/process", 200, processing_time
        )
        
        # 6. Verify cache hit
        cached_result = await cache_manager.get(cache_key)
        assert cached_result == result
        
        # 7. Verify performance metrics
        metrics = performance_monitor.get_metrics()
        assert metrics['performance']['request_count'] == 1
        assert metrics['performance']['response_time_avg'] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery workflow."""
        @handle_processing_errors
        async def failing_function():
            raise ValidationError("Simulated error")
        
        # Test error handling
        with pytest.raises(ValidationError):
            await failing_function()
        
        # Test error response creation
        error_response = create_error_response(
            error_code="VALIDATION_ERROR",
            message="Test error",
            request_id="test-123"
        )
        assert error_response["error"]["code"] == "VALIDATION_ERROR"

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache_manager):
        """Test concurrent cache operations."""
        async def cache_operation(i):
            key = f"concurrent_key_{i}"
            value = f"value_{i}"
            await cache_manager.set(key, value, ttl=60)
            return await cache_manager.get(key)
        
        # Run 10 concurrent operations
        tasks = [cache_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, valid_video_request):
        """Test batch processing performance."""
        # Create batch request
        batch_request = VideoClipBatchRequest(
            requests=[valid_video_request] * 5,
            max_workers=3
        )
        
        # Validate batch
        start_time = time.perf_counter()
        validation_result = validate_batch_request(batch_request)
        validation_time = time.perf_counter() - start_time
        
        assert validation_result.is_valid is True
        assert validation_time < 1.0  # Should be fast
    
    def test_model_creation_performance(self):
        """Test model creation performance."""
        start_time = time.perf_counter()
        
        # Create 100 video requests
        requests = []
        for i in range(100):
            request = VideoClipRequest(
                youtube_url=f"https://www.youtube.com/watch?v=test{i}",
                language=Language.EN,
                max_clip_length=60
            )
            requests.append(request)
        
        creation_time = time.perf_counter() - start_time
        
        assert len(requests) == 100
        assert creation_time < 1.0  # Should be fast

# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Test security features."""
    
    def test_url_sanitization_security(self):
        """Test URL sanitization for security."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "file:///etc/passwd",
            "ftp://malicious.com/steal_data"
        ]
        
        for url in malicious_urls:
            sanitized = sanitize_youtube_url(url)
            assert sanitized is None, f"Malicious URL not blocked: {url}"
    
    def test_content_validation_security(self):
        """Test content validation for security."""
        malicious_content = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "onload=alert('xss')",
            "onclick=alert('xss')",
            "eval(malicious_code)"
        ]
        
        for content in malicious_content:
            is_malicious = contains_malicious_content(content)
            assert is_malicious is True, f"Malicious content not detected: {content}"
    
    def test_input_length_limits(self):
        """Test input length limits for security."""
        # Test extremely long URL
        long_url = "https://www.youtube.com/watch?v=" + "a" * 1000
        with pytest.raises(ValidationError):
            VideoClipRequest(
                youtube_url=long_url,
                language=Language.EN
            )
    
    def test_batch_size_limits(self, valid_video_request):
        """Test batch size limits for security."""
        # Test batch with too many requests
        large_batch = VideoClipBatchRequest(
            requests=[valid_video_request] * 1000,  # Way too many
            max_workers=4
        )
        
        result = validate_batch_request(large_batch)
        assert result.is_valid is False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])






























