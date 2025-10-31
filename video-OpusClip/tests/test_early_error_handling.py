"""
Test Suite for Early Error Handling

Tests that errors and edge cases are handled at the beginning of functions
following the "fail fast" principle.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from ..error_handling import (
    ErrorHandler, 
    ErrorCode, 
    ValidationError, 
    ProcessingError, 
    ExternalServiceError,
    ResourceError,
    CriticalSystemError,
    SecurityError,
    ConfigurationError,
    create_validation_error,
    create_processing_error,
    create_external_service_error,
    create_resource_error,
    create_critical_system_error,
    create_security_error,
    create_configuration_error
)
from ..validation import (
    validate_youtube_url,
    validate_clip_length,
    validate_batch_size,
    validate_video_request_data,
    validate_batch_request_data,
    validate_viral_variant_data,
    check_system_resources,
    check_gpu_availability,
    validate_system_health,
    validate_gpu_health
)

# =============================================================================
# EARLY ERROR HANDLING TESTS
# =============================================================================

class TestEarlyErrorHandling:
    """Test early error handling at the beginning of functions."""
    
    def test_validate_youtube_url_early_none_check(self):
        """Test that None URLs are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(None)
        
        assert "YouTube URL is required" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_youtube_url_early_empty_check(self):
        """Test that empty URLs are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url("")
        
        assert "YouTube URL is required" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_youtube_url_early_type_check(self):
        """Test that wrong data types are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(123)
        
        assert "YouTube URL must be a string" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_youtube_url_early_length_check(self):
        """Test that extremely long URLs are caught early."""
        long_url = "https://youtube.com/watch?v=" + "a" * 3000
        
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(long_url)
        
        assert "too long" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_youtube_url_early_malicious_check(self):
        """Test that malicious patterns are caught early."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "eval('malicious_code')"
        ]
        
        for url in malicious_urls:
            with pytest.raises(ValidationError) as exc_info:
                validate_youtube_url(url)
            
            assert "Malicious URL pattern detected" in str(exc_info.value)
            assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL

# =============================================================================
# CLIP LENGTH EARLY VALIDATION TESTS
# =============================================================================

class TestClipLengthEarlyValidation:
    """Test early validation for clip length parameters."""
    
    def test_validate_clip_length_early_type_check(self):
        """Test that wrong data types are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length("30")
        
        assert "Clip length must be an integer" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH
    
    def test_validate_clip_length_early_negative_check(self):
        """Test that negative values are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(-5)
        
        assert "cannot be negative" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH
    
    def test_validate_clip_length_early_zero_check(self):
        """Test that zero values are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(0)
        
        assert "cannot be zero" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH
    
    def test_validate_clip_length_early_overflow_check(self):
        """Test that unrealistic values are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(100000)  # Over 24 hours
        
        assert "exceeds maximum allowed duration" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH

# =============================================================================
# BATCH SIZE EARLY VALIDATION TESTS
# =============================================================================

class TestBatchSizeEarlyValidation:
    """Test early validation for batch size parameters."""
    
    def test_validate_batch_size_early_type_check(self):
        """Test that wrong data types are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size("10")
        
        assert "Batch size must be an integer" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_size_early_negative_check(self):
        """Test that negative values are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(-10)
        
        assert "cannot be negative" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_size_early_zero_check(self):
        """Test that zero values are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(0)
        
        assert "cannot be zero" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_size_early_dos_check(self):
        """Test that extremely large batches are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(2000)  # Over limit
        
        assert "exceeds maximum allowed limit" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE

# =============================================================================
# VIDEO REQUEST DATA EARLY VALIDATION TESTS
# =============================================================================

class TestVideoRequestDataEarlyValidation:
    """Test early validation for video request data."""
    
    def test_validate_video_request_data_early_none_url(self):
        """Test that None YouTube URL is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url=None,
                language="en"
            )
        
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_video_request_data_early_empty_url(self):
        """Test that empty YouTube URL is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="",
                language="en"
            )
        
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_video_request_data_early_none_language(self):
        """Test that None language is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="https://youtube.com/watch?v=123",
                language=None
            )
        
        assert "Language is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_LANGUAGE_CODE
    
    def test_validate_video_request_data_early_wrong_types(self):
        """Test that wrong data types are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url=123,
                language="en"
            )
        
        assert "YouTube URL must be a string" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_video_request_data_early_logical_constraints(self):
        """Test that logical constraints are validated early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="https://youtube.com/watch?v=123",
                language="en",
                max_clip_length=10,
                min_clip_length=20
            )
        
        assert "max_clip_length cannot be less than min_clip_length" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH

# =============================================================================
# BATCH REQUEST DATA EARLY VALIDATION TESTS
# =============================================================================

class TestBatchRequestDataEarlyValidation:
    """Test early validation for batch request data."""
    
    def test_validate_batch_request_data_early_none_requests(self):
        """Test that None requests list is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(None)
        
        assert "Requests list cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_request_data_early_empty_requests(self):
        """Test that empty requests list is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data([])
        
        assert "Requests list cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_request_data_early_wrong_type(self):
        """Test that wrong data type is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data("not a list")
        
        assert "Requests must be a list" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_request_data_early_size_limit(self):
        """Test that size limits are checked early."""
        large_requests = [{"youtube_url": "https://youtube.com/watch?v=123", "language": "en"} for _ in range(1500)]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(large_requests)
        
        assert "exceeds maximum limit of 1000" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_BATCH_SIZE
    
    def test_validate_batch_request_data_early_none_request(self):
        """Test that None requests in list are caught early."""
        requests = [
            {"youtube_url": "https://youtube.com/watch?v=123", "language": "en"},
            None,
            {"youtube_url": "https://youtube.com/watch?v=456", "language": "es"}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(requests)
        
        assert "Request at index 1 cannot be None" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_YOUTUBE_URL
    
    def test_validate_batch_request_data_early_missing_fields(self):
        """Test that missing required fields are caught early."""
        requests = [
            {"youtube_url": "https://youtube.com/watch?v=123", "language": "en"},
            {"youtube_url": "https://youtube.com/watch?v=456"},  # Missing language
            {"youtube_url": "https://youtube.com/watch?v=789", "language": "es"}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(requests)
        
        assert "Language is required for request at index 1" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_LANGUAGE_CODE

# =============================================================================
# VIRAL VARIANT DATA EARLY VALIDATION TESTS
# =============================================================================

class TestViralVariantDataEarlyValidation:
    """Test early validation for viral variant data."""
    
    def test_validate_viral_variant_data_early_none_caption(self):
        """Test that None caption is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption=None,
                viral_score=0.8,
                variant_id="test_123"
            )
        
        assert "Caption is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CAPTION
    
    def test_validate_viral_variant_data_early_empty_caption(self):
        """Test that empty caption is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="",
                viral_score=0.8,
                variant_id="test_123"
            )
        
        assert "Caption is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CAPTION
    
    def test_validate_viral_variant_data_early_none_variant_id(self):
        """Test that None variant ID is caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id=None
            )
        
        assert "Variant ID is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_VARIANT_ID
    
    def test_validate_viral_variant_data_early_wrong_types(self):
        """Test that wrong data types are caught early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start="0.0",  # Should be float
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id="test_123"
            )
        
        assert "Start time must be a number" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH
    
    def test_validate_viral_variant_data_early_numeric_constraints(self):
        """Test that numeric constraints are validated early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="Test caption",
                viral_score=1.5,  # Should be <= 1.0
                variant_id="test_123"
            )
        
        assert "Viral score must be between 0.0 and 1.0" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_VIRAL_SCORE
    
    def test_validate_viral_variant_data_early_logical_constraints(self):
        """Test that logical constraints are validated early."""
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=15.0,
                end=10.0,  # End before start
                caption="Test caption",
                viral_score=0.8,
                variant_id="test_123"
            )
        
        assert "Start time must be less than end time" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.INVALID_CLIP_LENGTH

# =============================================================================
# SYSTEM HEALTH EARLY VALIDATION TESTS
# =============================================================================

class TestSystemHealthEarlyValidation:
    """Test early validation for system health checks."""
    
    @patch('psutil.virtual_memory')
    def test_validate_system_health_early_critical_memory(self, mock_memory):
        """Test that critical memory usage is caught early."""
        mock_memory.return_value = Mock(percent=95.0)
        
        with pytest.raises(CriticalSystemError) as exc_info:
            validate_system_health()
        
        assert "Critical memory usage detected" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.GPU_MEMORY_EXHAUSTED
    
    @patch('psutil.disk_usage')
    def test_validate_system_health_early_critical_disk(self, mock_disk):
        """Test that critical disk usage is caught early."""
        mock_disk.return_value = Mock(percent=98.0)
        
        with pytest.raises(CriticalSystemError) as exc_info:
            validate_system_health()
        
        assert "Critical disk space detected" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.DISK_SPACE_CRITICAL
    
    @patch('torch.cuda.is_available')
    def test_validate_gpu_health_early_no_gpu(self, mock_cuda_available):
        """Test that missing GPU is caught early."""
        mock_cuda_available.return_value = False
        
        with pytest.raises(ResourceError) as exc_info:
            validate_gpu_health()
        
        assert "GPU not available for processing" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.GPU_NOT_AVAILABLE

# =============================================================================
# PERFORMANCE TESTS FOR EARLY ERROR HANDLING
# =============================================================================

class TestEarlyErrorHandlingPerformance:
    """Test performance benefits of early error handling."""
    
    def test_early_validation_performance(self):
        """Test that early validation is faster than deep validation."""
        import time
        
        # Test early failure (should be fast)
        start_time = time.time()
        try:
            validate_youtube_url(None)
        except ValidationError:
            early_failure_time = time.time() - start_time
        
        # Test deep validation (should be slower)
        start_time = time.time()
        try:
            validate_youtube_url("https://youtube.com/watch?v=invalid_format_that_requires_regex")
        except ValidationError:
            deep_failure_time = time.time() - start_time
        
        # Early failure should be significantly faster
        assert early_failure_time < deep_failure_time * 0.1  # At least 10x faster
    
    def test_batch_validation_early_failure_performance(self):
        """Test that batch validation fails early for large invalid batches."""
        import time
        
        # Create a large batch with an early error
        large_batch = [{"youtube_url": "https://youtube.com/watch?v=123", "language": "en"} for _ in range(100)]
        large_batch[50] = None  # Insert None to trigger early failure
        
        start_time = time.time()
        try:
            validate_batch_request_data(large_batch)
        except ValidationError:
            early_failure_time = time.time() - start_time
        
        # Should fail quickly at index 50, not process all 100 items
        assert early_failure_time < 0.1  # Should be very fast
    
    def test_composite_validation_early_failure_performance(self):
        """Test that composite validation fails early."""
        import time
        
        start_time = time.time()
        try:
            validate_video_request_data(
                youtube_url=None,  # Should fail immediately
                language="en"
            )
        except ValidationError:
            early_failure_time = time.time() - start_time
        
        # Should fail immediately without processing other validations
        assert early_failure_time < 0.01  # Should be very fast

# =============================================================================
# INTEGRATION TESTS FOR EARLY ERROR HANDLING
# =============================================================================

class TestEarlyErrorHandlingIntegration:
    """Test integration of early error handling across components."""
    
    def test_api_endpoint_early_validation_flow(self):
        """Test that API endpoints validate early and fail fast."""
        from ..api import process_video
        
        # Mock request object with None URL
        mock_request = Mock()
        mock_request.youtube_url = None
        mock_request.language = "en"
        
        # Should fail early without reaching processing
        with pytest.raises(ValidationError) as exc_info:
            process_video(mock_request, Mock(), Mock())
        
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
    
    def test_batch_api_endpoint_early_validation_flow(self):
        """Test that batch API endpoints validate early and fail fast."""
        from ..api import process_video_batch
        
        # Mock batch request with None requests list
        mock_batch_request = Mock()
        mock_batch_request.requests = None
        
        # Should fail early without reaching processing
        with pytest.raises(ValidationError) as exc_info:
            process_video_batch(mock_batch_request, Mock(), Mock())
        
        assert "Batch request object is required" in str(exc_info.value)
    
    def test_gradio_demo_early_validation_flow(self):
        """Test that Gradio demo validates early and fails fast."""
        from ..gradio_demo import generate_image
        
        # Test with None prompt
        image, error = generate_image(None)
        
        assert image is None
        assert "Prompt cannot be None" in error

# =============================================================================
# EDGE CASE TESTS FOR EARLY ERROR HANDLING
# =============================================================================

class TestEarlyErrorHandlingEdgeCases:
    """Test edge cases for early error handling."""
    
    def test_validate_youtube_url_edge_cases(self):
        """Test edge cases for YouTube URL validation."""
        edge_cases = [
            ("", "empty string"),
            ("   ", "whitespace only"),
            (123, "integer"),
            (None, "None"),
            ([], "empty list"),
            ({}, "empty dict"),
            (True, "boolean"),
            (3.14, "float")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_youtube_url(value)
            
            assert "YouTube URL" in str(exc_info.value), f"Failed for {description}"
    
    def test_validate_clip_length_edge_cases(self):
        """Test edge cases for clip length validation."""
        edge_cases = [
            ("30", "string"),
            (None, "None"),
            (0, "zero"),
            (-1, "negative"),
            (3.14, "float"),
            (True, "boolean"),
            ([], "list"),
            ({}, "dict")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_clip_length(value)
            
            assert "Clip length" in str(exc_info.value), f"Failed for {description}"
    
    def test_validate_batch_size_edge_cases(self):
        """Test edge cases for batch size validation."""
        edge_cases = [
            ("10", "string"),
            (None, "None"),
            (0, "zero"),
            (-1, "negative"),
            (3.14, "float"),
            (True, "boolean"),
            ([], "list"),
            ({}, "dict")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_batch_size(value)
            
            assert "Batch size" in str(exc_info.value), f"Failed for {description}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 