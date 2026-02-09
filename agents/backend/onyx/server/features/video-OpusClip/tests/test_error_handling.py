"""
Comprehensive tests for error handling and validation system.
"""

import pytest
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from ..error_handling import (
    ErrorCode,
    ValidationError,
    ProcessingError,
    ExternalServiceError,
    ResourceError,
    ErrorHandler,
    ErrorResponse,
    create_validation_error,
    create_processing_error,
    create_external_service_error,
    create_resource_error
)
from ..validation import (
    validate_youtube_url,
    validate_language_code,
    validate_clip_length,
    validate_viral_score,
    validate_caption,
    validate_variant_id,
    validate_audience_profile,
    validate_batch_size,
    validate_video_request_data,
    validate_viral_variant_data,
    validate_batch_request_data,
    sanitize_youtube_url,
    extract_youtube_video_id,
    validate_and_sanitize_url
)

# =============================================================================
# ERROR CODE TESTS
# =============================================================================

def test_error_codes():
    """Test error code enumeration."""
    assert ErrorCode.INVALID_YOUTUBE_URL.value == 1001
    assert ErrorCode.VIDEO_PROCESSING_FAILED.value == 2001
    assert ErrorCode.YOUTUBE_API_ERROR.value == 3001
    assert ErrorCode.INSUFFICIENT_MEMORY.value == 4001
    assert ErrorCode.UNKNOWN_ERROR.value == 9001

# =============================================================================
# CUSTOM EXCEPTION TESTS
# =============================================================================

def test_validation_error():
    """Test ValidationError creation and properties."""
    error = ValidationError("Invalid URL", "youtube_url", "invalid-url")
    
    assert error.message == "Invalid URL"
    assert error.error_code == ErrorCode.INVALID_YOUTUBE_URL
    assert error.details["field"] == "youtube_url"
    assert error.details["value"] == "invalid-url"
    assert str(error) == "Invalid URL"

def test_processing_error():
    """Test ProcessingError creation and properties."""
    error = ProcessingError("Processing failed", "video_processing", {"duration": 10})
    
    assert error.message == "Processing failed"
    assert error.error_code == ErrorCode.VIDEO_PROCESSING_FAILED
    assert error.details["operation"] == "video_processing"
    assert error.details["duration"] == 10

def test_external_service_error():
    """Test ExternalServiceError creation and properties."""
    error = ExternalServiceError("API timeout", "youtube_api", 408)
    
    assert error.message == "API timeout"
    assert error.error_code == ErrorCode.YOUTUBE_API_ERROR
    assert error.details["service"] == "youtube_api"
    assert error.details["status_code"] == 408

def test_resource_error():
    """Test ResourceError creation and properties."""
    error = ResourceError("Memory insufficient", "gpu_memory", "4GB", "8GB")
    
    assert error.message == "Memory insufficient"
    assert error.error_code == ErrorCode.INSUFFICIENT_MEMORY
    assert error.details["resource"] == "gpu_memory"
    assert error.details["available"] == "4GB"
    assert error.details["required"] == "8GB"

# =============================================================================
# ERROR RESPONSE TESTS
# =============================================================================

def test_error_response():
    """Test ErrorResponse creation and serialization."""
    response = ErrorResponse(
        error_code=ErrorCode.INVALID_YOUTUBE_URL,
        message="Invalid YouTube URL",
        details={"field": "youtube_url"},
        timestamp=time.time(),
        request_id="test-123"
    )
    
    response_dict = response.to_dict()
    
    assert response_dict["error"]["code"] == 1001
    assert response_dict["error"]["message"] == "Invalid YouTube URL"
    assert response_dict["error"]["details"]["field"] == "youtube_url"
    assert response_dict["error"]["request_id"] == "test-123"

# =============================================================================
# ERROR HANDLER TESTS
# =============================================================================

def test_error_handler_validation():
    """Test ErrorHandler validation error handling."""
    handler = ErrorHandler()
    error = ValidationError("Invalid URL", "youtube_url", "invalid-url")
    
    with patch('structlog.get_logger') as mock_logger:
        response = handler.handle_validation_error(error, "test-123")
        
        assert response.error_code == ErrorCode.INVALID_YOUTUBE_URL
        assert response.message == "Invalid URL"
        assert response.request_id == "test-123"
        mock_logger.return_value.warning.assert_called_once()

def test_error_handler_processing():
    """Test ErrorHandler processing error handling."""
    handler = ErrorHandler()
    error = ProcessingError("Processing failed", "video_processing")
    
    with patch('structlog.get_logger') as mock_logger:
        response = handler.handle_processing_error(error, "test-123")
        
        assert response.error_code == ErrorCode.VIDEO_PROCESSING_FAILED
        assert response.message == "Processing failed"
        assert response.request_id == "test-123"
        mock_logger.return_value.error.assert_called_once()

def test_error_handler_unknown():
    """Test ErrorHandler unknown error handling."""
    handler = ErrorHandler()
    error = Exception("Unexpected error")
    
    with patch('structlog.get_logger') as mock_logger:
        response = handler.handle_unknown_error(error, "test-123")
        
        assert response.error_code == ErrorCode.UNKNOWN_ERROR
        assert "Unexpected error" in response.message
        assert response.request_id == "test-123"
        mock_logger.return_value.error.assert_called_once()

# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_validate_youtube_url_valid():
    """Test valid YouTube URL validation."""
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ"
    ]
    
    for url in valid_urls:
        validate_youtube_url(url)  # Should not raise

def test_validate_youtube_url_invalid():
    """Test invalid YouTube URL validation."""
    invalid_urls = [
        "",
        None,
        "not-a-url",
        "https://vimeo.com/123456",
        "https://youtube.com/invalid"
    ]
    
    for url in invalid_urls:
        with pytest.raises(ValidationError):
            validate_youtube_url(url)

def test_validate_language_code_valid():
    """Test valid language code validation."""
    valid_codes = ["en", "es", "fr", "en-US", "es-MX"]
    
    for code in valid_codes:
        validate_language_code(code)  # Should not raise

def test_validate_language_code_invalid():
    """Test invalid language code validation."""
    invalid_codes = ["", None, "invalid", "123", "EN"]
    
    for code in invalid_codes:
        with pytest.raises(ValidationError):
            validate_language_code(code)

def test_validate_clip_length_valid():
    """Test valid clip length validation."""
    validate_clip_length(30)  # Should not raise
    validate_clip_length(1, min_length=1, max_length=100)
    validate_clip_length(600, min_length=1, max_length=600)

def test_validate_clip_length_invalid():
    """Test invalid clip length validation."""
    with pytest.raises(ValidationError):
        validate_clip_length(0)
    
    with pytest.raises(ValidationError):
        validate_clip_length(601)
    
    with pytest.raises(ValidationError):
        validate_clip_length("30")  # Not an integer

def test_validate_viral_score_valid():
    """Test valid viral score validation."""
    validate_viral_score(0.0)  # Should not raise
    validate_viral_score(0.5)
    validate_viral_score(1.0)

def test_validate_viral_score_invalid():
    """Test invalid viral score validation."""
    with pytest.raises(ValidationError):
        validate_viral_score(-0.1)
    
    with pytest.raises(ValidationError):
        validate_viral_score(1.1)
    
    with pytest.raises(ValidationError):
        validate_viral_score("0.5")  # Not a number

def test_validate_caption_valid():
    """Test valid caption validation."""
    validate_caption("Valid caption")  # Should not raise
    validate_caption("A" * 1000)  # Max length

def test_validate_caption_invalid():
    """Test invalid caption validation."""
    with pytest.raises(ValidationError):
        validate_caption("")
    
    with pytest.raises(ValidationError):
        validate_caption("   ")  # Empty after strip
    
    with pytest.raises(ValidationError):
        validate_caption("A" * 1001)  # Too long
    
    with pytest.raises(ValidationError):
        validate_caption(123)  # Not a string

def test_validate_variant_id_valid():
    """Test valid variant ID validation."""
    validate_variant_id("var-1")  # Should not raise
    validate_variant_id("variant_123")
    validate_variant_id("test-var")

def test_validate_variant_id_invalid():
    """Test invalid variant ID validation."""
    with pytest.raises(ValidationError):
        validate_variant_id("")
    
    with pytest.raises(ValidationError):
        validate_variant_id("var@1")  # Invalid character
    
    with pytest.raises(ValidationError):
        validate_variant_id(123)  # Not a string

def test_validate_audience_profile_valid():
    """Test valid audience profile validation."""
    valid_profile = {
        "age": "18-24",
        "interests": ["technology", "gaming"]
    }
    validate_audience_profile(valid_profile)  # Should not raise

def test_validate_audience_profile_invalid():
    """Test invalid audience profile validation."""
    with pytest.raises(ValidationError):
        validate_audience_profile("not-a-dict")
    
    with pytest.raises(ValidationError):
        validate_audience_profile({"age": "18-24"})  # Missing interests
    
    with pytest.raises(ValidationError):
        validate_audience_profile({
            "age": "18-24",
            "interests": "not-a-list"
        })

def test_validate_batch_size_valid():
    """Test valid batch size validation."""
    validate_batch_size(10)  # Should not raise
    validate_batch_size(1, min_size=1, max_size=50)
    validate_batch_size(100, min_size=1, max_size=100)

def test_validate_batch_size_invalid():
    """Test invalid batch size validation."""
    with pytest.raises(ValidationError):
        validate_batch_size(0)
    
    with pytest.raises(ValidationError):
        validate_batch_size(101)
    
    with pytest.raises(ValidationError):
        validate_batch_size("10")  # Not an integer

# =============================================================================
# COMPOSITE VALIDATION TESTS
# =============================================================================

def test_validate_video_request_data_valid():
    """Test valid video request data validation."""
    valid_data = {
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "language": "en",
        "max_clip_length": 60,
        "min_clip_length": 10,
        "audience_profile": {
            "age": "18-24",
            "interests": ["technology"]
        }
    }
    validate_video_request_data(**valid_data)  # Should not raise

def test_validate_video_request_data_invalid():
    """Test invalid video request data validation."""
    with pytest.raises(ValidationError):
        validate_video_request_data(
            youtube_url="invalid-url",
            language="en"
        )

def test_validate_viral_variant_data_valid():
    """Test valid viral variant data validation."""
    valid_data = {
        "start": 0.0,
        "end": 10.0,
        "caption": "Test caption",
        "viral_score": 0.8,
        "variant_id": "var-1"
    }
    validate_viral_variant_data(**valid_data)  # Should not raise

def test_validate_viral_variant_data_invalid():
    """Test invalid viral variant data validation."""
    with pytest.raises(ValidationError):
        validate_viral_variant_data(
            start=10.0,
            end=5.0,  # End before start
            caption="Test caption",
            viral_score=0.8,
            variant_id="var-1"
        )

def test_validate_batch_request_data_valid():
    """Test valid batch request data validation."""
    valid_requests = [
        {
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en"
        }
    ]
    validate_batch_request_data(valid_requests, batch_size=1)  # Should not raise

def test_validate_batch_request_data_invalid():
    """Test invalid batch request data validation."""
    with pytest.raises(ValidationError):
        validate_batch_request_data([])  # Empty list
    
    with pytest.raises(ValidationError):
        validate_batch_request_data("not-a-list")

# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

def test_sanitize_youtube_url():
    """Test YouTube URL sanitization."""
    # Test HTTPS conversion
    assert sanitize_youtube_url("http://youtube.com/watch?v=123") == "https://youtube.com/watch?v=123"
    
    # Test whitespace removal
    assert sanitize_youtube_url("  https://youtube.com/watch?v=123  ") == "https://youtube.com/watch?v=123"
    
    # Test tracking parameter removal
    url_with_tracking = "https://youtube.com/watch?v=123&utm_source=test&fbclid=test"
    assert sanitize_youtube_url(url_with_tracking) == "https://youtube.com/watch?v=123"

def test_extract_youtube_video_id():
    """Test YouTube video ID extraction."""
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("invalid-url") is None

def test_validate_and_sanitize_url():
    """Test combined validation and sanitization."""
    # Valid URL should be sanitized
    result = validate_and_sanitize_url("https://www.youtube.com/watch?v=123&utm_source=test")
    assert result == "https://www.youtube.com/watch?v=123"
    
    # Invalid URL should raise error
    with pytest.raises(ValidationError):
        validate_and_sanitize_url("invalid-url")

# =============================================================================
# ERROR CREATION UTILITY TESTS
# =============================================================================

def test_create_validation_error():
    """Test validation error creation utility."""
    error = create_validation_error("Invalid URL", "youtube_url", "invalid-url", ErrorCode.INVALID_YOUTUBE_URL)
    
    assert error.message == "Invalid URL"
    assert error.error_code == ErrorCode.INVALID_YOUTUBE_URL
    assert error.details["field"] == "youtube_url"
    assert error.details["value"] == "invalid-url"

def test_create_processing_error():
    """Test processing error creation utility."""
    error = create_processing_error("Processing failed", "video_processing", ErrorCode.VIDEO_PROCESSING_FAILED)
    
    assert error.message == "Processing failed"
    assert error.error_code == ErrorCode.VIDEO_PROCESSING_FAILED
    assert error.details["operation"] == "video_processing"

def test_create_external_service_error():
    """Test external service error creation utility."""
    error = create_external_service_error("API timeout", "youtube_api", ErrorCode.YOUTUBE_API_ERROR)
    
    assert error.message == "API timeout"
    assert error.error_code == ErrorCode.YOUTUBE_API_ERROR
    assert error.details["service"] == "youtube_api"

def test_create_resource_error():
    """Test resource error creation utility."""
    error = create_resource_error("Memory insufficient", "gpu_memory", ErrorCode.INSUFFICIENT_MEMORY)
    
    assert error.message == "Memory insufficient"
    assert error.error_code == ErrorCode.INSUFFICIENT_MEMORY
    assert error.details["resource"] == "gpu_memory"

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_error_handling_integration():
    """Test complete error handling flow."""
    handler = ErrorHandler()
    
    # Create a validation error
    error = create_validation_error("Invalid URL", "youtube_url", "invalid-url")
    
    # Handle the error
    response = handler.handle_validation_error(error, "test-123")
    
    # Verify response
    assert response.error_code == ErrorCode.INVALID_YOUTUBE_URL
    assert response.message == "Invalid URL"
    assert response.request_id == "test-123"
    
    # Verify serialization
    response_dict = response.to_dict()
    assert response_dict["error"]["code"] == 1001
    assert response_dict["error"]["message"] == "Invalid URL"

if __name__ == "__main__":
    pytest.main([__file__]) 