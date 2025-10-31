"""
Test suite for error factories and custom error types.

Tests the error factory system, custom error types, error context management,
and integration with the existing error handling system.
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_factories import (
    ErrorCategory,
    ErrorContext,
    ErrorFactory,
    ErrorContextManager,
    VideoValidationError,
    VideoProcessingError,
    VideoEncodingError,
    VideoExtractionError,
    VideoAnalysisError,
    ModelInferenceError,
    PipelineError,
    ResourceExhaustionError,
    MemoryError,
    StorageError,
    NetworkError,
    APIError,
    DatabaseError,
    CacheError,
    ConfigurationError,
    SecurityViolationError,
    create_error_context,
    enrich_error_with_context,
    get_error_summary,
    with_error_context,
    handle_errors,
    error_factory,
    context_manager,
    create_validation_error,
    create_processing_error,
    create_encoding_error,
    create_inference_error,
    create_resource_error,
    create_api_error,
    create_security_error
)

from error_handling import (
    ErrorHandler,
    ValidationError,
    ProcessingError,
    ErrorCode
)

# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_ERROR_CONTEXT = ErrorContext(
    request_id="test-request-123",
    user_id="user123",
    session_id="session456",
    operation="video_processing",
    component="api",
    step="validation",
    resource_type="gpu_memory",
    resource_id="gpu-0",
    resource_usage={"memory_used": "2GB", "memory_total": "4GB"},
    start_time=datetime.utcnow(),
    duration=15.5,
    metadata={"test_key": "test_value"}
)

SAMPLE_VALIDATION_ERROR = VideoValidationError(
    "Invalid YouTube URL format",
    "youtube_url",
    "invalid_url",
    SAMPLE_ERROR_CONTEXT
)

SAMPLE_PROCESSING_ERROR = VideoProcessingError(
    "Video processing failed",
    "video_encoding",
    SAMPLE_ERROR_CONTEXT,
    {"video_id": "abc123", "error_type": "encoding_error"}
)

# =============================================================================
# ERROR CATEGORY TESTS
# =============================================================================

class TestErrorCategory:
    """Test error category enumeration."""
    
    def test_error_categories(self):
        """Test all error categories are defined."""
        categories = [
            ErrorCategory.VALIDATION,
            ErrorCategory.INPUT,
            ErrorCategory.FORMAT,
            ErrorCategory.PROCESSING,
            ErrorCategory.ENCODING,
            ErrorCategory.EXTRACTION,
            ErrorCategory.ANALYSIS,
            ErrorCategory.RESOURCE,
            ErrorCategory.MEMORY,
            ErrorCategory.STORAGE,
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL,
            ErrorCategory.API,
            ErrorCategory.DATABASE,
            ErrorCategory.CACHE,
            ErrorCategory.SYSTEM,
            ErrorCategory.CONFIGURATION,
            ErrorCategory.SECURITY,
            ErrorCategory.CRITICAL,
            ErrorCategory.MODEL,
            ErrorCategory.INFERENCE,
            ErrorCategory.TRAINING,
            ErrorCategory.PIPELINE
        ]
        
        for category in categories:
            assert category is not None
            assert isinstance(category.value, str)
            assert len(category.value) > 0
    
    def test_category_values(self):
        """Test category values are unique and meaningful."""
        values = [cat.value for cat in ErrorCategory]
        assert len(values) == len(set(values))  # All values are unique
        
        # Check for meaningful values
        assert "validation" in values
        assert "processing" in values
        assert "resource" in values
        assert "security" in values

# =============================================================================
# ERROR CONTEXT TESTS
# =============================================================================

class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_initialization(self):
        """Test error context initialization."""
        context = ErrorContext()
        
        assert context.request_id is None
        assert context.user_id is None
        assert context.session_id is None
        assert context.operation is None
        assert context.component is None
        assert context.step is None
        assert context.resource_type is None
        assert context.resource_id is None
        assert context.resource_usage is None
        assert context.start_time is None
        assert context.duration is None
        assert context.metadata == {}
    
    def test_error_context_with_data(self):
        """Test error context with data."""
        context = ErrorContext(
            request_id="test-123",
            user_id="user123",
            operation="video_processing",
            component="api",
            step="validation",
            metadata={"test": "value"}
        )
        
        assert context.request_id == "test-123"
        assert context.user_id == "user123"
        assert context.operation == "video_processing"
        assert context.component == "api"
        assert context.step == "validation"
        assert context.metadata["test"] == "value"
    
    def test_error_context_to_dict(self):
        """Test error context to dictionary conversion."""
        start_time = datetime.utcnow()
        context = ErrorContext(
            request_id="test-123",
            user_id="user123",
            operation="video_processing",
            start_time=start_time,
            duration=15.5,
            metadata={"test": "value"}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["request_id"] == "test-123"
        assert context_dict["user_id"] == "user123"
        assert context_dict["operation"] == "video_processing"
        assert context_dict["start_time"] == start_time.isoformat()
        assert context_dict["duration"] == 15.5
        assert context_dict["metadata"]["test"] == "value"
    
    def test_error_context_with_none_values(self):
        """Test error context with None values."""
        context = ErrorContext(
            request_id="test-123",
            user_id=None,
            session_id=None,
            operation="video_processing"
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["request_id"] == "test-123"
        assert context_dict["user_id"] is None
        assert context_dict["session_id"] is None
        assert context_dict["operation"] == "video_processing"

# =============================================================================
# CUSTOM ERROR TYPE TESTS
# =============================================================================

class TestCustomErrorTypes:
    """Test custom error types."""
    
    def test_video_validation_error(self):
        """Test VideoValidationError."""
        context = ErrorContext(request_id="test-123", operation="validation")
        error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)
        
        assert error.message == "Invalid URL"
        assert error.details["field"] == "youtube_url"
        assert error.details["value"] == "invalid_url"
        assert error.context == context
        assert error.category == ErrorCategory.VALIDATION
    
    def test_video_processing_error(self):
        """Test VideoProcessingError."""
        context = ErrorContext(request_id="test-123", operation="processing")
        error = VideoProcessingError("Processing failed", "video_encoding", context, {"video_id": "abc123"})
        
        assert error.message == "Processing failed"
        assert error.details["operation"] == "video_encoding"
        assert error.details["video_id"] == "abc123"
        assert error.context == context
        assert error.category == ErrorCategory.PROCESSING
    
    def test_video_encoding_error(self):
        """Test VideoEncodingError."""
        context = ErrorContext(request_id="test-123", operation="encoding")
        error = VideoEncodingError("Encoding failed", "video-123", context, {"codec": "h264"})
        
        assert error.message == "Encoding failed"
        assert error.video_id == "video-123"
        assert error.details["codec"] == "h264"
        assert error.context == context
        assert error.category == ErrorCategory.ENCODING
    
    def test_video_extraction_error(self):
        """Test VideoExtractionError."""
        context = ErrorContext(request_id="test-123", operation="extraction")
        error = VideoExtractionError("Extraction failed", "audio", context, {"format": "mp3"})
        
        assert error.message == "Extraction failed"
        assert error.extraction_type == "audio"
        assert error.details["format"] == "mp3"
        assert error.context == context
        assert error.category == ErrorCategory.EXTRACTION
    
    def test_model_inference_error(self):
        """Test ModelInferenceError."""
        context = ErrorContext(request_id="test-123", operation="inference")
        error = ModelInferenceError("Inference failed", "stable_diffusion", context, {"batch_size": 1})
        
        assert error.message == "Inference failed"
        assert error.model_name == "stable_diffusion"
        assert error.details["batch_size"] == 1
        assert error.context == context
        assert error.category == ErrorCategory.INFERENCE
    
    def test_pipeline_error(self):
        """Test PipelineError."""
        context = ErrorContext(request_id="test-123", operation="pipeline")
        error = PipelineError("Pipeline failed", "video_processing", "encoding", context, {"stage": "encoding"})
        
        assert error.message == "Pipeline failed"
        assert error.pipeline_name == "video_processing"
        assert error.stage == "encoding"
        assert error.details["stage"] == "encoding"
        assert error.context == context
        assert error.category == ErrorCategory.PIPELINE
    
    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError."""
        context = ErrorContext(request_id="test-123", operation="resource_check")
        error = ResourceExhaustionError("Memory exhausted", "gpu_memory", "2GB", "4GB", context)
        
        assert error.message == "Memory exhausted"
        assert error.details["resource"] == "gpu_memory"
        assert error.details["available"] == "2GB"
        assert error.details["required"] == "4GB"
        assert error.context == context
        assert error.category == ErrorCategory.RESOURCE
    
    def test_memory_error(self):
        """Test MemoryError."""
        context = ErrorContext(request_id="test-123", operation="memory_check")
        error = MemoryError("GPU memory full", "gpu", "2GB", "4GB", context)
        
        assert error.message == "GPU memory full"
        assert error.memory_type == "gpu"
        assert error.details["resource"] == "memory_gpu"
        assert error.details["available"] == "2GB"
        assert error.details["required"] == "4GB"
        assert error.context == context
        assert error.category == ErrorCategory.MEMORY
    
    def test_api_error(self):
        """Test APIError."""
        context = ErrorContext(request_id="test-123", operation="api_call")
        error = APIError("API call failed", "youtube", "/videos", 500, context)
        
        assert error.message == "API call failed"
        assert error.details["service"] == "youtube"
        assert error.endpoint == "/videos"
        assert error.details["status_code"] == 500
        assert error.context == context
        assert error.category == ErrorCategory.API
    
    def test_security_violation_error(self):
        """Test SecurityViolationError."""
        context = ErrorContext(request_id="test-123", operation="security_check")
        error = SecurityViolationError("Malicious input detected", "injection", "high", context)
        
        assert error.message == "Malicious input detected"
        assert error.details["threat_type"] == "injection"
        assert error.severity == "high"
        assert error.context == context
        assert error.category == ErrorCategory.SECURITY

# =============================================================================
# ERROR FACTORY TESTS
# =============================================================================

class TestErrorFactory:
    """Test error factory functionality."""
    
    def test_error_factory_initialization(self):
        """Test error factory initialization."""
        factory = ErrorFactory()
        
        assert factory.error_registry is not None
        assert len(factory.error_registry) > 0
        
        # Check that default errors are registered
        assert "validation" in factory.error_registry
        assert "processing" in factory.error_registry
        assert "encoding" in factory.error_registry
        assert "inference" in factory.error_registry
        assert "resource" in factory.error_registry
        assert "api" in factory.error_registry
        assert "security" in factory.error_registry
    
    def test_register_error(self):
        """Test registering custom error types."""
        factory = ErrorFactory()
        
        # Register a custom error
        factory.register_error("custom_error", VideoValidationError)
        
        assert "custom_error" in factory.error_registry
        assert factory.error_registry["custom_error"] == VideoValidationError
    
    def test_create_error(self):
        """Test creating errors with the factory."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        # Create validation error
        error = factory.create_error("validation", "Invalid input", field="test", value="invalid", context=context)
        
        assert isinstance(error, VideoValidationError)
        assert error.message == "Invalid input"
        assert error.details["field"] == "test"
        assert error.details["value"] == "invalid"
        assert error.context == context
    
    def test_create_error_unknown_type(self):
        """Test creating error with unknown type."""
        factory = ErrorFactory()
        
        with pytest.raises(ValueError, match="Unknown error type"):
            factory.create_error("unknown_type", "Test error")
    
    def test_create_validation_error(self):
        """Test creating validation error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_validation_error("youtube_url", "invalid_url", "Invalid URL format", context)
        
        assert isinstance(error, VideoValidationError)
        assert error.message == "Invalid URL format"
        assert error.details["field"] == "youtube_url"
        assert error.details["value"] == "invalid_url"
        assert error.context == context
    
    def test_create_processing_error(self):
        """Test creating processing error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_processing_error("video_encoding", "Encoding failed", context, {"video_id": "abc123"})
        
        assert isinstance(error, VideoProcessingError)
        assert error.message == "Encoding failed"
        assert error.details["operation"] == "video_encoding"
        assert error.details["video_id"] == "abc123"
        assert error.context == context
    
    def test_create_encoding_error(self):
        """Test creating encoding error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_encoding_error("Encoding failed", "video-123", context, {"codec": "h264"})
        
        assert isinstance(error, VideoEncodingError)
        assert error.message == "Encoding failed"
        assert error.video_id == "video-123"
        assert error.details["codec"] == "h264"
        assert error.context == context
    
    def test_create_inference_error(self):
        """Test creating inference error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_inference_error("Inference failed", "stable_diffusion", context, {"batch_size": 1})
        
        assert isinstance(error, ModelInferenceError)
        assert error.message == "Inference failed"
        assert error.model_name == "stable_diffusion"
        assert error.details["batch_size"] == 1
        assert error.context == context
    
    def test_create_resource_error(self):
        """Test creating resource error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_resource_error("Memory exhausted", "gpu_memory", "2GB", "4GB", context)
        
        assert isinstance(error, ResourceExhaustionError)
        assert error.message == "Memory exhausted"
        assert error.details["resource"] == "gpu_memory"
        assert error.details["available"] == "2GB"
        assert error.details["required"] == "4GB"
        assert error.context == context
    
    def test_create_api_error(self):
        """Test creating API error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_api_error("API call failed", "youtube", "/videos", 500, context)
        
        assert isinstance(error, APIError)
        assert error.message == "API call failed"
        assert error.details["service"] == "youtube"
        assert error.endpoint == "/videos"
        assert error.details["status_code"] == 500
        assert error.context == context
    
    def test_create_security_error(self):
        """Test creating security error."""
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        error = factory.create_security_error("Malicious input detected", "injection", "high", context)
        
        assert isinstance(error, SecurityViolationError)
        assert error.message == "Malicious input detected"
        assert error.details["threat_type"] == "injection"
        assert error.severity == "high"
        assert error.context == context

# =============================================================================
# ERROR CONTEXT MANAGER TESTS
# =============================================================================

class TestErrorContextManager:
    """Test error context manager functionality."""
    
    def test_context_manager_initialization(self):
        """Test context manager initialization."""
        manager = ErrorContextManager()
        
        assert manager.context is not None
        assert manager.context_stack == []
        assert manager.context.request_id is None
        assert manager.context.operation is None
    
    def test_set_request_context(self):
        """Test setting request context."""
        manager = ErrorContextManager()
        
        manager.set_request_context("test-123", "user123", "session456")
        
        assert manager.context.request_id == "test-123"
        assert manager.context.user_id == "user123"
        assert manager.context.session_id == "session456"
    
    def test_set_operation_context(self):
        """Test setting operation context."""
        manager = ErrorContextManager()
        
        manager.set_operation_context("video_processing", "api", "validation")
        
        assert manager.context.operation == "video_processing"
        assert manager.context.component == "api"
        assert manager.context.step == "validation"
    
    def test_set_resource_context(self):
        """Test setting resource context."""
        manager = ErrorContextManager()
        
        manager.set_resource_context("gpu_memory", "gpu-0", {"memory_used": "2GB"})
        
        assert manager.context.resource_type == "gpu_memory"
        assert manager.context.resource_id == "gpu-0"
        assert manager.context.resource_usage["memory_used"] == "2GB"
    
    def test_timing_functions(self):
        """Test timing functions."""
        manager = ErrorContextManager()
        
        # Start timing
        manager.start_timing()
        assert manager.context.start_time is not None
        assert isinstance(manager.context.start_time, datetime)
        
        # End timing
        manager.end_timing()
        assert manager.context.duration is not None
        assert manager.context.duration > 0
    
    def test_add_metadata(self):
        """Test adding metadata."""
        manager = ErrorContextManager()
        
        manager.add_metadata("test_key", "test_value")
        manager.add_metadata("number_key", 42)
        
        assert manager.context.metadata["test_key"] == "test_value"
        assert manager.context.metadata["number_key"] == 42
    
    def test_push_pop_context(self):
        """Test pushing and popping context."""
        manager = ErrorContextManager()
        
        # Set initial context
        manager.set_request_context("test-123", "user123")
        manager.set_operation_context("video_processing")
        
        # Push context
        manager.push_context()
        
        # Modify context
        manager.set_operation_context("image_processing")
        
        # Pop context
        manager.pop_context()
        
        # Should be back to original context
        assert manager.context.request_id == "test-123"
        assert manager.context.user_id == "user123"
        assert manager.context.operation == "video_processing"
    
    def test_get_context(self):
        """Test getting current context."""
        manager = ErrorContextManager()
        
        manager.set_request_context("test-123", "user123")
        manager.set_operation_context("video_processing")
        
        context = manager.get_context()
        
        assert context.request_id == "test-123"
        assert context.user_id == "user123"
        assert context.operation == "video_processing"
    
    def test_clear_context(self):
        """Test clearing context."""
        manager = ErrorContextManager()
        
        manager.set_request_context("test-123", "user123")
        manager.set_operation_context("video_processing")
        
        manager.clear_context()
        
        assert manager.context.request_id is None
        assert manager.context.user_id is None
        assert manager.context.operation is None

# =============================================================================
# ERROR UTILITIES TESTS
# =============================================================================

class TestErrorUtilities:
    """Test error utility functions."""
    
    def test_create_error_context(self):
        """Test creating error context."""
        context = create_error_context(
            request_id="test-123",
            user_id="user123",
            operation="video_processing",
            component="api",
            test_key="test_value"
        )
        
        assert context.request_id == "test-123"
        assert context.user_id == "user123"
        assert context.operation == "video_processing"
        assert context.component == "api"
        assert context.metadata["test_key"] == "test_value"
    
    def test_enrich_error_with_context(self):
        """Test enriching error with context."""
        error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url")
        context = ErrorContext(request_id="test-123", operation="validation")
        
        enrich_error_with_context(error, context)
        
        assert error.context == context
        assert error.context.request_id == "test-123"
        assert error.context.operation == "validation"
    
    def test_enrich_error_with_existing_context(self):
        """Test enriching error with existing context."""
        existing_context = ErrorContext(request_id="existing-123", operation="existing")
        error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", existing_context)
        
        new_context = ErrorContext(request_id="new-123", operation="new")
        enrich_error_with_context(error, new_context)
        
        # Should merge contexts
        assert error.context.request_id == "existing-123"  # Existing preserved
        assert error.context.operation == "new"  # New overrides
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        context = ErrorContext(request_id="test-123", operation="validation")
        error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)
        
        summary = get_error_summary(error)
        
        assert summary["error_type"] == "VideoValidationError"
        assert summary["error_message"] == "Invalid URL"
        assert summary["error_category"] == ErrorCategory.VALIDATION
        assert "timestamp" in summary
        assert "traceback" in summary
        assert "context" in summary
        assert summary["context"]["request_id"] == "test-123"
        assert summary["context"]["operation"] == "validation"
    
    def test_get_error_summary_without_context(self):
        """Test getting error summary without context."""
        error = ValueError("Test error")
        
        summary = get_error_summary(error)
        
        assert summary["error_type"] == "ValueError"
        assert summary["error_message"] == "Test error"
        assert "timestamp" in summary
        assert "traceback" in summary
        assert "context" not in summary

# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_validation_error_convenience(self):
        """Test create_validation_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_validation_error("youtube_url", "invalid_url", "Invalid URL format", context)
        
        assert isinstance(error, VideoValidationError)
        assert error.message == "Invalid URL format"
        assert error.details["field"] == "youtube_url"
        assert error.details["value"] == "invalid_url"
        assert error.context == context
    
    def test_create_processing_error_convenience(self):
        """Test create_processing_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_processing_error("video_encoding", "Encoding failed", context, {"video_id": "abc123"})
        
        assert isinstance(error, VideoProcessingError)
        assert error.message == "Encoding failed"
        assert error.details["operation"] == "video_encoding"
        assert error.details["video_id"] == "abc123"
        assert error.context == context
    
    def test_create_encoding_error_convenience(self):
        """Test create_encoding_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_encoding_error("Encoding failed", "video-123", context, {"codec": "h264"})
        
        assert isinstance(error, VideoEncodingError)
        assert error.message == "Encoding failed"
        assert error.video_id == "video-123"
        assert error.details["codec"] == "h264"
        assert error.context == context
    
    def test_create_inference_error_convenience(self):
        """Test create_inference_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_inference_error("Inference failed", "stable_diffusion", context, {"batch_size": 1})
        
        assert isinstance(error, ModelInferenceError)
        assert error.message == "Inference failed"
        assert error.model_name == "stable_diffusion"
        assert error.details["batch_size"] == 1
        assert error.context == context
    
    def test_create_resource_error_convenience(self):
        """Test create_resource_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_resource_error("Memory exhausted", "gpu_memory", "2GB", "4GB", context)
        
        assert isinstance(error, ResourceExhaustionError)
        assert error.message == "Memory exhausted"
        assert error.details["resource"] == "gpu_memory"
        assert error.details["available"] == "2GB"
        assert error.details["required"] == "4GB"
        assert error.context == context
    
    def test_create_api_error_convenience(self):
        """Test create_api_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_api_error("API call failed", "youtube", "/videos", 500, context)
        
        assert isinstance(error, APIError)
        assert error.message == "API call failed"
        assert error.details["service"] == "youtube"
        assert error.endpoint == "/videos"
        assert error.details["status_code"] == 500
        assert error.context == context
    
    def test_create_security_error_convenience(self):
        """Test create_security_error convenience function."""
        context = ErrorContext(request_id="test-123")
        error = create_security_error("Malicious input detected", "injection", "high", context)
        
        assert isinstance(error, SecurityViolationError)
        assert error.message == "Malicious input detected"
        assert error.details["threat_type"] == "injection"
        assert error.severity == "high"
        assert error.context == context

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestErrorFactoryIntegration:
    """Test integration of error factories with existing error handling."""
    
    def test_error_handler_with_custom_errors(self):
        """Test error handler with custom error types."""
        handler = ErrorHandler()
        context = ErrorContext(request_id="test-123", operation="video_processing")
        
        # Create custom error
        error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)
        
        # Handle custom error
        response = handler.handle_custom_error(error, "test-123")
        
        assert response.message is not None
        assert response.request_id == "test-123"
        assert "suggestion" in response.details
        assert "help_url" in response.details
        assert "error_category" in response.details
        assert response.details["error_category"] == "validation"
        assert "context" in response.details
    
    def test_error_factory_with_context_manager(self):
        """Test error factory with context manager."""
        factory = ErrorFactory()
        manager = ErrorContextManager()
        
        # Set up context
        manager.set_request_context("test-123", "user123")
        manager.set_operation_context("video_processing", "api")
        manager.start_timing()
        
        # Create error with context
        error = factory.create_validation_error("youtube_url", "invalid_url", "Invalid URL", manager.get_context())
        
        manager.end_timing()
        
        assert error.context.request_id == "test-123"
        assert error.context.user_id == "user123"
        assert error.context.operation == "video_processing"
        assert error.context.component == "api"
        assert error.context.duration is not None
        assert error.context.duration > 0
    
    def test_global_error_factory_instance(self):
        """Test global error factory instance."""
        assert error_factory is not None
        assert isinstance(error_factory, ErrorFactory)
        
        # Test that it has registered errors
        assert "validation" in error_factory.error_registry
        assert "processing" in error_factory.error_registry
        assert "encoding" in error_factory.error_registry
    
    def test_global_context_manager_instance(self):
        """Test global context manager instance."""
        assert context_manager is not None
        assert isinstance(context_manager, ErrorContextManager)
        
        # Test basic functionality
        context_manager.set_request_context("test-123")
        context_manager.set_operation_context("test_operation")
        
        context = context_manager.get_context()
        assert context.request_id == "test-123"
        assert context.operation == "test_operation"

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestErrorFactoryPerformance:
    """Test performance of error factory system."""
    
    def test_error_creation_performance(self):
        """Test performance of error creation."""
        import time
        
        factory = ErrorFactory()
        context = ErrorContext(request_id="test-123")
        
        start_time = time.perf_counter()
        
        for _ in range(1000):
            factory.create_validation_error("test_field", "test_value", "Test error", context)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should be very fast (less than 1 second for 1000 creations)
        assert duration < 1.0
    
    def test_context_manager_performance(self):
        """Test performance of context manager operations."""
        import time
        
        manager = ErrorContextManager()
        
        start_time = time.perf_counter()
        
        for _ in range(1000):
            manager.set_request_context("test-123")
            manager.set_operation_context("test_operation")
            manager.start_timing()
            manager.end_timing()
            context = manager.get_context()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should be very fast
        assert duration < 1.0

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestErrorFactoryEdgeCases:
    """Test edge cases in error factory system."""
    
    def test_error_with_none_context(self):
        """Test error creation with None context."""
        factory = ErrorFactory()
        
        error = factory.create_validation_error("test_field", "test_value", "Test error", None)
        
        assert error.context is not None
        assert error.context.request_id is None
        assert error.context.operation is None
    
    def test_error_with_empty_context(self):
        """Test error creation with empty context."""
        factory = ErrorFactory()
        context = ErrorContext()
        error = factory.create_validation_error("test_field", "test_value", "Test error", context)
        
        assert error.context == context
        assert error.context.request_id is None
        assert error.context.operation is None
    
    def test_context_manager_with_none_values(self):
        """Test context manager with None values."""
        manager = ErrorContextManager()
        
        manager.set_request_context("test-123", None, None)
        manager.set_operation_context("test_operation", None, None)
        
        context = manager.get_context()
        assert context.request_id == "test-123"
        assert context.user_id is None
        assert context.session_id is None
        assert context.operation == "test_operation"
        assert context.component is None
        assert context.step is None
    
    def test_error_summary_with_none_error(self):
        """Test error summary with None error."""
        summary = get_error_summary(None)
        
        assert summary["error_type"] == "NoneType"
        assert summary["error_message"] == "None"
        assert "timestamp" in summary
        assert "traceback" in summary

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 