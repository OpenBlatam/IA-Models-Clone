"""
Test Suite for Prioritized Error Handling

Tests critical error scenarios, edge cases, and system health monitoring.
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
    ErrorResponse,
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
    check_system_resources,
    check_gpu_availability,
    validate_system_health,
    validate_gpu_health
)

# =============================================================================
# CRITICAL ERROR TESTS
# =============================================================================

class TestCriticalSystemErrors:
    """Test critical system error handling."""
    
    def test_critical_system_error_creation(self):
        """Test creation of critical system errors."""
        error = create_critical_system_error(
            "Database connection lost",
            "database",
            ErrorCode.DATABASE_CONNECTION_LOST
        )
        
        assert error.message == "Database connection lost"
        assert error.error_code == ErrorCode.DATABASE_CONNECTION_LOST
        assert error.details["component"] == "database"
    
    def test_critical_system_error_handler(self):
        """Test critical system error handler with alerting."""
        handler = ErrorHandler()
        error = create_critical_system_error(
            "GPU memory exhausted",
            "gpu",
            ErrorCode.GPU_MEMORY_EXHAUSTED
        )
        
        with patch.object(handler, '_send_critical_alert') as mock_alert:
            response = handler.handle_critical_system_error(error, "test-request-id")
            
            assert response.error_code == ErrorCode.GPU_MEMORY_EXHAUSTED
            assert "GPU memory exhausted" in response.message
            assert response.request_id == "test-request-id"
            
            # Should not alert on first error
            mock_alert.assert_not_called()
    
    def test_critical_error_threshold_alerting(self):
        """Test that critical errors trigger alerts after threshold."""
        handler = ErrorHandler()
        error = create_critical_system_error(
            "System crash",
            "system",
            ErrorCode.SYSTEM_CRASH
        )
        
        with patch.object(handler, '_send_critical_alert') as mock_alert:
            # Trigger multiple critical errors
            for i in range(6):  # Threshold is 5
                handler.handle_critical_system_error(error, f"request-{i}")
            
            # Should have called alert once after threshold
            assert mock_alert.call_count == 1

# =============================================================================
# SECURITY ERROR TESTS
# =============================================================================

class TestSecurityErrors:
    """Test security error handling."""
    
    def test_security_error_creation(self):
        """Test creation of security errors."""
        error = create_security_error(
            "Malicious input detected",
            "injection",
            ErrorCode.INJECTION_ATTEMPT
        )
        
        assert error.message == "Malicious input detected"
        assert error.error_code == ErrorCode.INJECTION_ATTEMPT
        assert error.details["threat_type"] == "injection"
    
    def test_security_error_handler(self):
        """Test security error handler with threat response."""
        handler = ErrorHandler()
        error = create_security_error(
            "SQL injection attempt",
            "injection",
            ErrorCode.INJECTION_ATTEMPT
        )
        
        with patch.object(handler, '_block_suspicious_ip') as mock_block:
            response = handler.handle_security_error(error, "test-request-id")
            
            assert response.error_code == ErrorCode.INJECTION_ATTEMPT
            assert "Security violation detected" in response.message
            assert response.details["threat_type"] == "injection"
            
            # Should block suspicious IP for injection attempts
            mock_block.assert_called_once_with("test-request-id")
    
    def test_malicious_url_validation(self):
        """Test validation against malicious URL patterns."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "eval('malicious_code')",
            "https://youtube.com/watch?v=123&script=eval('xss')"
        ]
        
        for url in malicious_urls:
            with pytest.raises(ValidationError) as exc_info:
                validate_youtube_url(url)
            
            assert "Malicious URL pattern detected" in str(exc_info.value)

# =============================================================================
# EDGE CASE VALIDATION TESTS
# =============================================================================

class TestEdgeCaseValidation:
    """Test edge case validation scenarios."""
    
    def test_extremely_long_url(self):
        """Test validation of extremely long URLs."""
        long_url = "https://youtube.com/watch?v=" + "a" * 2000
        
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(long_url)
        
        assert "too long" in str(exc_info.value)
    
    def test_negative_clip_length(self):
        """Test validation of negative clip lengths."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(-5)
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_zero_clip_length(self):
        """Test validation of zero clip length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(0)
        
        assert "cannot be zero" in str(exc_info.value)
    
    def test_extremely_long_clip(self):
        """Test validation of extremely long clips."""
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(100000)  # Over 24 hours
        
        assert "exceeds maximum allowed duration" in str(exc_info.value)
    
    def test_negative_batch_size(self):
        """Test validation of negative batch sizes."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(-10)
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_zero_batch_size(self):
        """Test validation of zero batch size."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(0)
        
        assert "cannot be zero" in str(exc_info.value)
    
    def test_extremely_large_batch(self):
        """Test validation of extremely large batch sizes."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(2000)  # Over limit
        
        assert "exceeds maximum allowed limit" in str(exc_info.value)

# =============================================================================
# SYSTEM HEALTH MONITORING TESTS
# =============================================================================

class TestSystemHealthMonitoring:
    """Test system health monitoring functionality."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_health_check(self, mock_disk, mock_memory, mock_cpu):
        """Test system health checking."""
        # Mock system resources
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(percent=85.0)
        mock_disk.return_value = Mock(percent=80.0)
        
        health_status = check_system_resources()
        
        assert health_status["cpu_usage"] == 75.0
        assert health_status["memory_usage"] == 85.0
        assert health_status["disk_usage"] == 80.0
        assert "network_status" in health_status
    
    @patch('psutil.virtual_memory')
    def test_critical_memory_usage(self, mock_memory):
        """Test detection of critical memory usage."""
        mock_memory.return_value = Mock(percent=95.0)
        
        with pytest.raises(CriticalSystemError) as exc_info:
            validate_system_health()
        
        assert "Critical memory usage detected" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.GPU_MEMORY_EXHAUSTED
    
    @patch('psutil.disk_usage')
    def test_critical_disk_usage(self, mock_disk):
        """Test detection of critical disk usage."""
        mock_disk.return_value = Mock(percent=98.0)
        
        with pytest.raises(CriticalSystemError) as exc_info:
            validate_system_health()
        
        assert "Critical disk space detected" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.DISK_SPACE_CRITICAL
    
    @patch('torch.cuda.is_available')
    def test_gpu_availability_check(self, mock_cuda_available):
        """Test GPU availability checking."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.get_device_name', return_value="RTX 4090"), \
             patch('torch.cuda.memory_allocated', return_value=2*1024**3), \
             patch('torch.cuda.memory_reserved', return_value=3*1024**3), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value = Mock(total_memory=8*1024**3)
            
            gpu_status = check_gpu_availability()
            
            assert gpu_status["available"] is True
            assert gpu_status["device_count"] == 1
            assert gpu_status["device_name"] == "RTX 4090"
            assert gpu_status["memory_allocated"] == 2.0  # GB
            assert gpu_status["memory_total"] == 8.0  # GB
    
    @patch('torch.cuda.is_available')
    def test_gpu_not_available(self, mock_cuda_available):
        """Test handling when GPU is not available."""
        mock_cuda_available.return_value = False
        
        with pytest.raises(ResourceError) as exc_info:
            validate_gpu_health()
        
        assert "GPU not available for processing" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCode.GPU_NOT_AVAILABLE

# =============================================================================
# ERROR PATTERN ANALYSIS TESTS
# =============================================================================

class TestErrorPatternAnalysis:
    """Test error pattern analysis functionality."""
    
    def test_memory_related_error_pattern(self):
        """Test detection of memory-related error patterns."""
        handler = ErrorHandler()
        
        error = Exception("Out of memory error occurred")
        pattern = handler._analyze_error_pattern(error)
        
        assert pattern == "memory_related"
    
    def test_timeout_related_error_pattern(self):
        """Test detection of timeout-related error patterns."""
        handler = ErrorHandler()
        
        error = Exception("Request timed out after 30 seconds")
        pattern = handler._analyze_error_pattern(error)
        
        assert pattern == "timeout_related"
    
    def test_network_related_error_pattern(self):
        """Test detection of network-related error patterns."""
        handler = ErrorHandler()
        
        error = Exception("Connection failed to external service")
        pattern = handler._analyze_error_pattern(error)
        
        assert pattern == "network_related"
    
    def test_permission_related_error_pattern(self):
        """Test detection of permission-related error patterns."""
        handler = ErrorHandler()
        
        error = Exception("Permission denied accessing file")
        pattern = handler._analyze_error_pattern(error)
        
        assert pattern == "permission_related"
    
    def test_unknown_error_pattern(self):
        """Test handling of unknown error patterns."""
        handler = ErrorHandler()
        
        error = Exception("Some random error message")
        pattern = handler._analyze_error_pattern(error)
        
        assert pattern == "unknown_pattern"

# =============================================================================
# CONFIGURATION ERROR TESTS
# =============================================================================

class TestConfigurationErrors:
    """Test configuration error handling."""
    
    def test_configuration_error_creation(self):
        """Test creation of configuration errors."""
        error = create_configuration_error(
            "Missing API key",
            "api_key",
            ErrorCode.API_KEY_MISSING
        )
        
        assert error.message == "Missing API key"
        assert error.error_code == ErrorCode.API_KEY_MISSING
        assert error.details["config_key"] == "api_key"
    
    def test_configuration_error_handler_with_fallback(self):
        """Test configuration error handler with fallback strategies."""
        handler = ErrorHandler()
        error = create_configuration_error(
            "Invalid model path",
            "model_path",
            ErrorCode.MODEL_CONFIG_INVALID
        )
        
        response = handler.handle_configuration_error(error, "test-request-id")
        
        assert response.error_code == ErrorCode.MODEL_CONFIG_INVALID
        assert "Invalid model path" in response.message
        assert response.details["config_key"] == "model_path"
    
    def test_fallback_config_retrieval(self):
        """Test fallback configuration retrieval."""
        handler = ErrorHandler()
        
        fallback = handler._get_fallback_config("model_path")
        assert fallback == "/default/models/"
        
        fallback = handler._get_fallback_config("api_key")
        assert fallback == "default_key"
        
        fallback = handler._get_fallback_config("nonexistent_key")
        assert fallback is None

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestErrorHandlingIntegration:
    """Test integration of error handling components."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization with thresholds."""
        handler = ErrorHandler()
        
        assert handler.critical_error_count == 0
        assert handler.error_thresholds["critical"] == 5
        assert handler.error_thresholds["high"] == 20
        assert handler.error_thresholds["medium"] == 50
    
    def test_error_response_formatting(self):
        """Test error response formatting."""
        response = ErrorResponse(
            error_code=ErrorCode.VIDEO_PROCESSING_FAILED,
            message="Processing failed",
            details={"operation": "video_encoding"},
            timestamp=time.time(),
            request_id="test-123"
        )
        
        response_dict = response.to_dict()
        
        assert "error" in response_dict
        assert response_dict["error"]["code"] == ErrorCode.VIDEO_PROCESSING_FAILED.value
        assert response_dict["error"]["message"] == "Processing failed"
        assert response_dict["error"]["request_id"] == "test-123"
    
    def test_error_code_prioritization(self):
        """Test error code prioritization structure."""
        # Critical errors (1000-1999)
        assert ErrorCode.SYSTEM_CRASH.value == 1001
        assert ErrorCode.GPU_MEMORY_EXHAUSTED.value == 1004
        
        # High priority errors (2000-2999)
        assert ErrorCode.VIDEO_PROCESSING_FAILED.value == 2001
        assert ErrorCode.MODEL_INFERENCE_FAILED.value == 2007
        
        # Medium priority errors (4000-5999)
        assert ErrorCode.INSUFFICIENT_MEMORY.value == 4001
        assert ErrorCode.MISSING_CONFIG.value == 5001
        
        # Low priority errors (6000-7999)
        assert ErrorCode.INVALID_YOUTUBE_URL.value == 6001
        assert ErrorCode.UNAUTHORIZED_ACCESS.value == 7001

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestErrorHandlingPerformance:
    """Test error handling performance under load."""
    
    def test_error_handler_performance(self):
        """Test error handler performance with multiple errors."""
        handler = ErrorHandler()
        start_time = time.time()
        
        # Process multiple errors
        for i in range(100):
            error = create_validation_error(
                f"Test error {i}",
                "test_field",
                i,
                ErrorCode.INVALID_YOUTUBE_URL
            )
            handler.handle_validation_error(error, f"request-{i}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 errors in under 1 second
        assert processing_time < 1.0
    
    def test_system_health_check_performance(self):
        """Test system health check performance."""
        start_time = time.time()
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=60.0)), \
             patch('psutil.disk_usage', return_value=Mock(percent=70.0)):
            
            health_status = check_system_resources()
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete health check in under 0.1 seconds
        assert processing_time < 0.1
        assert "cpu_usage" in health_status
        assert "memory_usage" in health_status
        assert "disk_usage" in health_status

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 