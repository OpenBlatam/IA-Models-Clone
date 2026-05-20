"""
Tests for Error Handling
Tests for error recovery, graceful degradation, and error formatting
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from core.infrastructure.error_recovery import ErrorRecovery
from core.infrastructure.graceful_degradation import GracefulDegradation
from core.infrastructure.error_formatter import ErrorFormatter
from core.application.exceptions import (
    ValidationError,
    ProcessingError,
    NotFoundError
)


class TestErrorRecovery:
    """Tests for ErrorRecovery"""
    
    @pytest.fixture
    def error_recovery(self):
        """Create error recovery"""
        return ErrorRecovery(max_retries=3, retry_delay=0.1)
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, error_recovery):
        """Test retrying on failure"""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await error_recovery.retry(failing_operation)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, error_recovery):
        """Test max retries exceeded"""
        async def always_failing():
            raise Exception("Permanent failure")
        
        with pytest.raises(Exception):
            await error_recovery.retry(always_failing)
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, error_recovery):
        """Test retry with exponential backoff"""
        call_times = []
        
        async def failing_operation():
            call_times.append(asyncio.get_event_loop().time())
            raise Exception("Failure")
        
        try:
            await error_recovery.retry_with_backoff(failing_operation)
        except Exception:
            pass
        
        # Should have multiple calls with delays
        assert len(call_times) > 1


class TestGracefulDegradation:
    """Tests for GracefulDegradation"""
    
    @pytest.fixture
    def graceful_degradation(self):
        """Create graceful degradation"""
        return GracefulDegradation()
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, graceful_degradation):
        """Test fallback on failure"""
        async def primary_operation():
            raise Exception("Primary failed")
        
        async def fallback_operation():
            return "fallback_result"
        
        result = await graceful_degradation.execute_with_fallback(
            primary_operation,
            fallback_operation
        )
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_primary_success(self, graceful_degradation):
        """Test primary operation success"""
        async def primary_operation():
            return "primary_result"
        
        async def fallback_operation():
            return "fallback_result"
        
        result = await graceful_degradation.execute_with_fallback(
            primary_operation,
            fallback_operation
        )
        
        assert result == "primary_result"
    
    @pytest.mark.asyncio
    async def test_degraded_mode(self, graceful_degradation):
        """Test degraded mode operation"""
        async def operation():
            return {"data": "limited", "degraded": True}
        
        result = await graceful_degradation.execute_degraded(operation)
        
        assert result["degraded"] is True
        assert "data" in result


class TestErrorFormatter:
    """Tests for ErrorFormatter"""
    
    @pytest.fixture
    def error_formatter(self):
        """Create error formatter"""
        return ErrorFormatter()
    
    def test_format_validation_error(self, error_formatter):
        """Test formatting validation error"""
        error = ValidationError("Invalid input")
        
        formatted = error_formatter.format(error)
        
        assert "error" in formatted
        assert formatted["error"] == "validation_error"
        assert "message" in formatted
    
    def test_format_processing_error(self, error_formatter):
        """Test formatting processing error"""
        error = ProcessingError("Processing failed")
        
        formatted = error_formatter.format(error)
        
        assert "error" in formatted
        assert formatted["error"] == "processing_error"
    
    def test_format_not_found_error(self, error_formatter):
        """Test formatting not found error"""
        error = NotFoundError("Resource not found")
        
        formatted = error_formatter.format(error)
        
        assert "error" in formatted
        assert formatted["error"] == "not_found_error"
    
    def test_format_generic_exception(self, error_formatter):
        """Test formatting generic exception"""
        error = Exception("Generic error")
        
        formatted = error_formatter.format(error)
        
        assert "error" in formatted
        assert "message" in formatted
    
    def test_format_with_context(self, error_formatter):
        """Test formatting error with context"""
        error = ValidationError("Invalid input")
        context = {"user_id": "user-123", "operation": "analyze_image"}
        
        formatted = error_formatter.format(error, context=context)
        
        assert "context" in formatted or "user_id" in formatted
        assert formatted.get("message") is not None


class TestErrorHandlingIntegration:
    """Integration tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_graceful_degradation(self):
        """Test error recovery with graceful degradation"""
        from core.infrastructure.error_recovery import ErrorRecovery
        from core.infrastructure.graceful_degradation import GracefulDegradation
        
        recovery = ErrorRecovery(max_retries=2)
        degradation = GracefulDegradation()
        
        async def primary_operation():
            raise ConnectionError("Service unavailable")
        
        async def fallback_operation():
            return {"data": "cached", "degraded": True}
        
        # Try recovery first
        try:
            result = await recovery.retry(primary_operation)
        except Exception:
            # Then fallback
            result = await degradation.execute_with_fallback(
                primary_operation,
                fallback_operation
            )
        
        assert result["degraded"] is True
    
    @pytest.mark.asyncio
    async def test_error_formatting_in_api_response(self):
        """Test error formatting in API responses"""
        from core.infrastructure.error_formatter import ErrorFormatter
        
        formatter = ErrorFormatter()
        
        error = ValidationError("Invalid image format")
        formatted = formatter.format(error)
        
        # Should be suitable for API response
        assert isinstance(formatted, dict)
        assert "error" in formatted
        assert "message" in formatted



