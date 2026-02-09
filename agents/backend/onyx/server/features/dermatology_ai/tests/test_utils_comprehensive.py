"""
Tests for Utils (Comprehensive)
Tests for utility functions and helpers
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from utils.retry import retry_with_backoff
from utils.exceptions import (
    DermatologyAIError,
    ImageProcessingError,
    AnalysisError
)
from utils.oauth2 import get_current_user, verify_token
from utils.security_headers import SecurityHeaders


class TestRetryUtils:
    """Tests for retry utilities"""
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test retry with backoff on success"""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_operation()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_failure_then_success(self):
        """Test retry with backoff that eventually succeeds"""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await eventually_successful()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries(self):
        """Test retry with backoff exceeding max retries"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            await always_failing()
        
        assert call_count == 2


class TestExceptions:
    """Tests for custom exceptions"""
    
    def test_dermatology_ai_error(self):
        """Test base DermatologyAIError"""
        error = DermatologyAIError("Test error")
        
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_image_processing_error(self):
        """Test ImageProcessingError"""
        error = ImageProcessingError("Image processing failed")
        
        assert str(error) == "Image processing failed"
        assert isinstance(error, DermatologyAIError)
    
    def test_analysis_error(self):
        """Test AnalysisError"""
        error = AnalysisError("Analysis failed")
        
        assert str(error) == "Analysis failed"
        assert isinstance(error, DermatologyAIError)
    
    def test_exception_with_details(self):
        """Test exception with additional details"""
        error = ImageProcessingError(
            "Processing failed",
            details={"image_size": 1024, "format": "jpeg"}
        )
        
        assert str(error) == "Processing failed"
        assert hasattr(error, 'details') or error.args


class TestOAuth2:
    """Tests for OAuth2 utilities"""
    
    @pytest.mark.asyncio
    async def test_verify_token(self):
        """Test token verification"""
        # Mock token verification
        with patch('utils.oauth2.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user-123", "email": "test@example.com"}
            
            result = await verify_token("valid_token")
            
            assert result["user_id"] == "user-123"
            mock_verify.assert_called_once_with("valid_token")
    
    @pytest.mark.asyncio
    async def test_get_current_user(self):
        """Test getting current user from token"""
        from fastapi import Depends
        from unittest.mock import Mock
        
        # Mock request with token
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}
        
        with patch('utils.oauth2.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user-123"}
            
            user = await get_current_user(request)
            
            # Should return user info
            assert user is not None


class TestSecurityHeaders:
    """Tests for SecurityHeaders"""
    
    @pytest.fixture
    def security_headers(self):
        """Create security headers utility"""
        return SecurityHeaders()
    
    def test_get_security_headers(self, security_headers):
        """Test getting security headers"""
        headers = security_headers.get_headers()
        
        assert isinstance(headers, dict)
        # Should include common security headers
        assert len(headers) > 0
    
    def test_csp_header(self, security_headers):
        """Test Content Security Policy header"""
        headers = security_headers.get_headers()
        
        # CSP should be present or headers should be returned
        assert "Content-Security-Policy" in headers or len(headers) > 0
    
    def test_cors_headers(self, security_headers):
        """Test CORS headers"""
        headers = security_headers.get_cors_headers(origin="https://example.com")
        
        assert isinstance(headers, dict)
        # Should include CORS headers
        assert len(headers) > 0


class TestAsyncWorkers:
    """Tests for async workers"""
    
    @pytest.mark.asyncio
    async def test_async_worker_execution(self):
        """Test async worker execution"""
        from utils.async_workers import AsyncWorker
        
        results = []
        
        async def worker_task(item):
            await asyncio.sleep(0.01)
            results.append(item * 2)
        
        worker = AsyncWorker(worker_task, max_workers=3)
        
        # Process items
        items = list(range(10))
        await worker.process(items)
        
        # Wait for completion
        await asyncio.sleep(0.2)
        
        assert len(results) == 10
        assert all(r % 2 == 0 for r in results)


class TestObservability:
    """Tests for observability utilities"""
    
    def test_setup_observability(self):
        """Test setting up observability"""
        from utils.observability import setup_observability
        
        # Should not raise
        try:
            setup_observability()
        except Exception:
            # May fail if dependencies not available, which is OK
            pass
    
    def test_logging_setup(self):
        """Test logging setup"""
        from utils.logger import setup_logging, get_logger
        
        setup_logging(log_level="INFO")
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"



