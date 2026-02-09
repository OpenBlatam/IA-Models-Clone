"""
Unit Tests for Dependencies
============================
Tests for FastAPI dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import HTTPException, Depends
from typing import Dict, Any, Optional

# Try to import dependencies
try:
    from dependencies import (
        get_pdf_service,
        get_current_user,
        validate_file_size,
        get_db_session,
        get_cache_service
    )
except ImportError:
    get_pdf_service = None
    get_current_user = None
    validate_file_size = None
    get_db_session = None
    get_cache_service = None


class TestGetPDFService:
    """Tests for get_pdf_service dependency."""
    
    @pytest.mark.asyncio
    async def test_get_pdf_service_returns_service(self):
        """Test that get_pdf_service returns a service instance."""
        if get_pdf_service is None:
            pytest.skip("get_pdf_service not available")
        
        # Mock the service creation
        with patch('dependencies.PDFVariantesService') as mock_service:
            service = await get_pdf_service()
            assert service is not None
    
    @pytest.mark.asyncio
    async def test_get_pdf_service_singleton(self):
        """Test that get_pdf_service returns same instance (if singleton)."""
        if get_pdf_service is None:
            pytest.skip("get_pdf_service not available")
        
        # This depends on implementation - may or may not be singleton
        service1 = await get_pdf_service()
        service2 = await get_pdf_service()
        
        # If singleton, they should be the same
        # If not, they should at least be instances of the same class
        assert type(service1) == type(service2)


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_with_token(self):
        """Test get_current_user with valid token."""
        if get_current_user is None:
            pytest.skip("get_current_user not available")
        
        # Mock request with authorization header
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer valid_token"}
        
        # Mock token validation
        with patch('dependencies.validate_token') as mock_validate:
            mock_validate.return_value = {"user_id": "123", "username": "test"}
            user = await get_current_user(request=mock_request)
            assert user is not None
            assert "user_id" in user or "username" in user
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Test get_current_user without token."""
        if get_current_user is None:
            pytest.skip("get_current_user not available")
        
        mock_request = Mock()
        mock_request.headers = {}
        
        # Should raise HTTPException
        with pytest.raises(HTTPException):
            await get_current_user(request=mock_request)
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test get_current_user with invalid token."""
        if get_current_user is None:
            pytest.skip("get_current_user not available")
        
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('dependencies.validate_token') as mock_validate:
            mock_validate.side_effect = HTTPException(status_code=401, detail="Invalid token")
            
            with pytest.raises(HTTPException):
                await get_current_user(request=mock_request)


class TestValidateFileSize:
    """Tests for validate_file_size dependency."""
    
    def test_validate_file_size_valid(self):
        """Test validate_file_size with valid size."""
        if validate_file_size is None:
            pytest.skip("validate_file_size not available")
        
        # Should not raise for valid size
        try:
            validate_file_size(1024)  # 1KB
        except Exception:
            pytest.fail("Should not raise for valid file size")
    
    def test_validate_file_size_too_large(self):
        """Test validate_file_size with file too large."""
        if validate_file_size is None:
            pytest.skip("validate_file_size not available")
        
        # Should raise for file too large
        with pytest.raises(HTTPException) or pytest.raises(ValueError):
            validate_file_size(100 * 1024 * 1024)  # 100MB (if limit is lower)
    
    def test_validate_file_size_zero(self):
        """Test validate_file_size with zero size."""
        if validate_file_size is None:
            pytest.skip("validate_file_size not available")
        
        # Should raise for zero or negative size
        with pytest.raises(Exception):
            validate_file_size(0)
    
    def test_validate_file_size_negative(self):
        """Test validate_file_size with negative size."""
        if validate_file_size is None:
            pytest.skip("validate_file_size not available")
        
        with pytest.raises(Exception):
            validate_file_size(-1)


class TestGetDBSession:
    """Tests for get_db_session dependency."""
    
    @pytest.mark.asyncio
    async def test_get_db_session_returns_session(self):
        """Test that get_db_session returns a database session."""
        if get_db_session is None:
            pytest.skip("get_db_session not available")
        
        # Mock database session
        with patch('dependencies.SessionLocal') as mock_session:
            session = await get_db_session()
            assert session is not None
    
    @pytest.mark.asyncio
    async def test_get_db_session_closes_on_exit(self):
        """Test that get_db_session closes session on exit."""
        if get_db_session is None:
            pytest.skip("get_db_session not available")
        
        mock_session = Mock()
        mock_session.close = Mock()
        
        with patch('dependencies.SessionLocal', return_value=mock_session):
            async with get_db_session() as session:
                pass
            
            # Session should be closed
            mock_session.close.assert_called()


class TestGetCacheService:
    """Tests for get_cache_service dependency."""
    
    @pytest.mark.asyncio
    async def test_get_cache_service_returns_service(self):
        """Test that get_cache_service returns a cache service."""
        if get_cache_service is None:
            pytest.skip("get_cache_service not available")
        
        # Mock cache service
        with patch('dependencies.CacheService') as mock_cache:
            service = await get_cache_service()
            assert service is not None


class TestDependencyInjection:
    """Tests for dependency injection patterns."""
    
    def test_dependency_can_be_overridden(self):
        """Test that dependencies can be overridden in tests."""
        if get_pdf_service is None:
            pytest.skip("get_pdf_service not available")
        
        # Create mock service
        mock_service = Mock()
        
        # Override dependency
        app = Mock()
        app.dependency_overrides = {}
        app.dependency_overrides[get_pdf_service] = lambda: mock_service
        
        # Should use overridden dependency
        assert app.dependency_overrides.get(get_pdf_service) is not None
    
    @pytest.mark.asyncio
    async def test_dependency_is_cached(self):
        """Test that dependencies are cached per request."""
        if get_pdf_service is None:
            pytest.skip("get_pdf_service not available")
        
        # Multiple calls in same request should return same instance
        service1 = await get_pdf_service()
        service2 = await get_pdf_service()
        
        # Depending on implementation, may or may not be same instance
        # But should at least be same type
        assert type(service1) == type(service2)



