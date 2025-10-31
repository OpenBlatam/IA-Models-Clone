"""
BUL System - Practical Tests
Real, practical tests for the BUL system
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.practical_api import (
    DocumentRequest, DocumentResponse, UserAuth, APIStats,
    generate_document, get_current_user
)
from config.practical_config import PracticalConfig

class TestDocumentRequest:
    """Test document request model"""
    
    def test_document_request_creation(self):
        """Test creating a document request"""
        request = DocumentRequest(
            content="Test document content",
            template_type="business_letter",
            language="es",
            format="pdf"
        )
        
        assert request.content == "Test document content"
        assert request.template_type == "business_letter"
        assert request.language == "es"
        assert request.format == "pdf"
    
    def test_document_request_defaults(self):
        """Test document request with defaults"""
        request = DocumentRequest(
            content="Test content",
            template_type="email"
        )
        
        assert request.language == "es"
        assert request.format == "pdf"
        assert request.metadata is None

class TestDocumentResponse:
    """Test document response model"""
    
    def test_document_response_creation(self):
        """Test creating a document response"""
        response = DocumentResponse(
            document_id="doc_123",
            status="completed",
            content="Generated content",
            created_at=datetime.utcnow(),
            file_url="/documents/doc_123.pdf"
        )
        
        assert response.document_id == "doc_123"
        assert response.status == "completed"
        assert response.content == "Generated content"
        assert response.file_url == "/documents/doc_123.pdf"

class TestUserAuth:
    """Test user authentication model"""
    
    def test_user_auth_creation(self):
        """Test creating user authentication"""
        user = UserAuth(
            user_id="user_123",
            email="test@example.com",
            permissions=["read", "write"]
        )
        
        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.permissions == ["read", "write"]
    
    def test_user_auth_default_permissions(self):
        """Test user auth with default permissions"""
        user = UserAuth(
            user_id="user_456",
            email="test2@example.com"
        )
        
        assert user.permissions == []

class TestPracticalConfig:
    """Test practical configuration"""
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = PracticalConfig()
        
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.redis.host == "localhost"
        assert config.redis.port == 6379
        assert config.api.host == "0.0.0.0"
        assert config.api.port == 8000
    
    def test_database_url_generation(self):
        """Test database URL generation"""
        config = PracticalConfig()
        url = config.get_database_url()
        
        assert "postgresql://" in url
        assert "bul_user" in url
        assert "bul_password" in url
        assert "localhost:5432" in url
        assert "bul_db" in url
    
    def test_redis_url_generation(self):
        """Test Redis URL generation"""
        config = PracticalConfig()
        url = config.get_redis_url()
        
        assert "redis://" in url
        assert "localhost:6379" in url
        assert "/0" in url
    
    def test_ai_config(self):
        """Test AI configuration"""
        config = PracticalConfig()
        ai_config = config.get_ai_config()
        
        assert "api_key" in ai_config
        assert "model" in ai_config
        assert "max_tokens" in ai_config
        assert "temperature" in ai_config
        assert "timeout" in ai_config
    
    def test_security_config(self):
        """Test security configuration"""
        config = PracticalConfig()
        security_config = config.get_security_config()
        
        assert "secret_key" in security_config
        assert "algorithm" in security_config
        assert "access_token_expire_minutes" in security_config
        assert "refresh_token_expire_days" in security_config

class TestDocumentGeneration:
    """Test document generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_document_success(self):
        """Test successful document generation"""
        request = DocumentRequest(
            content="Test document",
            template_type="business_letter"
        )
        
        user = UserAuth(
            user_id="user_123",
            email="test@example.com"
        )
        
        response = await generate_document(request, user)
        
        assert response.document_id is not None
        assert response.status == "completed"
        assert response.content == "Test document"
        assert response.file_url is not None
        assert "processing_time" in response.metadata
    
    @pytest.mark.asyncio
    async def test_generate_document_with_metadata(self):
        """Test document generation with metadata"""
        request = DocumentRequest(
            content="Test document with metadata",
            template_type="email",
            language="en",
            format="html",
            metadata={"priority": "high", "category": "business"}
        )
        
        user = UserAuth(
            user_id="user_456",
            email="test2@example.com"
        )
        
        response = await generate_document(request, user)
        
        assert response.metadata["template_type"] == "email"
        assert response.metadata["language"] == "en"
        assert response.metadata["format"] == "html"
        assert response.metadata["user_id"] == "user_456"

class TestAuthentication:
    """Test authentication functionality"""
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token"""
        credentials = Mock()
        credentials.credentials = "valid_token_1234567890"
        
        user = await get_current_user(credentials)
        
        assert user.user_id == "user_123"
        assert user.email == "user@example.com"
        assert "read" in user.permissions
        assert "write" in user.permissions
        assert "generate_documents" in user.permissions
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test getting current user with invalid token"""
        credentials = Mock()
        credentials.credentials = "invalid"
        
        with pytest.raises(Exception):  # HTTPException
            await get_current_user(credentials)
    
    @pytest.mark.asyncio
    async def test_get_current_user_empty_token(self):
        """Test getting current user with empty token"""
        credentials = Mock()
        credentials.credentials = ""
        
        with pytest.raises(Exception):  # HTTPException
            await get_current_user(credentials)

class TestAPIStats:
    """Test API statistics"""
    
    def test_api_stats_creation(self):
        """Test creating API stats"""
        stats = APIStats(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_response_time=0.5,
            active_users=10
        )
        
        assert stats.total_requests == 100
        assert stats.successful_requests == 95
        assert stats.failed_requests == 5
        assert stats.average_response_time == 0.5
        assert stats.active_users == 10

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_document_workflow(self):
        """Test complete document generation workflow"""
        # Create request
        request = DocumentRequest(
            content="Integration test document",
            template_type="report",
            language="es",
            format="pdf"
        )
        
        # Create user
        user = UserAuth(
            user_id="integration_user",
            email="integration@test.com",
            permissions=["read", "write", "generate_documents"]
        )
        
        # Generate document
        response = await generate_document(request, user)
        
        # Verify response
        assert response.document_id is not None
        assert response.status == "completed"
        assert response.content == "Integration test document"
        assert response.file_url.endswith(".pdf")
        assert response.metadata["user_id"] == "integration_user"
        assert response.metadata["template_type"] == "report"
        assert response.metadata["language"] == "es"
        assert response.metadata["format"] == "pdf"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])













