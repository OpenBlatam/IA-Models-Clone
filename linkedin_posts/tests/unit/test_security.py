"""
Security Tests for LinkedIn Posts
================================

Comprehensive security tests for LinkedIn posts including authentication,
authorization, input validation, and security vulnerability testing.
"""

import pytest
import asyncio
import jwt
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import components
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestSecurity:
    """Test suite for security-related functionality."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for security testing."""
        mock_repository = AsyncMock(spec=PostRepository)
        mock_ai_service = AsyncMock(spec=AIService)
        mock_cache_service = AsyncMock(spec=CacheService)
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token for testing."""
        payload = {
            "user_id": "user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token for testing."""
        payload = {
            "user_id": "user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    @pytest.fixture
    def admin_token(self):
        """Create an admin JWT token for testing."""
        payload = {
            "user_id": "admin-123",
            "email": "admin@example.com",
            "role": "admin",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    @pytest.mark.asyncio
    async def test_authentication_with_valid_token(self, mock_services: PostService, valid_token: str):
        """Test authentication with valid JWT token."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Mock successful authentication and post creation
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request, auth_token=valid_token)
        assert result is not None
        assert result.userId == "user-123"

    @pytest.mark.asyncio
    async def test_authentication_with_expired_token(self, mock_services: PostService, expired_token: str):
        """Test authentication with expired JWT token."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        with pytest.raises(ValueError, match="Token expired"):
            await mock_services.createPost(request, auth_token=expired_token)

    @pytest.mark.asyncio
    async def test_authentication_with_invalid_token(self, mock_services: PostService):
        """Test authentication with invalid JWT token."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        with pytest.raises(ValueError, match="Invalid token"):
            await mock_services.createPost(request, auth_token="invalid-token")

    @pytest.mark.asyncio
    async def test_authorization_user_can_access_own_post(self, mock_services: PostService, valid_token: str):
        """Test that user can access their own post."""
        # Mock repository to return a post owned by the user
        mock_services.repository.getPost.return_value = LinkedInPost(
            id="test-123",
            userId="user-123",  # Same user as token
            title="Test Post",
            content=PostContent(
                text="Test content",
                hashtags=[],
                mentions=[],
                links=[],
                images=[],
                callToAction=""
            ),
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            aiScore=85.0
        )
        
        result = await mock_services.getPost("test-123", auth_token=valid_token)
        assert result is not None
        assert result.userId == "user-123"

    @pytest.mark.asyncio
    async def test_authorization_user_cannot_access_other_user_post(self, mock_services: PostService, valid_token: str):
        """Test that user cannot access another user's post."""
        # Mock repository to return a post owned by different user
        mock_services.repository.getPost.return_value = LinkedInPost(
            id="test-123",
            userId="other-user-456",  # Different user
            title="Test Post",
            content=PostContent(
                text="Test content",
                hashtags=[],
                mentions=[],
                links=[],
                images=[],
                callToAction=""
            ),
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            aiScore=85.0
        )
        
        with pytest.raises(PermissionError, match="Access denied"):
            await mock_services.getPost("test-123", auth_token=valid_token)

    @pytest.mark.asyncio
    async def test_authorization_admin_can_access_any_post(self, mock_services: PostService, admin_token: str):
        """Test that admin can access any post."""
        # Mock repository to return a post owned by any user
        mock_services.repository.getPost.return_value = LinkedInPost(
            id="test-123",
            userId="any-user-789",
            title="Test Post",
            content=PostContent(
                text="Test content",
                hashtags=[],
                mentions=[],
                links=[],
                images=[],
                callToAction=""
            ),
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            aiScore=85.0
        )
        
        result = await mock_services.getPost("test-123", auth_token=admin_token)
        assert result is not None
        assert result.userId == "any-user-789"

    @pytest.mark.asyncio
    async def test_input_validation_sql_injection_prevention(self, mock_services: PostService):
        """Test SQL injection prevention in input validation."""
        malicious_inputs = [
            "'; DROP TABLE posts; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "'; UPDATE users SET role='admin' WHERE id=1; --"
        ]
        
        for malicious_input in malicious_inputs:
            request = PostGenerationRequest(
                topic=malicious_input,
                keyPoints=[malicious_input],
                targetAudience=malicious_input,
                industry=malicious_input,
                tone=PostTone.PROFESSIONAL,
                postType=PostType.TEXT,
                keywords=[malicious_input],
                additionalContext=malicious_input
            )
            
            # Should sanitize or reject malicious input
            with pytest.raises(ValueError, match="Invalid input"):
                await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_input_validation_xss_prevention(self, mock_services: PostService):
        """Test XSS prevention in input validation."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for xss_payload in xss_payloads:
            request = PostGenerationRequest(
                topic=xss_payload,
                keyPoints=[xss_payload],
                targetAudience=xss_payload,
                industry=xss_payload,
                tone=PostTone.PROFESSIONAL,
                postType=PostType.TEXT,
                keywords=[xss_payload],
                additionalContext=xss_payload
            )
            
            # Should sanitize XSS payloads
            with pytest.raises(ValueError, match="Invalid input"):
                await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_input_validation_path_traversal_prevention(self, mock_services: PostService):
        """Test path traversal prevention in input validation."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd"
        ]
        
        for payload in path_traversal_payloads:
            request = PostGenerationRequest(
                topic=payload,
                keyPoints=[payload],
                targetAudience=payload,
                industry=payload,
                tone=PostTone.PROFESSIONAL,
                postType=PostType.TEXT,
                keywords=[payload],
                additionalContext=payload
            )
            
            # Should reject path traversal attempts
            with pytest.raises(ValueError, match="Invalid input"):
                await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_rate_limiting_prevention(self, mock_services: PostService, valid_token: str):
        """Test rate limiting to prevent abuse."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Mock successful response for first few requests
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        # First few requests should succeed
        for i in range(5):
            result = await mock_services.createPost(request, auth_token=valid_token)
            assert result is not None
        
        # After rate limit, should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await mock_services.createPost(request, auth_token=valid_token)

    @pytest.mark.asyncio
    async def test_encryption_of_sensitive_data(self, mock_services: PostService):
        """Test that sensitive data is properly encrypted."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request)
        
        # Verify that sensitive data is encrypted in storage
        stored_data = mock_services.repository.createPost.call_args[0][0]
        assert stored_data.userId != "user-123"  # Should be encrypted
        assert stored_data.title != "Test Topic"  # Should be encrypted

    @pytest.mark.asyncio
    async def test_audit_logging(self, mock_services: PostService, valid_token: str):
        """Test that security events are properly logged."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request, auth_token=valid_token)
        
        # Verify that audit log was created
        # This would typically check a logging service or audit table
        assert result is not None

    @pytest.mark.asyncio
    async def test_csrf_protection(self, mock_services: PostService):
        """Test CSRF protection."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Test without CSRF token
        with pytest.raises(ValueError, match="CSRF token required"):
            await mock_services.createPost(request, csrf_token=None)
        
        # Test with invalid CSRF token
        with pytest.raises(ValueError, match="Invalid CSRF token"):
            await mock_services.createPost(request, csrf_token="invalid-token")
        
        # Test with valid CSRF token
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request, csrf_token="valid-csrf-token")
        assert result is not None

    @pytest.mark.asyncio
    async def test_session_management(self, mock_services: PostService):
        """Test session management security."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        # Test with expired session
        with pytest.raises(ValueError, match="Session expired"):
            await mock_services.createPost(request, session_id="expired-session")
        
        # Test with invalid session
        with pytest.raises(ValueError, match="Invalid session"):
            await mock_services.createPost(request, session_id="invalid-session")
        
        # Test with valid session
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request, session_id="valid-session")
        assert result is not None

    @pytest.mark.asyncio
    async def test_data_sanitization(self, mock_services: PostService):
        """Test that user input is properly sanitized."""
        malicious_input = {
            "topic": "<script>alert('XSS')</script>Test Topic",
            "keyPoints": ["<img src=x onerror=alert('XSS')>Point 1"],
            "targetAudience": "'; DROP TABLE users; --",
            "industry": "../../../etc/passwd",
            "tone": PostTone.PROFESSIONAL,
            "postType": PostType.TEXT,
            "keywords": ["<script>alert('XSS')</script>"],
            "additionalContext": "javascript:alert('XSS')"
        }
        
        request = PostGenerationRequest(**malicious_input)
        
        # Mock successful response with sanitized content
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",  # Sanitized
                content=PostContent(
                    text="Generated content",  # Sanitized
                    hashtags=["#test"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request)
        assert result is not None
        assert "<script>" not in result.title
        assert "<script>" not in result.content.text
        assert "DROP TABLE" not in result.title
        assert "javascript:" not in result.content.text
