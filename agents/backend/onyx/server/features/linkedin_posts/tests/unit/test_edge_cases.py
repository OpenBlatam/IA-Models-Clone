"""
Edge Cases and Boundary Tests for LinkedIn Posts
===============================================

Comprehensive edge case tests for LinkedIn posts including boundary conditions,
error scenarios, and unusual input handling.
"""

import pytest
import asyncio
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


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for edge case testing."""
        mock_repository = AsyncMock(spec=PostRepository)
        mock_ai_service = AsyncMock(spec=AIService)
        mock_cache_service = AsyncMock(spec=CacheService)
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.mark.asyncio
    async def test_create_post_with_empty_content(self, mock_services: PostService):
        """Test creating post with empty content."""
        request = PostGenerationRequest(
            topic="",
            keyPoints=[],
            targetAudience="",
            industry="",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=[],
            additionalContext=""
        )
        
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_extremely_long_content(self, mock_services: PostService):
        """Test creating post with extremely long content."""
        long_text = "A" * 10000  # Very long text
        request = PostGenerationRequest(
            topic=long_text,
            keyPoints=[long_text] * 10,
            targetAudience="Test",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext=long_text
        )
        
        # Should handle gracefully or throw appropriate error
        with pytest.raises(ValueError, match="Content too long"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_special_characters(self, mock_services: PostService):
        """Test creating post with special characters and emojis."""
        request = PostGenerationRequest(
            topic="AI & ML in 2024 🚀",
            keyPoints=["Increased efficiency ⚡", "Cost reduction 💰", "Innovation 🎯"],
            targetAudience="Business leaders 👔",
            industry="Technology 💻",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["AI", "ML", "🚀"],
            additionalContext="Focus on practical applications 📈"
        )
        
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI & ML in 2024 🚀",
                content=PostContent(
                    text="Generated content with emojis 🚀",
                    hashtags=["#AI", "#ML", "#🚀"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction="Learn more! 📈"
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
        assert "🚀" in result.title
        assert "🚀" in result.content.text

    @pytest.mark.asyncio
    async def test_create_post_with_null_values(self, mock_services: PostService):
        """Test creating post with null values."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=None,
            targetAudience=None,
            industry=None,
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=None,
            additionalContext=None
        )
        
        # Should handle null values gracefully
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
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
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request)
        assert result is not None
        assert result.title == "Test Topic"

    @pytest.mark.asyncio
    async def test_create_post_with_duplicate_keywords(self, mock_services: PostService):
        """Test creating post with duplicate keywords."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1", "Point 2"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["AI", "AI", "ML", "ML", "Technology"],
            additionalContext="Test context"
        )
        
        # Should deduplicate keywords
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#ML", "#Technology"],
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
                aiScore=85.0,
                keywords=["AI", "ML", "Technology"]  # Deduplicated
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request)
        assert result is not None
        assert len(result.keywords) == 3  # Deduplicated
        assert "AI" in result.keywords
        assert "ML" in result.keywords
        assert "Technology" in result.keywords

    @pytest.mark.asyncio
    async def test_create_post_with_invalid_tone(self, mock_services: PostService):
        """Test creating post with invalid tone."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone="INVALID_TONE",  # Invalid tone
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context"
        )
        
        with pytest.raises(ValueError, match="Invalid tone"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_invalid_post_type(self, mock_services: PostService):
        """Test creating post with invalid post type."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType="INVALID_TYPE",  # Invalid post type
            keywords=["test"],
            additionalContext="Test context"
        )
        
        with pytest.raises(ValueError, match="Invalid post type"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_very_short_content(self, mock_services: PostService):
        """Test creating post with very short content."""
        request = PostGenerationRequest(
            topic="A",  # Very short topic
            keyPoints=["A"],  # Very short key points
            targetAudience="A",
            industry="A",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["A"],
            additionalContext="A"
        )
        
        with pytest.raises(ValueError, match="Content too short"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_html_content(self, mock_services: PostService):
        """Test creating post with HTML content."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["<script>alert('test')</script>", "Point 2"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="<div>Test context</div>"
        )
        
        # Should sanitize HTML content
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content with sanitized HTML",
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
        assert "<script>" not in result.content.text
        assert "<div>" not in result.content.text

    @pytest.mark.asyncio
    async def test_create_post_with_sql_injection(self, mock_services: PostService):
        """Test creating post with SQL injection attempt."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["'; DROP TABLE posts; --"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="'; DROP TABLE users; --"
        )
        
        # Should handle SQL injection attempts safely
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content with sanitized input",
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
        assert "DROP TABLE" not in result.content.text

    @pytest.mark.asyncio
    async def test_create_post_with_xss_attempt(self, mock_services: PostService):
        """Test creating post with XSS attempt."""
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["<img src=x onerror=alert('XSS')>"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="<script>alert('XSS')</script>"
        )
        
        # Should sanitize XSS attempts
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="Test Topic",
                content=PostContent(
                    text="Generated content with sanitized XSS",
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
        assert "onerror=" not in result.content.text
        assert "<script>" not in result.content.text

    @pytest.mark.asyncio
    async def test_create_post_with_unicode_characters(self, mock_services: PostService):
        """Test creating post with Unicode characters."""
        request = PostGenerationRequest(
            topic="AI en Español: Inteligencia Artificial",
            keyPoints=["Eficiencia aumentada", "Reducción de costos", "Innovación"],
            targetAudience="Líderes empresariales",
            industry="Tecnología",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["IA", "inteligencia", "artificial"],
            additionalContext="Enfoque en aplicaciones prácticas"
        )
        
        # Should handle Unicode characters properly
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI en Español: Inteligencia Artificial",
                content=PostContent(
                    text="Contenido generado con caracteres Unicode",
                    hashtags=["#IA", "#inteligencia", "#artificial"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction="¡Aprende más!"
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
        assert "Español" in result.title
        assert "Unicode" in result.content.text

    @pytest.mark.asyncio
    async def test_create_post_with_very_large_keywords_list(self, mock_services: PostService):
        """Test creating post with very large keywords list."""
        large_keywords = [f"keyword_{i}" for i in range(1000)]  # 1000 keywords
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=large_keywords,
            additionalContext="Test context"
        )
        
        # Should handle large keyword lists gracefully
        with pytest.raises(ValueError, match="Too many keywords"):
            await mock_services.createPost(request)

    @pytest.mark.asyncio
    async def test_create_post_with_future_date(self, mock_services: PostService):
        """Test creating post with future date."""
        future_date = datetime.now() + timedelta(days=365)
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context",
            scheduledDate=future_date
        )
        
        # Should handle future dates properly
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
                status=PostStatus.SCHEDULED,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                scheduledDate=future_date,
                aiScore=85.0
            ),
            message="Post generated successfully"
        )
        
        result = await mock_services.createPost(request)
        assert result is not None
        assert result.status == PostStatus.SCHEDULED
        assert result.scheduledDate == future_date

    @pytest.mark.asyncio
    async def test_create_post_with_past_date(self, mock_services: PostService):
        """Test creating post with past date."""
        past_date = datetime.now() - timedelta(days=1)
        request = PostGenerationRequest(
            topic="Test Topic",
            keyPoints=["Point 1"],
            targetAudience="Test Audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["test"],
            additionalContext="Test context",
            scheduledDate=past_date
        )
        
        # Should reject past dates
        with pytest.raises(ValueError, match="Cannot schedule post in the past"):
            await mock_services.createPost(request)
