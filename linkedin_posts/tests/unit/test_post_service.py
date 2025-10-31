"""
Unit Tests for PostService
==========================

Comprehensive unit tests for the LinkedIn Post Service with proper mocking,
edge case handling, and error scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import the service and entities
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestPostService:
    """Test suite for PostService class."""

    @pytest.fixture
    def mock_repository(self) -> PostRepository:
        """Mock repository for testing."""
        mock = AsyncMock(spec=PostRepository)
        mock.createPost = AsyncMock()
        mock.updatePost = AsyncMock()
        mock.getPost = AsyncMock()
        mock.listPosts = AsyncMock()
        mock.deletePost = AsyncMock()
        return mock

    @pytest.fixture
    def mock_ai_service(self) -> AIService:
        """Mock AI service for testing."""
        mock = AsyncMock(spec=AIService)
        mock.analyzeContent = AsyncMock()
        mock.generatePost = AsyncMock()
        mock.optimizePost = AsyncMock()
        return mock

    @pytest.fixture
    def mock_cache_service(self) -> CacheService:
        """Mock cache service for testing."""
        mock = AsyncMock(spec=CacheService)
        mock.get = AsyncMock()
        mock.set = AsyncMock()
        mock.delete = AsyncMock()
        mock.clear = AsyncMock()
        return mock

    @pytest.fixture
    def post_service(
        self, 
        mock_repository: PostRepository,
        mock_ai_service: AIService,
        mock_cache_service: CacheService
    ) -> PostService:
        """Create PostService instance with mocked dependencies."""
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.fixture
    def sample_post_request(self) -> PostGenerationRequest:
        """Sample post generation request."""
        return PostGenerationRequest(
            topic="AI in Modern Business",
            keyPoints=["Increased efficiency", "Cost reduction", "Innovation"],
            targetAudience="Business leaders",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["AI", "business", "innovation"],
            additionalContext="Focus on practical applications"
        )

    @pytest.fixture
    def sample_post(self) -> LinkedInPost:
        """Sample LinkedIn post for testing."""
        return LinkedInPost(
            id="test-post-123",
            userId="user-123",
            title="AI in Modern Business",
            content=PostContent(
                text="AI is transforming modern business practices...",
                hashtags=["#AI", "#Business", "#Innovation"],
                mentions=[],
                links=[],
                images=[],
                callToAction="Learn more about AI implementation"
            ),
            postType=PostType.TEXT,
            tone=PostTone.PROFESSIONAL,
            status=PostStatus.DRAFT,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            engagement=EngagementMetrics(
                likes=0,
                comments=0,
                shares=0,
                clicks=0,
                impressions=0,
                reach=0,
                engagementRate=0.0
            ),
            aiScore=85.5,
            optimizationSuggestions=["Add more hashtags", "Include call-to-action"],
            keywords=["AI", "business", "innovation"],
            externalMetadata={},
            performanceScore=0,
            reachScore=0,
            engagementScore=0
        )

    @pytest.mark.asyncio
    async def test_create_post_success(self, post_service: PostService, sample_post_request: PostGenerationRequest, sample_post: LinkedInPost):
        """Test successful post creation."""
        # Arrange
        mock_ai_response = PostGenerationResponse(
            post=sample_post,
            generationTime=1.5,
            aiScore=85.5,
            optimizationSuggestions=["Add more hashtags"]
        )
        
        post_service.ai_service.generatePost.return_value = mock_ai_response
        post_service.postRepository.createPost.return_value = sample_post
        post_service.cacheService.get.return_value = None  # No cache hit
        
        # Act
        result = await post_service.createPost(sample_post_request)
        
        # Assert
        assert result.id == sample_post.id
        assert result.title == sample_post.title
        assert result.status == PostStatus.DRAFT
        post_service.ai_service.generatePost.assert_called_once_with(sample_post_request)
        post_service.postRepository.createPost.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_post_cache_hit(self, post_service: PostService, sample_post_request: PostGenerationRequest, sample_post: LinkedInPost):
        """Test post creation with cache hit."""
        # Arrange
        post_service.cacheService.get.return_value = sample_post
        
        # Act
        result = await post_service.createPost(sample_post_request)
        
        # Assert
        assert result == sample_post
        post_service.ai_service.generatePost.assert_not_called()
        post_service.postRepository.createPost.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_post_invalid_request(self, post_service: PostService, sample_post_request: PostGenerationRequest):
        """Test post creation with invalid request."""
        # Arrange
        validation_result = PostValidationResult(
            isValid=False,
            errors=["Topic is too short", "Missing target audience"],
            warnings=[],
            suggestions=[]
        )
        
        with patch.object(post_service, 'validatePostRequest', return_value=validation_result):
            # Act & Assert
            with pytest.raises(ValueError, match="Invalid post request"):
                await post_service.createPost(sample_post_request)

    @pytest.mark.asyncio
    async def test_update_post_success(self, post_service: PostService, sample_post: LinkedInPost):
        """Test successful post update."""
        # Arrange
        post_id = "test-post-123"
        updates = {"title": "Updated Title", "status": PostStatus.SCHEDULED}
        updated_post = LinkedInPost(**{**sample_post.__dict__, **updates})
        
        post_service.postRepository.updatePost.return_value = updated_post
        post_service.cacheService.delete.return_value = None
        
        # Act
        result = await post_service.updatePost(post_id, updates)
        
        # Assert
        assert result.title == "Updated Title"
        assert result.status == PostStatus.SCHEDULED
        post_service.postRepository.updatePost.assert_called_once_with(post_id, updates)
        post_service.cacheService.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_post_success(self, post_service: PostService, sample_post: LinkedInPost):
        """Test successful post retrieval."""
        # Arrange
        post_id = "test-post-123"
        post_service.postRepository.getPost.return_value = sample_post
        
        # Act
        result = await post_service.getPost(post_id)
        
        # Assert
        assert result == sample_post
        post_service.postRepository.getPost.assert_called_once_with(post_id)

    @pytest.mark.asyncio
    async def test_get_post_not_found(self, post_service: PostService):
        """Test post retrieval when post doesn't exist."""
        # Arrange
        post_id = "non-existent-post"
        post_service.postRepository.getPost.return_value = None
        
        # Act
        result = await post_service.getPost(post_id)
        
        # Assert
        assert result is None
        post_service.postRepository.getPost.assert_called_once_with(post_id)

    @pytest.mark.asyncio
    async def test_list_posts_success(self, post_service: PostService, sample_post: LinkedInPost):
        """Test successful post listing."""
        # Arrange
        user_id = "user-123"
        filters = {"status": PostStatus.DRAFT, "limit": 10}
        posts = [sample_post]
        
        post_service.postRepository.listPosts.return_value = posts
        
        # Act
        result = await post_service.listPosts(user_id, filters)
        
        # Assert
        assert result == posts
        assert len(result) == 1
        post_service.postRepository.listPosts.assert_called_once_with(user_id, filters)

    @pytest.mark.asyncio
    async def test_delete_post_success(self, post_service: PostService):
        """Test successful post deletion."""
        # Arrange
        post_id = "test-post-123"
        post_service.postRepository.deletePost.return_value = True
        post_service.cacheService.delete.return_value = None
        
        # Act
        result = await post_service.deletePost(post_id)
        
        # Assert
        assert result is True
        post_service.postRepository.deletePost.assert_called_once_with(post_id)
        post_service.cacheService.delete.assert_called()

    @pytest.mark.asyncio
    async def test_optimize_post_success(self, post_service: PostService, sample_post: LinkedInPost):
        """Test successful post optimization."""
        # Arrange
        post_id = "test-post-123"
        optimization_result = PostOptimizationResult(
            originalPost=sample_post,
            optimizedPost=sample_post,
            optimizationScore=90.0,
            suggestions=["Improve headline", "Add more hashtags"],
            processingTime=2.5
        )
        
        post_service.postRepository.getPost.return_value = sample_post
        post_service.ai_service.optimizePost.return_value = optimization_result
        post_service.postRepository.updatePost.return_value = sample_post
        
        # Act
        result = await post_service.optimizePost(post_id)
        
        # Assert
        assert result == optimization_result
        post_service.ai_service.optimizePost.assert_called_once_with(sample_post)

    @pytest.mark.asyncio
    async def test_optimize_post_not_found(self, post_service: PostService):
        """Test post optimization when post doesn't exist."""
        # Arrange
        post_id = "non-existent-post"
        post_service.postRepository.getPost.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Post not found"):
            await post_service.optimizePost(post_id)

    @pytest.mark.asyncio
    async def test_validate_post_request_valid(self, post_service: PostService, sample_post_request: PostGenerationRequest):
        """Test post request validation with valid request."""
        # Act
        result = await post_service.validatePostRequest(sample_post_request)
        
        # Assert
        assert result.isValid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_post_request_invalid(self, post_service: PostService):
        """Test post request validation with invalid request."""
        # Arrange
        invalid_request = PostGenerationRequest(
            topic="",  # Empty topic
            keyPoints=[],
            targetAudience="",
            industry="",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT
        )
        
        # Act
        result = await post_service.validatePostRequest(invalid_request)
        
        # Assert
        assert result.isValid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_generate_post_analytics_success(self, post_service: PostService, sample_post: LinkedInPost):
        """Test successful post analytics generation."""
        # Arrange
        post_id = "test-post-123"
        analysis_result = ContentAnalysisResult(
            sentimentScore=0.8,
            readabilityScore=85.0,
            engagementScore=90.0,
            keywordDensity=0.15,
            structureScore=88.0,
            callToActionScore=92.0
        )
        
        post_service.postRepository.getPost.return_value = sample_post
        post_service.ai_service.analyzeContent.return_value = analysis_result
        
        # Act
        result = await post_service.generatePostAnalytics(post_id)
        
        # Assert
        assert result == analysis_result
        post_service.ai_service.analyzeContent.assert_called_once_with(sample_post.content.text)

    @pytest.mark.asyncio
    async def test_generate_post_analytics_post_not_found(self, post_service: PostService):
        """Test analytics generation when post doesn't exist."""
        # Arrange
        post_id = "non-existent-post"
        post_service.postRepository.getPost.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Post not found"):
            await post_service.generatePostAnalytics(post_id)

    def test_generate_post_id(self, post_service: PostService):
        """Test post ID generation."""
        # Act
        post_id = post_service._PostService__generatePostId()
        
        # Assert
        assert isinstance(post_id, str)
        assert len(post_id) > 0

    def test_generate_cache_key(self, post_service: PostService, sample_post_request: PostGenerationRequest):
        """Test cache key generation."""
        # Act
        cache_key = post_service._PostService__generateCacheKey(sample_post_request)
        
        # Assert
        assert isinstance(cache_key, str)
        assert "linkedin_post" in cache_key
        assert sample_post_request.topic in cache_key

    def test_create_empty_engagement_metrics(self, post_service: PostService):
        """Test empty engagement metrics creation."""
        # Act
        metrics = post_service._PostService__createEmptyEngagementMetrics()
        
        # Assert
        assert isinstance(metrics, EngagementMetrics)
        assert metrics.likes == 0
        assert metrics.comments == 0
        assert metrics.shares == 0
        assert metrics.clicks == 0
        assert metrics.impressions == 0
        assert metrics.reach == 0
        assert metrics.engagementRate == 0.0

    @pytest.mark.asyncio
    async def test_clear_list_cache(self, post_service: PostService):
        """Test list cache clearing."""
        # Act
        await post_service._PostService__clearListCache()
        
        # Assert
        post_service.cacheService.delete.assert_called()

    @pytest.mark.asyncio
    async def test_create_post_ai_service_error(self, post_service: PostService, sample_post_request: PostGenerationRequest):
        """Test post creation when AI service fails."""
        # Arrange
        post_service.cacheService.get.return_value = None
        post_service.ai_service.generatePost.side_effect = Exception("AI service unavailable")
        
        # Act & Assert
        with pytest.raises(Exception, match="AI service unavailable"):
            await post_service.createPost(sample_post_request)

    @pytest.mark.asyncio
    async def test_create_post_repository_error(self, post_service: PostService, sample_post_request: PostGenerationRequest, sample_post: LinkedInPost):
        """Test post creation when repository fails."""
        # Arrange
        mock_ai_response = PostGenerationResponse(
            post=sample_post,
            generationTime=1.5,
            aiScore=85.5,
            optimizationSuggestions=[]
        )
        
        post_service.ai_service.generatePost.return_value = mock_ai_response
        post_service.postRepository.createPost.side_effect = Exception("Database error")
        post_service.cacheService.get.return_value = None
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await post_service.createPost(sample_post_request)

    @pytest.mark.asyncio
    async def test_update_post_not_found(self, post_service: PostService):
        """Test post update when post doesn't exist."""
        # Arrange
        post_id = "non-existent-post"
        updates = {"title": "Updated Title"}
        post_service.postRepository.updatePost.side_effect = Exception("Post not found")
        
        # Act & Assert
        with pytest.raises(Exception, match="Post not found"):
            await post_service.updatePost(post_id, updates)

    @pytest.mark.asyncio
    async def test_list_posts_empty_result(self, post_service: PostService):
        """Test post listing with empty result."""
        # Arrange
        user_id = "user-123"
        post_service.postRepository.listPosts.return_value = []
        
        # Act
        result = await post_service.listPosts(user_id)
        
        # Assert
        assert result == []
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_delete_post_not_found(self, post_service: PostService):
        """Test post deletion when post doesn't exist."""
        # Arrange
        post_id = "non-existent-post"
        post_service.postRepository.deletePost.return_value = False
        
        # Act
        result = await post_service.deletePost(post_id)
        
        # Assert
        assert result is False
