"""
Integration Tests for LinkedIn Posts
===================================

Integration tests that verify the interaction between different components
of the LinkedIn posts system including services, repositories, and external APIs.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import components for integration testing
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestPostServiceIntegration:
    """Integration tests for PostService with real dependencies."""

    @pytest.fixture
    def mock_repository(self) -> PostRepository:
        """Mock repository with realistic behavior."""
        mock = AsyncMock(spec=PostRepository)
        
        # Store posts in memory for realistic testing
        self.posts = {}
        
        async def create_post(post: LinkedInPost) -> LinkedInPost:
            self.posts[post.id] = post
            return post
        
        async def update_post(post_id: str, updates: Dict[str, Any]) -> LinkedInPost:
            if post_id not in self.posts:
                raise ValueError("Post not found")
            post = self.posts[post_id]
            updated_post = LinkedInPost(**{**post.__dict__, **updates})
            self.posts[post_id] = updated_post
            return updated_post
        
        async def get_post(post_id: str) -> Optional[LinkedInPost]:
            return self.posts.get(post_id)
        
        async def list_posts(user_id: str, filters: Optional[Dict] = None) -> List[LinkedInPost]:
            user_posts = [post for post in self.posts.values() if post.userId == user_id]
            if filters:
                if filters.get('status'):
                    user_posts = [post for post in user_posts if post.status == filters['status']]
                if filters.get('limit'):
                    user_posts = user_posts[:filters['limit']]
            return user_posts
        
        async def delete_post(post_id: str) -> bool:
            if post_id in self.posts:
                del self.posts[post_id]
                return True
            return False
        
        mock.createPost = create_post
        mock.updatePost = update_post
        mock.getPost = get_post
        mock.listPosts = list_posts
        mock.deletePost = delete_post
        
        return mock

    @pytest.fixture
    def mock_ai_service(self) -> AIService:
        """Mock AI service with realistic behavior."""
        mock = AsyncMock(spec=AIService)
        
        async def analyze_content(content: str) -> ContentAnalysisResult:
            # Simulate content analysis
            sentiment_score = 0.7 if "positive" in content.lower() else 0.3
            readability_score = 85.0 if len(content) > 100 else 60.0
            engagement_score = 90.0 if "hashtag" in content.lower() else 70.0
            
            return ContentAnalysisResult(
                sentimentScore=sentiment_score,
                readabilityScore=readability_score,
                engagementScore=engagement_score,
                keywordDensity=0.12,
                structureScore=88.0,
                callToActionScore=85.0
            )
        
        async def generate_post(request: PostGenerationRequest) -> PostGenerationResponse:
            # Simulate post generation
            content_text = f"Generated post about {request.topic} for {request.targetAudience}."
            if request.keywords:
                content_text += f" Keywords: {', '.join(request.keywords)}"
            
            post_content = PostContent(
                text=content_text,
                hashtags=[f"#{kw.lower()}" for kw in (request.keywords or [])],
                mentions=[],
                links=[],
                images=[],
                callToAction="Learn more about this topic!"
            )
            
            generated_post = LinkedInPost(
                id="generated-post-123",
                userId=request.userId or "default-user",
                title=request.topic,
                content=post_content,
                postType=request.postType,
                tone=request.tone,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                engagement=EngagementMetrics(
                    likes=0, comments=0, shares=0, clicks=0,
                    impressions=0, reach=0, engagementRate=0.0
                ),
                aiScore=85.0,
                optimizationSuggestions=["Add more hashtags", "Include call-to-action"],
                keywords=request.keywords or [],
                externalMetadata={},
                performanceScore=0,
                reachScore=0,
                engagementScore=0
            )
            
            return PostGenerationResponse(
                post=generated_post,
                generationTime=1.5,
                aiScore=85.0,
                optimizationSuggestions=["Add more hashtags"]
            )
        
        async def optimize_post(post: LinkedInPost) -> PostOptimizationResult:
            # Simulate post optimization
            optimized_content = PostContent(
                text=post.content.text + " [Optimized]",
                hashtags=post.content.hashtags + ["#optimized"],
                mentions=post.content.mentions,
                links=post.content.links,
                images=post.content.images,
                callToAction=post.content.callToAction or "Learn more!"
            )
            
            optimized_post = LinkedInPost(
                **{**post.__dict__, 'content': optimized_content, 'aiScore': post.aiScore + 5}
            )
            
            return PostOptimizationResult(
                originalPost=post,
                optimizedPost=optimized_post,
                optimizationScore=90.0,
                suggestions=["Improved content", "Added hashtags", "Enhanced call-to-action"],
                processingTime=2.0
            )
        
        mock.analyzeContent = analyze_content
        mock.generatePost = generate_post
        mock.optimizePost = optimize_post
        
        return mock

    @pytest.fixture
    def mock_cache_service(self) -> CacheService:
        """Mock cache service with realistic behavior."""
        mock = AsyncMock(spec=CacheService)
        
        # Store cache in memory
        self.cache = {}
        
        async def get(key: str):
            return self.cache.get(key)
        
        async def set(key: str, value: Any, ttl: Optional[int] = None):
            self.cache[key] = value
        
        async def delete(key: str):
            if key in self.cache:
                del self.cache[key]
        
        async def clear():
            self.cache.clear()
        
        mock.get = get
        mock.set = set
        mock.delete = delete
        mock.clear = clear
        
        return mock

    @pytest.fixture
    def post_service(
        self,
        mock_repository: PostRepository,
        mock_ai_service: AIService,
        mock_cache_service: CacheService
    ) -> PostService:
        """Create PostService with mocked dependencies."""
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.fixture
    def sample_request(self) -> PostGenerationRequest:
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

    @pytest.mark.asyncio
    async def test_create_post_full_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test complete post creation flow with all components."""
        # Act
        result = await post_service.createPost(sample_request)
        
        # Assert
        assert result is not None
        assert result.title == "AI in Modern Business"
        assert result.postType == PostType.TEXT
        assert result.tone == PostTone.PROFESSIONAL
        assert result.status == PostStatus.DRAFT
        assert result.aiScore == 85.0
        assert len(result.optimizationSuggestions) > 0
        assert "AI" in result.keywords

    @pytest.mark.asyncio
    async def test_create_and_update_post_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test creating and updating a post."""
        # Create post
        created_post = await post_service.createPost(sample_request)
        
        # Update post
        updates = {
            "title": "Updated AI in Modern Business",
            "status": PostStatus.SCHEDULED,
            "scheduledAt": datetime.now() + timedelta(hours=1)
        }
        
        updated_post = await post_service.updatePost(created_post.id, updates)
        
        # Assert
        assert updated_post.title == "Updated AI in Modern Business"
        assert updated_post.status == PostStatus.SCHEDULED
        assert updated_post.scheduledAt is not None

    @pytest.mark.asyncio
    async def test_create_and_list_posts_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test creating and listing posts."""
        # Create multiple posts
        post1 = await post_service.createPost(sample_request)
        
        # Create second post with different topic
        request2 = PostGenerationRequest(
            topic="Digital Transformation",
            keyPoints=["Technology adoption", "Process improvement"],
            targetAudience="IT professionals",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["digital", "transformation", "technology"]
        )
        post2 = await post_service.createPost(request2)
        
        # List all posts
        all_posts = await post_service.listPosts("default-user")
        
        # Assert
        assert len(all_posts) == 2
        assert any(post.id == post1.id for post in all_posts)
        assert any(post.id == post2.id for post in all_posts)

    @pytest.mark.asyncio
    async def test_create_and_optimize_post_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test creating and optimizing a post."""
        # Create post
        created_post = await post_service.createPost(sample_request)
        
        # Optimize post
        optimization_result = await post_service.optimizePost(created_post.id)
        
        # Assert
        assert optimization_result.originalPost.id == created_post.id
        assert optimization_result.optimizationScore == 90.0
        assert len(optimization_result.suggestions) > 0
        assert optimization_result.processingTime > 0

    @pytest.mark.asyncio
    async def test_create_and_analyze_post_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test creating and analyzing a post."""
        # Create post
        created_post = await post_service.createPost(sample_request)
        
        # Analyze post
        analysis_result = await post_service.generatePostAnalytics(created_post.id)
        
        # Assert
        assert analysis_result.sentimentScore >= 0
        assert analysis_result.readabilityScore > 0
        assert analysis_result.engagementScore > 0
        assert analysis_result.keywordDensity > 0

    @pytest.mark.asyncio
    async def test_cache_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test cache integration with post creation."""
        # Create post (should be cached)
        post1 = await post_service.createPost(sample_request)
        
        # Create same post again (should hit cache)
        post2 = await post_service.createPost(sample_request)
        
        # Assert both posts are the same (from cache)
        assert post1.id == post2.id
        assert post1.title == post2.title

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, post_service: PostService):
        """Test error handling in integration scenarios."""
        # Test getting non-existent post
        result = await post_service.getPost("non-existent-post")
        assert result is None
        
        # Test updating non-existent post
        with pytest.raises(Exception):
            await post_service.updatePost("non-existent-post", {"title": "Updated"})
        
        # Test optimizing non-existent post
        with pytest.raises(ValueError, match="Post not found"):
            await post_service.optimizePost("non-existent-post")

    @pytest.mark.asyncio
    async def test_concurrent_post_creation(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test concurrent post creation."""
        # Create multiple posts concurrently
        tasks = [
            post_service.createPost(sample_request),
            post_service.createPost(sample_request),
            post_service.createPost(sample_request)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Assert all posts were created successfully
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert result.title == "AI in Modern Business"

    @pytest.mark.asyncio
    async def test_post_lifecycle_integration(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test complete post lifecycle."""
        # 1. Create post
        post = await post_service.createPost(sample_request)
        assert post.status == PostStatus.DRAFT
        
        # 2. Update post with scheduling
        scheduled_updates = {
            "status": PostStatus.SCHEDULED,
            "scheduledAt": datetime.now() + timedelta(hours=1)
        }
        scheduled_post = await post_service.updatePost(post.id, scheduled_updates)
        assert scheduled_post.status == PostStatus.SCHEDULED
        
        # 3. Optimize post
        optimization_result = await post_service.optimizePost(post.id)
        assert optimization_result.optimizationScore > 0
        
        # 4. Analyze post
        analysis_result = await post_service.generatePostAnalytics(post.id)
        assert analysis_result.sentimentScore >= 0
        
        # 5. Update to published
        published_updates = {
            "status": PostStatus.PUBLISHED,
            "publishedAt": datetime.now()
        }
        published_post = await post_service.updatePost(post.id, published_updates)
        assert published_post.status == PostStatus.PUBLISHED
        
        # 6. Delete post
        delete_result = await post_service.deletePost(post.id)
        assert delete_result is True
        
        # 7. Verify post is deleted
        deleted_post = await post_service.getPost(post.id)
        assert deleted_post is None

    @pytest.mark.asyncio
    async def test_filtered_post_listing(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test filtered post listing."""
        # Create posts with different statuses
        draft_post = await post_service.createPost(sample_request)
        
        # Create scheduled post
        scheduled_request = PostGenerationRequest(
            topic="Scheduled Post",
            keyPoints=["Point 1"],
            targetAudience="Audience",
            industry="Industry",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT
        )
        scheduled_post = await post_service.createPost(scheduled_request)
        await post_service.updatePost(scheduled_post.id, {
            "status": PostStatus.SCHEDULED,
            "scheduledAt": datetime.now() + timedelta(hours=1)
        })
        
        # List draft posts only
        draft_posts = await post_service.listPosts("default-user", {"status": PostStatus.DRAFT})
        assert len(draft_posts) >= 1
        assert all(post.status == PostStatus.DRAFT for post in draft_posts)
        
        # List scheduled posts only
        scheduled_posts = await post_service.listPosts("default-user", {"status": PostStatus.SCHEDULED})
        assert len(scheduled_posts) >= 1
        assert all(post.status == PostStatus.SCHEDULED for post in scheduled_posts)

    @pytest.mark.asyncio
    async def test_ai_service_integration_errors(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test AI service integration error handling."""
        # Mock AI service to throw error
        post_service.ai_service.generatePost.side_effect = Exception("AI service unavailable")
        
        # Attempt to create post should fail
        with pytest.raises(Exception, match="AI service unavailable"):
            await post_service.createPost(sample_request)

    @pytest.mark.asyncio
    async def test_repository_integration_errors(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test repository integration error handling."""
        # Mock repository to throw error
        post_service.postRepository.createPost.side_effect = Exception("Database connection failed")
        
        # Attempt to create post should fail
        with pytest.raises(Exception, match="Database connection failed"):
            await post_service.createPost(sample_request)

    @pytest.mark.asyncio
    async def test_cache_integration_errors(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test cache integration error handling."""
        # Mock cache service to throw error
        post_service.cacheService.get.side_effect = Exception("Cache service unavailable")
        
        # Post creation should still work (cache is optional)
        result = await post_service.createPost(sample_request)
        assert result is not None
        assert result.title == "AI in Modern Business"


class TestPerformanceIntegration:
    """Integration tests focusing on performance aspects."""

    @pytest.mark.asyncio
    async def test_bulk_post_creation_performance(self, post_service: PostService):
        """Test performance of bulk post creation."""
        import time
        
        # Create multiple posts and measure time
        start_time = time.time()
        
        requests = []
        for i in range(10):
            request = PostGenerationRequest(
                topic=f"Performance Test Post {i}",
                keyPoints=[f"Point {i}"],
                targetAudience="Test audience",
                industry="Technology",
                tone=PostTone.PROFESSIONAL,
                postType=PostType.TEXT,
                keywords=[f"test{i}", "performance"]
            )
            requests.append(request)
        
        # Create posts concurrently
        tasks = [post_service.createPost(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert all posts were created
        assert len(results) == 10
        assert execution_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test cache performance impact."""
        import time
        
        # First request (cache miss)
        start_time = time.time()
        post1 = await post_service.createPost(sample_request)
        first_request_time = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        post2 = await post_service.createPost(sample_request)
        second_request_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_request_time < first_request_time
        assert post1.id == post2.id

    @pytest.mark.asyncio
    async def test_concurrent_optimization_performance(self, post_service: PostService, sample_request: PostGenerationRequest):
        """Test concurrent optimization performance."""
        import time
        
        # Create post
        post = await post_service.createPost(sample_request)
        
        # Run multiple optimizations concurrently
        start_time = time.time()
        
        tasks = [post_service.optimizePost(post.id) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert all optimizations completed
        assert len(results) == 5
        assert execution_time < 15.0  # Should complete within 15 seconds
        
        for result in results:
            assert result.optimizationScore > 0
            assert result.processingTime > 0
