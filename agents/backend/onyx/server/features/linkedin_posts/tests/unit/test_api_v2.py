from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import orjson
from fastapi import HTTPException
from ...core.domain.entities.linkedin_post import PostStatus, PostType, PostTone
from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ...shared.schemas.linkedin_post_schemas import (
        from ...shared.cache import CacheManager
from typing import Any, List, Dict, Optional
import logging
"""
Unit Tests for LinkedIn Posts API V2
====================================

Comprehensive unit tests for all API endpoints and functionality.
"""


    LinkedInPostCreate,
    LinkedInPostUpdate,
    PostOptimizationRequest,
    BatchOptimizationRequest
)


class TestLinkedInPostUseCases:
    """Test LinkedIn post use cases."""
    
    @pytest.fixture
    def mock_repository(self) -> Any:
        """Mock repository."""
        return AsyncMock(spec=LinkedInPostRepository)
    
    @pytest.fixture
    def use_cases(self, mock_repository) -> Any:
        """Use cases with mocked repository."""
        return LinkedInPostUseCases(mock_repository)
    
    @pytest.mark.asyncio
    async def test_generate_post_success(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test successful post generation."""
        # Arrange
        mock_repository.create.return_value = sample_linkedin_post
        
        # Act
        result = await use_cases.generate_post(
            content="Test content",
            post_type=PostType.ANNOUNCEMENT,
            tone=PostTone.PROFESSIONAL,
            target_audience="professionals",
            industry="technology"
        )
        
        # Assert
        assert result == sample_linkedin_post
        mock_repository.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_post_with_nlp(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test post generation with NLP enhancement."""
        # Arrange
        mock_repository.create.return_value = sample_linkedin_post
        
        # Act
        result = await use_cases.generate_post(
            content="Test content",
            post_type=PostType.ANNOUNCEMENT,
            tone=PostTone.PROFESSIONAL,
            target_audience="professionals",
            industry="technology",
            use_fast_nlp=True,
            use_async_nlp=True
        )
        
        # Assert
        assert result == sample_linkedin_post
        assert result.nlp_enhanced is True
    
    @pytest.mark.asyncio
    async def test_list_posts_success(self, use_cases, mock_repository, sample_posts_batch) -> List[Any]:
        """Test successful post listing."""
        # Arrange
        mock_repository.list_posts.return_value = sample_posts_batch
        
        # Act
        result = await use_cases.list_posts(
            user_id="test-user",
            status=PostStatus.DRAFT,
            limit=10,
            offset=0
        )
        
        # Assert
        assert result == sample_posts_batch
        mock_repository.list_posts.assert_called_once_with(
            user_id="test-user",
            status=PostStatus.DRAFT,
            limit=10,
            offset=0
        )
    
    @pytest.mark.asyncio
    async def test_update_post_success(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test successful post update."""
        # Arrange
        mock_repository.get_by_id.return_value = sample_linkedin_post
        mock_repository.update.return_value = sample_linkedin_post
        
        # Act
        result = await use_cases.update_post(
            post_id="test-post-123",
            content="Updated content",
            post_type=PostType.EDUCATIONAL
        )
        
        # Assert
        assert result == sample_linkedin_post
        mock_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_post_not_found(self, use_cases, mock_repository) -> Any:
        """Test post update when post not found."""
        # Arrange
        mock_repository.get_by_id.return_value = None
        
        # Act & Assert
        result = await use_cases.update_post(
            post_id="non-existent",
            content="Updated content"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_post_success(self, use_cases, mock_repository) -> Any:
        """Test successful post deletion."""
        # Arrange
        mock_repository.delete.return_value = True
        
        # Act
        result = await use_cases.delete_post("test-post-123")
        
        # Assert
        assert result is True
        mock_repository.delete.assert_called_once_with("test-post-123")
    
    @pytest.mark.asyncio
    async def test_optimize_post_success(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test successful post optimization."""
        # Arrange
        mock_repository.get_by_id.return_value = sample_linkedin_post
        mock_repository.update.return_value = sample_linkedin_post
        
        # Act
        result = await use_cases.optimize_post(
            post_id="test-post-123",
            use_async_nlp=True
        )
        
        # Assert
        assert result == sample_linkedin_post
        assert result.nlp_enhanced is True
    
    @pytest.mark.asyncio
    async def test_batch_optimize_posts_success(self, use_cases, mock_repository, sample_posts_batch) -> Any:
        """Test successful batch optimization."""
        # Arrange
        post_ids = [post.id for post in sample_posts_batch]
        mock_repository.batch_get.return_value = sample_posts_batch
        mock_repository.batch_update.return_value = sample_posts_batch
        
        # Act
        result = await use_cases.batch_optimize_posts(
            post_ids=post_ids,
            use_async_nlp=True
        )
        
        # Assert
        assert result == sample_posts_batch
        mock_repository.batch_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_post_engagement_success(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test successful post analysis."""
        # Arrange
        mock_repository.get_by_id.return_value = sample_linkedin_post
        
        # Act
        result = await use_cases.analyze_post_engagement(
            post_id="test-post-123",
            use_async_nlp=True
        )
        
        # Assert
        assert result is not None
        assert "sentiment_score" in result
        assert "readability_score" in result
        assert "keywords" in result
        assert "entities" in result


class TestLinkedInPostRepository:
    """Test LinkedIn post repository."""
    
    @pytest.fixture
    def repository(self) -> Any:
        """Repository instance."""
        return LinkedInPostRepository()
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self, repository, sample_linkedin_post) -> Optional[Dict[str, Any]]:
        """Test successful get by ID."""
        # This would require a real database or mock
        # For now, we'll test the interface
        assert hasattr(repository, 'get_by_id')
        assert asyncio.iscoroutinefunction(repository.get_by_id)
    
    @pytest.mark.asyncio
    async def test_list_posts_interface(self, repository) -> List[Any]:
        """Test list posts interface."""
        assert hasattr(repository, 'list_posts')
        assert asyncio.iscoroutinefunction(repository.list_posts)
    
    @pytest.mark.asyncio
    async def test_create_interface(self, repository) -> Any:
        """Test create interface."""
        assert hasattr(repository, 'create')
        assert asyncio.iscoroutinefunction(repository.create)
    
    @pytest.mark.asyncio
    async def test_update_interface(self, repository) -> Any:
        """Test update interface."""
        assert hasattr(repository, 'update')
        assert asyncio.iscoroutinefunction(repository.update)
    
    @pytest.mark.asyncio
    async def test_delete_interface(self, repository) -> Any:
        """Test delete interface."""
        assert hasattr(repository, 'delete')
        assert asyncio.iscoroutinefunction(repository.delete)


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_linkedin_post_create_valid(self) -> Any:
        """Test valid LinkedIn post creation schema."""
        data = {
            "content": "Test post content",
            "post_type": "announcement",
            "tone": "professional",
            "target_audience": "professionals",
            "industry": "technology"
        }
        
        post = LinkedInPostCreate(**data)
        
        assert post.content == data["content"]
        assert post.post_type.value == data["post_type"]
        assert post.tone.value == data["tone"]
        assert post.target_audience == data["target_audience"]
        assert post.industry == data["industry"]
    
    def test_linkedin_post_create_invalid(self) -> Any:
        """Test invalid LinkedIn post creation schema."""
        data = {
            "content": "",  # Empty content
            "post_type": "invalid_type",
            "tone": "invalid_tone"
        }
        
        with pytest.raises(ValueError):
            LinkedInPostCreate(**data)
    
    def test_linkedin_post_update_partial(self) -> Any:
        """Test partial LinkedIn post update schema."""
        data = {
            "content": "Updated content"
        }
        
        post = LinkedInPostUpdate(**data)
        
        assert post.content == data["content"]
        assert post.post_type is None
        assert post.tone is None
    
    async def test_post_optimization_request_valid(self) -> Any:
        """Test valid optimization request schema."""
        data = {
            "use_async_nlp": True
        }
        
        request = PostOptimizationRequest(**data)
        
        assert request.use_async_nlp is True
    
    async def test_batch_optimization_request_valid(self) -> Any:
        """Test valid batch optimization request schema."""
        data = {
            "post_ids": ["post-1", "post-2", "post-3"],
            "use_async_nlp": True
        }
        
        request = BatchOptimizationRequest(**data)
        
        assert request.post_ids == data["post_ids"]
        assert request.use_async_nlp is True
    
    async def test_batch_optimization_request_empty_ids(self) -> Any:
        """Test batch optimization with empty post IDs."""
        data = {
            "post_ids": [],
            "use_async_nlp": True
        }
        
        with pytest.raises(ValueError):
            BatchOptimizationRequest(**data)


class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_manager(self) -> Any:
        """Cache manager instance."""
        return CacheManager(memory_size=100, memory_ttl=60)
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager) -> Optional[Dict[str, Any]]:
        """Test basic cache set and get operations."""
        # Arrange
        key = "test_key"
        value = {"test": "data"}
        
        # Act
        await cache_manager.set(key, value, expire=60)
        result = await cache_manager.get(key)
        
        # Assert
        assert result == value
    
    @pytest.mark.asyncio
    async def test_cache_get_missing(self, cache_manager) -> Optional[Dict[str, Any]]:
        """Test getting non-existent cache key."""
        # Act
        result = await cache_manager.get("non_existent")
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_manager) -> Any:
        """Test cache deletion."""
        # Arrange
        key = "test_key"
        value = {"test": "data"}
        await cache_manager.set(key, value)
        
        # Act
        result = await cache_manager.delete(key)
        
        # Assert
        assert result is True
        
        # Verify deletion
        cached_value = await cache_manager.get(key)
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_get_many(self, cache_manager) -> Optional[Dict[str, Any]]:
        """Test getting multiple cache keys."""
        # Arrange
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        for key, value in items.items():
            await cache_manager.set(key, value)
        
        # Act
        result = await cache_manager.get_many(list(items.keys()))
        
        # Assert
        assert result == items
    
    @pytest.mark.asyncio
    async def test_cache_set_many(self, cache_manager) -> Any:
        """Test setting multiple cache keys."""
        # Arrange
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        # Act
        result = await cache_manager.set_many(items)
        
        # Assert
        assert result is True
        
        # Verify all items are cached
        for key, value in items.items():
            cached_value = await cache_manager.get(key)
            assert cached_value == value
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, cache_manager) -> Any:
        """Test cache clearing."""
        # Arrange
        await cache_manager.set("key1", "value1")
        await cache_manager.set("key2", "value2")
        
        # Act
        result = await cache_manager.clear()
        
        # Assert
        assert result is True
        
        # Verify cache is empty
        assert await cache_manager.get("key1") is None
        assert await cache_manager.get("key2") is None


class TestMiddleware:
    """Test middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_middleware(self, test_app, async_client) -> Any:
        """Test performance middleware."""
        # Act
        response = await async_client.get("/linkedin-posts/health")
        
        # Assert
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        assert "X-Request-ID" in response.headers
    
    @pytest.mark.asyncio
    async def test_cache_middleware(self, test_app, async_client, mock_cache_manager) -> Any:
        """Test cache middleware."""
        # Arrange
        mock_cache_manager.get.return_value = orjson.dumps({
            "content": '{"test": "data"}',
            "status_code": 200,
            "headers": {},
            "media_type": "application/json",
            "etag": '"test-etag"'
        })
        
        # Act
        response = await async_client.get("/linkedin-posts/health")
        
        # Assert
        assert response.status_code == 200
        assert "X-Cache" in response.headers
        assert response.headers["X-Cache"] == "HIT"
    
    @pytest.mark.asyncio
    async def test_security_middleware(self, test_app, async_client) -> Any:
        """Test security middleware."""
        # Act
        response = await async_client.get("/linkedin-posts/health")
        
        # Assert
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self, use_cases, mock_repository) -> Any:
        """Test handling of database connection errors."""
        # Arrange
        mock_repository.create.side_effect = Exception("Database connection failed")
        
        # Act & Assert
        with pytest.raises(Exception):
            await use_cases.generate_post(
                content="Test content",
                post_type=PostType.ANNOUNCEMENT,
                tone=PostTone.PROFESSIONAL,
                target_audience="professionals",
                industry="technology"
            )
    
    @pytest.mark.asyncio
    async def test_nlp_service_error(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test handling of NLP service errors."""
        # Arrange
        mock_repository.create.return_value = sample_linkedin_post
        
        with patch('linkedin_posts.infrastructure.nlp.nlp_processor') as mock_nlp:
            mock_nlp.process_text.side_effect = Exception("NLP service unavailable")
            
            # Act
            result = await use_cases.generate_post(
                content="Test content",
                post_type=PostType.ANNOUNCEMENT,
                tone=PostTone.PROFESSIONAL,
                target_audience="professionals",
                industry="technology",
                use_fast_nlp=True
            )
            
            # Assert - should still work without NLP
            assert result == sample_linkedin_post
    
    @pytest.mark.asyncio
    async def test_validation_error(self) -> Any:
        """Test handling of validation errors."""
        # Act & Assert
        with pytest.raises(ValueError):
            LinkedInPostCreate(
                content="",  # Invalid empty content
                post_type="invalid_type",
                tone="invalid_tone"
            )


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_post_creation(self, use_cases, mock_repository, sample_linkedin_post) -> Any:
        """Test concurrent post creation performance."""
        # Arrange
        mock_repository.create.return_value = sample_linkedin_post
        
        # Act
        async def create_post():
            
    """create_post function."""
return await use_cases.generate_post(
                content="Test content",
                post_type=PostType.ANNOUNCEMENT,
                tone=PostTone.PROFESSIONAL,
                target_audience="professionals",
                industry="technology"
            )
        
        # Create 10 posts concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[create_post() for _ in range(10)])
        end_time = asyncio.get_event_loop().time()
        
        # Assert
        assert len(results) == 10
        assert all(result == sample_linkedin_post for result in results)
        
        # Performance assertion (should be fast with async)
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete in less than 1 second
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, use_cases, mock_repository, sample_posts_batch) -> Any:
        """Test batch processing performance."""
        # Arrange
        mock_repository.batch_create.return_value = sample_posts_batch
        
        # Act
        start_time = asyncio.get_event_loop().time()
        result = await use_cases.batch_create_posts(
            posts_data=[{
                "content": f"Test post {i}",
                "post_type": PostType.EDUCATIONAL,
                "tone": PostTone.FRIENDLY,
                "target_audience": "professionals",
                "industry": "technology"
            } for i in range(50)]
        )
        end_time = asyncio.get_event_loop().time()
        
        # Assert
        assert len(result) == 50
        total_time = end_time - start_time
        assert total_time < 2.0  # Should complete in less than 2 seconds


# Export test classes
__all__ = [
    "TestLinkedInPostUseCases",
    "TestLinkedInPostRepository",
    "TestSchemas",
    "TestCacheManager",
    "TestMiddleware",
    "TestErrorHandling",
    "TestPerformance"
] 