from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
import aioresponses
import responses
import httpretty
from httpx import AsyncClient
import redis.asyncio as redis
from fastapi.testclient import TestClient
from ...core.domain.entities.linkedin_post import LinkedInPost, PostStatus, PostType, PostTone
from ...application.use_cases.linkedin_post_use_cases import LinkedInPostUseCases
from ...infrastructure.repositories.linkedin_post_repository import LinkedInPostRepository
from ...shared.cache import CacheManager
from ...shared.config import Settings
from ..conftest_advanced import (
        from fastapi import FastAPI
        from ...presentation.api.linkedin_post_router_v2 import router
        import psutil
        import os
from typing import Any, List, Dict, Optional
import logging
"""
Advanced Integration Tests with Best Libraries
=============================================

Integration tests using TestContainers, aioresponses, responses, and other advanced libraries.
"""


# Advanced integration testing libraries

# FastAPI testing

# Our modules

# Import fixtures and factories
    LinkedInPostFactory,
    PostDataFactory,
    test_data_generator
)


class TestDatabaseIntegrationAdvanced:
    """Advanced database integration tests using TestContainers."""
    
    @pytest.fixture(scope="class")
    def postgres_container(self) -> Any:
        """PostgreSQL container for integration testing."""
        with DockerContainer("postgres:15-alpine") as container:
            container.with_env("POSTGRES_DB", "test_db")
            container.with_env("POSTGRES_USER", "test_user")
            container.with_env("POSTGRES_PASSWORD", "test_password")
            container.with_exposed_ports(5432)
            container.start()
            
            wait_for_logs(container, "database system is ready to accept connections")
            
            # Get connection details
            host = container.get_container_host_ip()
            port = container.get_exposed_port(5432)
            
            database_url = f"postgresql://test_user:test_password@{host}:{port}/test_db"
            
            yield database_url
    
    @pytest.fixture(scope="class")
    def redis_container(self) -> Any:
        """Redis container for integration testing."""
        with DockerContainer("redis:7-alpine") as container:
            container.with_exposed_ports(6379)
            container.start()
            
            wait_for_logs(container, "Ready to accept connections")
            
            host = container.get_container_host_ip()
            port = container.get_exposed_port(6379)
            
            redis_url = f"redis://{host}:{port}"
            
            yield redis_url
    
    @pytest.mark.asyncio
    async def test_database_connection_integration(self, postgres_container) -> Any:
        """Test database connection integration."""
        # This would test actual database operations
        # For now, we'll test the connection string format
        assert "postgresql://" in postgres_container
        assert "test_db" in postgres_container
    
    @pytest.mark.asyncio
    async def test_redis_connection_integration(self, redis_container) -> Any:
        """Test Redis connection integration."""
        # Test actual Redis connection
        client = redis.from_url(redis_container, decode_responses=True)
        
        try:
            # Test basic operations
            await client.ping()
            
            # Test set/get
            await client.set("test_key", "test_value")
            value = await client.get("test_key")
            assert value == "test_value"
            
            # Test delete
            await client.delete("test_key")
            value = await client.get("test_key")
            assert value is None
            
        finally:
            await client.close()


class TestAPIIntegrationAdvanced:
    """Advanced API integration tests using aioresponses and responses."""
    
    @pytest.fixture
    def test_app(self) -> Any:
        """Test FastAPI application."""
        
        app = FastAPI(title="LinkedIn Posts API Integration Test")
        app.include_router(router)
        
        return app
    
    @pytest.fixture
    def test_client(self, test_app) -> Any:
        """Test client for FastAPI."""
        return TestClient(test_app)
    
    @pytest.fixture
    async def async_client(self, test_app) -> Any:
        """Async test client for FastAPI."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_create_post_integration(self, async_client) -> Any:
        """Test post creation integration."""
        post_data = PostDataFactory()
        
        response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["content"] == post_data["content"]
        assert data["post_type"] == post_data["post_type"]
        assert data["tone"] == post_data["tone"]
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_batch_create_posts_integration(self, async_client) -> Any:
        """Test batch post creation integration."""
        batch_data = PostDataFactory.build_batch(3)
        
        response = await async_client.post(
            "/linkedin-posts/batch",
            json={"posts": batch_data},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert len(data["posts"]) == 3
        for post, original_data in zip(data["posts"], batch_data):
            assert post["content"] == original_data["content"]
    
    @pytest.mark.asyncio
    async def test_get_post_integration(self, async_client) -> Optional[Dict[str, Any]]:
        """Test get post integration."""
        # First create a post
        post_data = PostDataFactory()
        create_response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        post_id = create_response.json()["id"]
        
        # Then get the post
        response = await async_client.get(
            f"/linkedin-posts/{post_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == post_id
        assert data["content"] == post_data["content"]
    
    @pytest.mark.asyncio
    async def test_list_posts_integration(self, async_client) -> List[Any]:
        """Test list posts integration."""
        # Create some posts first
        for _ in range(3):
            post_data = PostDataFactory()
            await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        # List posts
        response = await async_client.get(
            "/linkedin-posts/",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "posts" in data
        assert len(data["posts"]) >= 3
    
    @pytest.mark.asyncio
    async def test_update_post_integration(self, async_client) -> Any:
        """Test update post integration."""
        # First create a post
        post_data = PostDataFactory()
        create_response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        post_id = create_response.json()["id"]
        
        # Update the post
        update_data = {"content": "Updated content"}
        response = await async_client.put(
            f"/linkedin-posts/{post_id}",
            json=update_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["content"] == "Updated content"
        assert data["id"] == post_id
    
    @pytest.mark.asyncio
    async def test_delete_post_integration(self, async_client) -> Any:
        """Test delete post integration."""
        # First create a post
        post_data = PostDataFactory()
        create_response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        post_id = create_response.json()["id"]
        
        # Delete the post
        response = await async_client.delete(
            f"/linkedin-posts/{post_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 204
        
        # Verify post is deleted
        get_response = await async_client.get(
            f"/linkedin-posts/{post_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert get_response.status_code == 404


class TestExternalAPIIntegrationAdvanced:
    """Advanced external API integration tests using aioresponses."""
    
    @pytest.mark.asyncio
    async def test_nlp_service_integration(self) -> Any:
        """Test NLP service integration with mocked responses."""
        with aioresponses() as m:
            # Mock NLP service responses
            m.post(
                "http://nlp-service/api/process",
                payload={
                    "sentiment_score": 0.8,
                    "readability_score": 75.5,
                    "keywords": ["test", "linkedin", "post"],
                    "entities": ["LinkedIn", "testing"],
                    "processing_time": 0.12
                },
                status=200
            )
            
            # Test the integration
            async with AsyncClient() as client:
                response = await client.post(
                    "http://nlp-service/api/process",
                    json={"text": "Test LinkedIn post"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "sentiment_score" in data
                assert "readability_score" in data
                assert "keywords" in data
                assert "entities" in data
    
    @pytest.mark.asyncio
    async def test_ai_service_integration(self) -> Any:
        """Test AI service integration with mocked responses."""
        with aioresponses() as m:
            # Mock AI service responses
            m.post(
                "http://ai-service/api/generate",
                payload={
                    "content": "Generated LinkedIn post content",
                    "confidence": 0.95,
                    "processing_time": 0.5
                },
                status=200
            )
            
            # Test the integration
            async with AsyncClient() as client:
                response = await client.post(
                    "http://ai-service/api/generate",
                    json={
                        "prompt": "Generate a LinkedIn post about technology",
                        "tone": "professional"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "content" in data
                assert "confidence" in data
                assert "processing_time" in data
    
    @pytest.mark.asyncio
    async async def test_external_api_error_handling(self) -> Any:
        """Test external API error handling."""
        with aioresponses() as m:
            # Mock service unavailable
            m.post(
                "http://external-service/api/process",
                status=503,
                payload={"error": "Service unavailable"}
            )
            
            # Test error handling
            async with AsyncClient() as client:
                response = await client.post(
                    "http://external-service/api/process",
                    json={"data": "test"}
                )
                
                assert response.status_code == 503
                data = response.json()
                assert "error" in data


class TestCacheIntegrationAdvanced:
    """Advanced cache integration tests."""
    
    @pytest.fixture
    async def cache_manager(self) -> Any:
        """Cache manager for testing."""
        manager = CacheManager(memory_size=100, memory_ttl=60)
        yield manager
        await manager.clear()
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, cache_manager) -> Any:
        """Test cache integration."""
        # Test set/get
        await cache_manager.set("test_key", "test_value", expire=60)
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Test set_many/get_many
        items = {"key1": "value1", "key2": "value2", "key3": "value3"}
        await cache_manager.set_many(items)
        result = await cache_manager.get_many(["key1", "key2", "key3"])
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["key3"] == "value3"
        
        # Test delete
        await cache_manager.delete("key1")
        value = await cache_manager.get("key1")
        assert value is None
        
        # Test clear
        await cache_manager.clear()
        result = await cache_manager.get_many(["key2", "key3"])
        assert all(v is None for v in result.values())
    
    @pytest.mark.asyncio
    async def test_cache_expiration_integration(self, cache_manager) -> Any:
        """Test cache expiration integration."""
        # Set item with short expiration
        await cache_manager.set("expire_key", "expire_value", expire=1)
        
        # Should be available immediately
        value = await cache_manager.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be expired
        value = await cache_manager.get("expire_key")
        assert value is None


class TestPerformanceIntegrationAdvanced:
    """Advanced performance integration tests."""
    
    @pytest.fixture
    async def async_client(self, test_app) -> Any:
        """Async test client for FastAPI."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async async def test_concurrent_requests_integration(self, async_client) -> Any:
        """Test concurrent requests integration."""
        # Create multiple concurrent requests
        async def make_request():
            
    """make_request function."""
post_data = PostDataFactory()
            response = await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
            return response.status_code
        
        # Run 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(status == 201 for status in results)
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, async_client) -> Any:
        """Test batch processing integration."""
        # Create large batch
        batch_data = PostDataFactory.build_batch(50)
        
        start_time = asyncio.get_event_loop().time()
        
        response = await async_client.post(
            "/linkedin-posts/batch",
            json={"posts": batch_data},
            headers={"Authorization": "Bearer test-token"}
        )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        assert response.status_code == 201
        assert processing_time < 5.0  # Should complete within 5 seconds
        
        data = response.json()
        assert len(data["posts"]) == 50
    
    @pytest.mark.asyncio
    async def test_memory_usage_integration(self, async_client) -> Any:
        """Test memory usage integration."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many posts
        for _ in range(100):
            post_data = PostDataFactory()
            await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestSecurityIntegrationAdvanced:
    """Advanced security integration tests."""
    
    @pytest.fixture
    async def async_client(self, test_app) -> Any:
        """Async test client for FastAPI."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_authentication_integration(self, async_client) -> Any:
        """Test authentication integration."""
        post_data = PostDataFactory()
        
        # Test without authentication
        response = await async_client.post(
            "/linkedin-posts/",
            json=post_data
        )
        
        assert response.status_code == 401
        
        # Test with invalid token
        response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
        
        # Test with valid token
        response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, async_client) -> Any:
        """Test rate limiting integration."""
        post_data = PostDataFactory()
        
        # Make many requests quickly
        responses = []
        for _ in range(20):
            response = await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
            responses.append(response.status_code)
        
        # Some requests should be rate limited
        assert 429 in responses
    
    @pytest.mark.asyncio
    async def test_input_validation_integration(self, async_client) -> Any:
        """Test input validation integration."""
        # Test with invalid data
        invalid_data = {
            "content": "",  # Empty content
            "post_type": "invalid_type",  # Invalid post type
            "tone": "invalid_tone",  # Invalid tone
            "target_audience": "",  # Empty audience
            "industry": ""  # Empty industry
        }
        
        response = await async_client.post(
            "/linkedin-posts/",
            json=invalid_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        
        # Test with valid data
        valid_data = PostDataFactory()
        
        response = await async_client.post(
            "/linkedin-posts/",
            json=valid_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201


class TestErrorHandlingIntegrationAdvanced:
    """Advanced error handling integration tests."""
    
    @pytest.fixture
    async def async_client(self, test_app) -> Any:
        """Async test client for FastAPI."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_database_error_integration(self, async_client) -> Any:
        """Test database error handling integration."""
        # This would require mocking the database to throw errors
        # For now, we'll test the error response format
        
        # Test with invalid ID format
        response = await async_client.get(
            "/linkedin-posts/invalid-id",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_nlp_service_error_integration(self, async_client) -> Any:
        """Test NLP service error handling integration."""
        post_data = PostDataFactory()
        
        # Test with NLP service unavailable
        with aioresponses() as m:
            m.post(
                "http://nlp-service/api/process",
                status=503,
                payload={"error": "Service unavailable"}
            )
            
            response = await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Should still work without NLP
            assert response.status_code == 201
            data = response.json()
            assert data["nlp_enhanced"] is False
    
    @pytest.mark.asyncio
    async def test_timeout_error_integration(self, async_client) -> Any:
        """Test timeout error handling integration."""
        post_data = PostDataFactory()
        
        # Test with slow external service
        with aioresponses() as m:
            m.post(
                "http://external-service/api/process",
                exception=asyncio.TimeoutError("Request timeout")
            )
            
            response = await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Should handle timeout gracefully
            assert response.status_code in [201, 500]


class TestMonitoringIntegrationAdvanced:
    """Advanced monitoring integration tests."""
    
    @pytest.fixture
    async def async_client(self, test_app) -> Any:
        """Async test client for FastAPI."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_metrics_integration(self, async_client) -> Any:
        """Test metrics integration."""
        # Make some requests
        post_data = PostDataFactory()
        
        for _ in range(5):
            await async_client.post(
                "/linkedin-posts/",
                json=post_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        # Check metrics endpoint
        response = await async_client.get("/metrics")
        
        assert response.status_code == 200
        metrics_text = response.text
        
        # Should contain Prometheus metrics
        assert "http_requests_total" in metrics_text
        assert "http_request_duration_seconds" in metrics_text
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, async_client) -> Any:
        """Test health check integration."""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_logging_integration(self, async_client) -> Any:
        """Test logging integration."""
        post_data = PostDataFactory()
        
        # Make request and check logs
        response = await async_client.post(
            "/linkedin-posts/",
            json=post_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 201
        
        # Check that request ID is in response headers
        assert "X-Request-ID" in response.headers


# Export test classes
__all__ = [
    "TestDatabaseIntegrationAdvanced",
    "TestAPIIntegrationAdvanced",
    "TestExternalAPIIntegrationAdvanced",
    "TestCacheIntegrationAdvanced",
    "TestPerformanceIntegrationAdvanced",
    "TestSecurityIntegrationAdvanced",
    "TestErrorHandlingIntegrationAdvanced",
    "TestMonitoringIntegrationAdvanced"
] 