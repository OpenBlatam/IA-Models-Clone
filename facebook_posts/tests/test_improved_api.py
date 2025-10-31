"""
Comprehensive tests for the improved Facebook Posts API
Following FastAPI testing best practices
"""

import pytest
import asyncio
from typing import Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

from ..app import app
from ..core.models import PostRequest, ContentType, AudienceType, OptimizationLevel
from ..api.schemas import PostUpdateRequest, BatchPostRequest


class TestImprovedFacebookPostsAPI:
    """Test suite for the improved Facebook Posts API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def sample_post_request(self) -> Dict[str, Any]:
        """Sample post request data"""
        return {
            "topic": "AI in Modern Business",
            "audience_type": "professionals",
            "content_type": "educational",
            "tone": "professional",
            "optimization_level": "standard",
            "include_hashtags": True,
            "tags": ["ai", "business", "technology"]
        }
    
    @pytest.fixture
    def sample_batch_request(self) -> Dict[str, Any]:
        """Sample batch request data"""
        return {
            "requests": [
                {
                    "topic": "Digital Marketing Trends",
                    "audience_type": "professionals",
                    "content_type": "educational",
                    "tone": "professional"
                },
                {
                    "topic": "Remote Work Tips",
                    "audience_type": "general",
                    "content_type": "educational",
                    "tone": "friendly"
                }
            ],
            "parallel_processing": True
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_generate_post_success(self, client, sample_post_request):
        """Test successful post generation"""
        response = client.post("/api/v1/posts/generate", json=sample_post_request)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert "post" in data
        assert "processing_time" in data
        assert "optimizations_applied" in data
        
        post = data["post"]
        assert "id" in post
        assert "content" in post
        assert "status" in post
        assert "content_type" in post
        assert "audience_type" in post
    
    def test_generate_post_validation_error(self, client):
        """Test post generation with validation errors"""
        # Test empty topic
        invalid_request = {
            "topic": "",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        response = client.post("/api/v1/posts/generate", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_generate_post_short_topic(self, client):
        """Test post generation with topic too short"""
        invalid_request = {
            "topic": "AI",  # Too short
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        response = client.post("/api/v1/posts/generate", json=invalid_request)
        assert response.status_code == 400
        assert "at least 3 characters" in response.json()["detail"]
    
    def test_generate_batch_posts(self, client, sample_batch_request):
        """Test batch post generation"""
        response = client.post("/api/v1/posts/generate/batch", json=sample_batch_request)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert "total_processing_time" in data
        assert "successful_posts" in data
        assert "failed_posts" in data
        assert "batch_id" in data
        
        assert len(data["results"]) == 2
        assert data["successful_posts"] >= 0
        assert data["failed_posts"] >= 0
    
    def test_get_post_success(self, client):
        """Test getting a post by ID"""
        # First generate a post
        post_request = {
            "topic": "Test Post",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        generate_response = client.post("/api/v1/posts/generate", json=post_request)
        assert generate_response.status_code == 201
        
        post_id = generate_response.json()["post"]["id"]
        
        # Then get the post
        response = client.get(f"/api/v1/posts/{post_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == post_id
        assert "content" in data
        assert "status" in data
    
    def test_get_post_not_found(self, client):
        """Test getting a non-existent post"""
        response = client.get("/api/v1/posts/not-found")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_post_invalid_id(self, client):
        """Test getting a post with invalid ID"""
        response = client.get("/api/v1/posts/")
        assert response.status_code == 404  # FastAPI route not found
    
    def test_list_posts_default(self, client):
        """Test listing posts with default parameters"""
        response = client.get("/api/v1/posts")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_posts_with_filters(self, client):
        """Test listing posts with filters"""
        response = client.get(
            "/api/v1/posts",
            params={
                "skip": 0,
                "limit": 5,
                "status": "draft",
                "content_type": "educational"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
    
    def test_list_posts_invalid_filters(self, client):
        """Test listing posts with invalid filters"""
        response = client.get(
            "/api/v1/posts",
            params={
                "status": "invalid_status"
            }
        )
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
    
    def test_list_posts_invalid_pagination(self, client):
        """Test listing posts with invalid pagination"""
        response = client.get(
            "/api/v1/posts",
            params={
                "skip": -1
            }
        )
        assert response.status_code == 400
        assert "non-negative" in response.json()["detail"]
        
        response = client.get(
            "/api/v1/posts",
            params={
                "limit": 101
            }
        )
        assert response.status_code == 400
        assert "between 1 and 100" in response.json()["detail"]
    
    def test_update_post_success(self, client):
        """Test updating a post"""
        # First generate a post
        post_request = {
            "topic": "Test Post for Update",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        generate_response = client.post("/api/v1/posts/generate", json=post_request)
        assert generate_response.status_code == 201
        
        post_id = generate_response.json()["post"]["id"]
        
        # Then update the post
        update_data = {
            "content": "Updated content for the post",
            "tags": ["updated", "test"]
        }
        
        response = client.put(f"/api/v1/posts/{post_id}", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == post_id
    
    def test_update_post_not_found(self, client):
        """Test updating a non-existent post"""
        update_data = {
            "content": "Updated content"
        }
        
        response = client.put("/api/v1/posts/not-found", json=update_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_delete_post_success(self, client):
        """Test deleting a post"""
        # First generate a post
        post_request = {
            "topic": "Test Post for Deletion",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        generate_response = client.post("/api/v1/posts/generate", json=post_request)
        assert generate_response.status_code == 201
        
        post_id = generate_response.json()["post"]["id"]
        
        # Then delete the post
        response = client.delete(f"/api/v1/posts/{post_id}")
        assert response.status_code == 204
    
    def test_delete_post_not_found(self, client):
        """Test deleting a non-existent post"""
        response = client.delete("/api/v1/posts/not-found")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_optimize_post(self, client):
        """Test post optimization"""
        # First generate a post
        post_request = {
            "topic": "Test Post for Optimization",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        generate_response = client.post("/api/v1/posts/generate", json=post_request)
        assert generate_response.status_code == 201
        
        post_id = generate_response.json()["post"]["id"]
        
        # Then optimize the post
        optimization_request = {
            "optimization_level": "advanced",
            "focus_areas": ["engagement", "readability"]
        }
        
        response = client.post(f"/api/v1/posts/{post_id}/optimize", json=optimization_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "improvements" in data
        assert "processing_time" in data
    
    def test_get_metrics(self, client):
        """Test getting system metrics"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_processing_time" in data
        assert "cache_hit_rate" in data
        assert "memory_usage" in data
        assert "cpu_usage" in data
        assert "active_connections" in data
    
    def test_get_analytics(self, client):
        """Test getting post analytics"""
        # First generate a post
        post_request = {
            "topic": "Test Post for Analytics",
            "audience_type": "professionals",
            "content_type": "educational"
        }
        
        generate_response = client.post("/api/v1/posts/generate", json=post_request)
        assert generate_response.status_code == 201
        
        post_id = generate_response.json()["post"]["id"]
        
        # Then get analytics
        response = client.get(f"/api/v1/analytics/{post_id}")
        assert response.status_code == 404  # Mock implementation returns None
    
    @pytest.mark.asyncio
    async def test_async_operations(self, async_client, sample_post_request):
        """Test async operations"""
        response = await async_client.post("/api/v1/posts/generate", json=sample_post_request)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
    
    def test_error_handling(self, client):
        """Test error handling"""
        # Test with malformed JSON
        response = client.post(
            "/api/v1/posts/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.options("/api/v1/posts")
        assert response.status_code == 200
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_request_id_header(self, client):
        """Test request ID header"""
        response = client.get("/api/v1/posts")
        assert response.status_code == 200
        
        # Check for request ID header
        assert "x-request-id" in response.headers
    
    def test_process_time_header(self, client):
        """Test process time header"""
        response = client.get("/api/v1/posts")
        assert response.status_code == 200
        
        # Check for process time header
        assert "x-process-time" in response.headers
        
        # Verify it's a valid float
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0


class TestConfiguration:
    """Test configuration management"""
    
    def test_settings_validation(self):
        """Test settings validation"""
        from ..core.config import Settings
        
        # Test valid settings
        settings = Settings(
            api_title="Test API",
            api_version="1.0.0",
            debug=True
        )
        
        assert settings.api_title == "Test API"
        assert settings.api_version == "1.0.0"
        assert settings.debug is True
    
    def test_invalid_settings(self):
        """Test invalid settings validation"""
        from ..core.config import Settings
        
        # Test invalid port
        with pytest.raises(ValueError):
            Settings(port=70000)  # Port too high
        
        # Test invalid log level
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")
    
    def test_environment_validation(self):
        """Test environment validation"""
        from ..core.config import validate_environment
        
        # This should pass with default settings
        assert validate_environment() is True


class TestModels:
    """Test data models"""
    
    def test_post_request_validation(self):
        """Test PostRequest validation"""
        from ..core.models import PostRequest, AudienceType, ContentType, OptimizationLevel
        
        # Valid request
        request = PostRequest(
            topic="Test Topic",
            audience_type=AudienceType.PROFESSIONALS,
            content_type=ContentType.EDUCATIONAL,
            optimization_level=OptimizationLevel.STANDARD
        )
        
        assert request.topic == "Test Topic"
        assert request.audience_type == AudienceType.PROFESSIONALS
        assert request.content_type == ContentType.EDUCATIONAL
    
    def test_post_request_invalid_topic(self):
        """Test PostRequest with invalid topic"""
        from ..core.models import PostRequest, AudienceType, ContentType
        
        with pytest.raises(ValueError):
            PostRequest(
                topic="",  # Empty topic
                audience_type=AudienceType.PROFESSIONALS,
                content_type=ContentType.EDUCATIONAL
            )
    
    def test_facebook_post_creation(self):
        """Test FacebookPost creation"""
        from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType
        from datetime import datetime
        
        post = FacebookPost(
            id="test-id",
            content="Test content",
            status=PostStatus.DRAFT,
            content_type=ContentType.EDUCATIONAL,
            audience_type=AudienceType.PROFESSIONALS,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert post.id == "test-id"
        assert post.content == "Test content"
        assert post.status == PostStatus.DRAFT
    
    def test_post_metrics_calculation(self):
        """Test PostMetrics calculation"""
        from ..core.models import PostMetrics
        
        metrics = PostMetrics(
            engagement_score=0.8,
            quality_score=0.9,
            readability_score=0.7,
            sentiment_score=0.6,
            creativity_score=0.8,
            relevance_score=0.9
        )
        
        overall_score = metrics.overall_score
        assert 0 <= overall_score <= 1
        assert overall_score > 0.7  # Should be high with these scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])






























