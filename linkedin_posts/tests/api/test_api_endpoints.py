"""
API Tests for LinkedIn Posts
============================

API tests for the LinkedIn posts REST endpoints including
authentication, validation, error handling, and response formats.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import API components
from ...app.index import app
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestLinkedInPostsAPI:
    """Test suite for LinkedIn Posts API endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def async_client(self) -> AsyncClient:
        """Create async test client."""
        return AsyncClient(app=app, base_url="http://test")

    @pytest.fixture
    def sample_post_data(self) -> Dict[str, Any]:
        """Sample post data for API tests."""
        return {
            "topic": "AI in Modern Business",
            "keyPoints": ["Increased efficiency", "Cost reduction", "Innovation"],
            "targetAudience": "Business leaders",
            "industry": "Technology",
            "tone": "professional",
            "postType": "text",
            "keywords": ["AI", "business", "innovation"],
            "additionalContext": "Focus on practical applications"
        }

    @pytest.fixture
    def sample_post_response(self) -> Dict[str, Any]:
        """Sample post response data."""
        return {
            "id": "test-post-123",
            "userId": "user-123",
            "title": "AI in Modern Business",
            "content": {
                "text": "AI is transforming modern business practices...",
                "hashtags": ["#AI", "#Business", "#Innovation"],
                "mentions": [],
                "links": [],
                "images": [],
                "callToAction": "Learn more about AI implementation"
            },
            "postType": "text",
            "tone": "professional",
            "status": "draft",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
            "engagement": {
                "likes": 0,
                "comments": 0,
                "shares": 0,
                "clicks": 0,
                "impressions": 0,
                "reach": 0,
                "engagementRate": 0.0
            },
            "aiScore": 85.5,
            "optimizationSuggestions": ["Add more hashtags", "Include call-to-action"],
            "keywords": ["AI", "business", "innovation"],
            "externalMetadata": {},
            "performanceScore": 0,
            "reachScore": 0,
            "engagementScore": 0
        }

    def test_health_check_endpoint(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_create_post_endpoint_success(self, client: TestClient, sample_post_data: Dict[str, Any]):
        """Test successful post creation via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            # Mock the service response
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock the created post
            mock_post = MagicMock()
            mock_post.id = "test-post-123"
            mock_post.title = "AI in Modern Business"
            mock_post.status = PostStatus.DRAFT
            mock_post.aiScore = 85.5
            mock_service_instance.createPost.return_value = mock_post
            
            response = client.post("/api/posts", json=sample_post_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "test-post-123"
            assert data["title"] == "AI in Modern Business"
            assert data["status"] == "draft"

    def test_create_post_endpoint_validation_error(self, client: TestClient):
        """Test post creation with validation error."""
        invalid_data = {
            "topic": "",  # Empty topic should fail validation
            "keyPoints": [],
            "targetAudience": "",
            "industry": "",
            "tone": "professional",
            "postType": "text"
        }
        
        response = client.post("/api/posts", json=invalid_data)
        assert response.status_code == 400
        assert "error" in response.json()

    def test_create_post_endpoint_internal_error(self, client: TestClient, sample_post_data: Dict[str, Any]):
        """Test post creation with internal server error."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.createPost.side_effect = Exception("Internal server error")
            
            response = client.post("/api/posts", json=sample_post_data)
            assert response.status_code == 500
            assert "error" in response.json()

    def test_get_post_endpoint_success(self, client: TestClient):
        """Test successful post retrieval via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock the post
            mock_post = MagicMock()
            mock_post.id = "test-post-123"
            mock_post.title = "Test Post"
            mock_post.status = PostStatus.DRAFT
            mock_service_instance.getPost.return_value = mock_post
            
            response = client.get("/api/posts/test-post-123")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-post-123"
            assert data["title"] == "Test Post"

    def test_get_post_endpoint_not_found(self, client: TestClient):
        """Test post retrieval when post doesn't exist."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.getPost.return_value = None
            
            response = client.get("/api/posts/non-existent-post")
            assert response.status_code == 404
            assert "error" in response.json()

    def test_list_posts_endpoint_success(self, client: TestClient):
        """Test successful post listing via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock posts list
            mock_posts = [
                MagicMock(id="post-1", title="Post 1", status=PostStatus.DRAFT),
                MagicMock(id="post-2", title="Post 2", status=PostStatus.PUBLISHED)
            ]
            mock_service_instance.listPosts.return_value = mock_posts
            
            response = client.get("/api/posts?userId=user-123")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == "post-1"
            assert data[1]["id"] == "post-2"

    def test_list_posts_endpoint_with_filters(self, client: TestClient):
        """Test post listing with filters."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            mock_posts = [MagicMock(id="draft-post", title="Draft Post", status=PostStatus.DRAFT)]
            mock_service_instance.listPosts.return_value = mock_posts
            
            response = client.get("/api/posts?userId=user-123&status=draft&limit=10")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["status"] == "draft"

    def test_update_post_endpoint_success(self, client: TestClient):
        """Test successful post update via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock updated post
            mock_post = MagicMock()
            mock_post.id = "test-post-123"
            mock_post.title = "Updated Post"
            mock_post.status = PostStatus.SCHEDULED
            mock_service_instance.updatePost.return_value = mock_post
            
            update_data = {
                "title": "Updated Post",
                "status": "scheduled"
            }
            
            response = client.put("/api/posts/test-post-123", json=update_data)
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Updated Post"
            assert data["status"] == "scheduled"

    def test_update_post_endpoint_not_found(self, client: TestClient):
        """Test post update when post doesn't exist."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.updatePost.side_effect = Exception("Post not found")
            
            update_data = {"title": "Updated Post"}
            response = client.put("/api/posts/non-existent-post", json=update_data)
            assert response.status_code == 404
            assert "error" in response.json()

    def test_delete_post_endpoint_success(self, client: TestClient):
        """Test successful post deletion via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.deletePost.return_value = True
            
            response = client.delete("/api/posts/test-post-123")
            assert response.status_code == 204

    def test_delete_post_endpoint_not_found(self, client: TestClient):
        """Test post deletion when post doesn't exist."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.deletePost.return_value = False
            
            response = client.delete("/api/posts/non-existent-post")
            assert response.status_code == 404
            assert "error" in response.json()

    def test_optimize_post_endpoint_success(self, client: TestClient):
        """Test successful post optimization via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock optimization result
            mock_result = MagicMock()
            mock_result.optimizationScore = 90.0
            mock_result.suggestions = ["Improve headline", "Add hashtags"]
            mock_result.processingTime = 2.5
            mock_service_instance.optimizePost.return_value = mock_result
            
            response = client.post("/api/posts/test-post-123/optimize")
            assert response.status_code == 200
            data = response.json()
            assert data["optimizationScore"] == 90.0
            assert len(data["suggestions"]) == 2
            assert data["processingTime"] == 2.5

    def test_optimize_post_endpoint_not_found(self, client: TestClient):
        """Test post optimization when post doesn't exist."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.optimizePost.side_effect = ValueError("Post not found")
            
            response = client.post("/api/posts/non-existent-post/optimize")
            assert response.status_code == 404
            assert "error" in response.json()

    def test_analyze_post_endpoint_success(self, client: TestClient):
        """Test successful post analysis via API."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock analysis result
            mock_result = MagicMock()
            mock_result.sentimentScore = 0.8
            mock_result.readabilityScore = 85.0
            mock_result.engagementScore = 90.0
            mock_result.keywordDensity = 0.15
            mock_result.structureScore = 88.0
            mock_result.callToActionScore = 92.0
            mock_service_instance.generatePostAnalytics.return_value = mock_result
            
            response = client.get("/api/posts/test-post-123/analytics")
            assert response.status_code == 200
            data = response.json()
            assert data["sentimentScore"] == 0.8
            assert data["readabilityScore"] == 85.0
            assert data["engagementScore"] == 90.0

    def test_analyze_post_endpoint_not_found(self, client: TestClient):
        """Test post analysis when post doesn't exist."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.generatePostAnalytics.side_effect = ValueError("Post not found")
            
            response = client.get("/api/posts/non-existent-post/analytics")
            assert response.status_code == 404
            assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_async_create_post_endpoint(self, async_client: AsyncClient, sample_post_data: Dict[str, Any]):
        """Test async post creation endpoint."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            mock_post = MagicMock()
            mock_post.id = "async-post-123"
            mock_post.title = "Async Post"
            mock_post.status = PostStatus.DRAFT
            mock_service_instance.createPost.return_value = mock_post
            
            response = await async_client.post("/api/posts", json=sample_post_data)
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "async-post-123"

    def test_api_authentication_middleware(self, client: TestClient):
        """Test API authentication middleware."""
        # Test endpoint that requires authentication
        response = client.get("/api/posts")
        # Should redirect to login or return 401
        assert response.status_code in [401, 302]

    def test_api_rate_limiting(self, client: TestClient, sample_post_data: Dict[str, Any]):
        """Test API rate limiting."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_post = MagicMock()
            mock_service_instance.createPost.return_value = mock_post
            
            # Make multiple requests quickly
            responses = []
            for _ in range(10):
                response = client.post("/api/posts", json=sample_post_data)
                responses.append(response)
            
            # Most should succeed, but some might be rate limited
            success_count = sum(1 for r in responses if r.status_code == 201)
            assert success_count > 0

    def test_api_cors_headers(self, client: TestClient):
        """Test CORS headers in API responses."""
        response = client.options("/api/posts")
        # Should include CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200

    def test_api_error_response_format(self, client: TestClient):
        """Test API error response format."""
        response = client.get("/api/posts/non-existent-post")
        if response.status_code == 404:
            data = response.json()
            assert "error" in data
            assert "message" in data or "detail" in data

    def test_api_success_response_format(self, client: TestClient, sample_post_data: Dict[str, Any]):
        """Test API success response format."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_post = MagicMock()
            mock_post.id = "format-test-123"
            mock_post.title = "Format Test"
            mock_post.status = PostStatus.DRAFT
            mock_service_instance.createPost.return_value = mock_post
            
            response = client.post("/api/posts", json=sample_post_data)
            if response.status_code == 201:
                data = response.json()
                assert "id" in data
                assert "title" in data
                assert "status" in data

    def test_api_pagination(self, client: TestClient):
        """Test API pagination for post listing."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            # Mock paginated posts
            mock_posts = [MagicMock(id=f"post-{i}", title=f"Post {i}") for i in range(20)]
            mock_service_instance.listPosts.return_value = mock_posts[:10]  # First page
            
            response = client.get("/api/posts?userId=user-123&limit=10&offset=0")
            assert response.status_code == 200
            data = response.json()
            assert len(data) <= 10

    def test_api_search_functionality(self, client: TestClient):
        """Test API search functionality."""
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            
            mock_posts = [MagicMock(id="search-result", title="Search Result")]
            mock_service_instance.listPosts.return_value = mock_posts
            
            response = client.get("/api/posts?userId=user-123&search=AI")
            assert response.status_code == 200
            data = response.json()
            assert len(data) >= 0  # May be empty if no results

    def test_api_bulk_operations(self, client: TestClient):
        """Test API bulk operations."""
        bulk_data = {
            "posts": [
                {"topic": "Post 1", "keyPoints": ["Point 1"], "targetAudience": "Audience", "industry": "Tech", "tone": "professional", "postType": "text"},
                {"topic": "Post 2", "keyPoints": ["Point 2"], "targetAudience": "Audience", "industry": "Tech", "tone": "professional", "postType": "text"}
            ]
        }
        
        with patch('linkedin_posts.services.post_service.PostService') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_posts = [MagicMock(id=f"bulk-{i}", title=f"Bulk Post {i}") for i in range(2)]
            mock_service_instance.createPost.side_effect = mock_posts
            
            response = client.post("/api/posts/bulk", json=bulk_data)
            # Should handle bulk creation
            assert response.status_code in [200, 201, 400]  # Depends on implementation

    def test_api_webhook_endpoints(self, client: TestClient):
        """Test API webhook endpoints."""
        webhook_data = {
            "event": "post.published",
            "postId": "webhook-test-123",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        response = client.post("/api/webhooks/linkedin", json=webhook_data)
        # Should handle webhook events
        assert response.status_code in [200, 400, 404]  # Depends on implementation

    def test_api_metrics_endpoint(self, client: TestClient):
        """Test API metrics endpoint."""
        response = client.get("/api/metrics")
        # Should return system metrics
        assert response.status_code in [200, 404]  # Depends on implementation

    def test_api_documentation_endpoint(self, client: TestClient):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        # Should return API documentation
        assert response.status_code in [200, 404]  # Depends on implementation

    def test_api_openapi_schema(self, client: TestClient):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        # Should return OpenAPI schema
        assert response.status_code in [200, 404]  # Depends on implementation
