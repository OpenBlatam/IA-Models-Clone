"""
Tests for blog posts API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from ..models.schemas import PostStatus, PostCategory


class TestBlogPostsAPI:
    """Test class for blog posts API endpoints."""
    
    def test_create_blog_post(self, client: TestClient, sample_blog_post_data: dict):
        """Test creating a blog post."""
        response = client.post("/api/v1/blog-posts/", json=sample_blog_post_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == sample_blog_post_data["title"]
        assert data["content"] == sample_blog_post_data["content"]
        assert data["category"] == sample_blog_post_data["category"]
        assert "id" in data
        assert "uuid" in data
        assert "slug" in data
        assert "created_at" in data
    
    def test_create_blog_post_validation_error(self, client: TestClient):
        """Test blog post creation with validation errors."""
        invalid_data = {
            "title": "",  # Empty title should fail
            "content": "Valid content"
        }
        
        response = client.post("/api/v1/blog-posts/", json=invalid_data)
        assert response.status_code == 422
    
    def test_get_blog_post(self, client: TestClient, sample_blog_post: dict):
        """Test getting a blog post by ID."""
        post_id = sample_blog_post["id"]
        response = client.get(f"/api/v1/blog-posts/{post_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == post_id
        assert data["title"] == sample_blog_post["title"]
    
    def test_get_blog_post_not_found(self, client: TestClient):
        """Test getting a non-existent blog post."""
        response = client.get("/api/v1/blog-posts/99999")
        assert response.status_code == 404
    
    def test_get_blog_post_by_slug(self, client: TestClient, sample_blog_post: dict):
        """Test getting a blog post by slug."""
        slug = sample_blog_post["slug"]
        response = client.get(f"/api/v1/blog-posts/slug/{slug}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["slug"] == slug
        assert data["title"] == sample_blog_post["title"]
    
    def test_list_blog_posts(self, client: TestClient):
        """Test listing blog posts with pagination."""
        response = client.get("/api/v1/blog-posts/?page=1&size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert "pages" in data
        assert isinstance(data["items"], list)
    
    def test_list_blog_posts_with_filters(self, client: TestClient):
        """Test listing blog posts with filters."""
        response = client.get("/api/v1/blog-posts/?category=technology&status=published")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_search_blog_posts(self, client: TestClient):
        """Test searching blog posts."""
        response = client.get("/api/v1/blog-posts/search?query=test&page=1&size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_update_blog_post(self, client: TestClient, sample_blog_post: dict):
        """Test updating a blog post."""
        post_id = sample_blog_post["id"]
        update_data = {
            "title": "Updated Title",
            "content": "Updated content with more meaningful text."
        }
        
        response = client.put(f"/api/v1/blog-posts/{post_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == update_data["title"]
        assert data["content"] == update_data["content"]
    
    def test_delete_blog_post(self, client: TestClient, sample_blog_post: dict):
        """Test deleting a blog post."""
        post_id = sample_blog_post["id"]
        response = client.delete(f"/api/v1/blog-posts/{post_id}")
        
        assert response.status_code == 204
        
        # Verify the post is deleted
        get_response = client.get(f"/api/v1/blog-posts/{post_id}")
        assert get_response.status_code == 404
    
    def test_increment_view_count(self, client: TestClient, sample_blog_post: dict):
        """Test incrementing view count."""
        post_id = sample_blog_post["id"]
        response = client.post(f"/api/v1/blog-posts/{post_id}/view")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


@pytest.mark.asyncio
class TestBlogPostsAsync:
    """Async tests for blog posts API."""
    
    async def test_async_blog_post_operations(self, async_client: AsyncClient, sample_blog_post_data: dict):
        """Test async blog post operations."""
        # Create blog post
        response = await async_client.post("/api/v1/blog-posts/", json=sample_blog_post_data)
        assert response.status_code == 201
        
        post_data = response.json()
        post_id = post_data["id"]
        
        # Get blog post
        response = await async_client.get(f"/api/v1/blog-posts/{post_id}")
        assert response.status_code == 200
        
        # Update blog post
        update_data = {"title": "Async Updated Title"}
        response = await async_client.put(f"/api/v1/blog-posts/{post_id}", json=update_data)
        assert response.status_code == 200
        
        # Delete blog post
        response = await async_client.delete(f"/api/v1/blog-posts/{post_id}")
        assert response.status_code == 204






























