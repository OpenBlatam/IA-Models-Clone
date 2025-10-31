"""
Gamma App - API Integration Tests
"""

import pytest
import asyncio
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from api.main import app
from models.database import get_db
from tests.conftest import override_get_db

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

class TestAPIIntegration:
    """Integration tests for the complete API workflow"""
    
    def test_complete_content_generation_workflow(self):
        """Test the complete workflow from content generation to export"""
        # 1. Generate content
        response = client.post("/api/content/generate", json={
            "prompt": "Create a presentation about AI",
            "content_type": "presentation",
            "style": "professional"
        })
        assert response.status_code == 200
        content_id = response.json()["id"]
        
        # 2. Get generated content
        response = client.get(f"/api/content/{content_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        
        # 3. Update content
        response = client.put(f"/api/content/{content_id}", json={
            "title": "Updated AI Presentation",
            "content": {"slides": [{"title": "AI Overview", "content": "Updated content"}]}
        })
        assert response.status_code == 200
        
        # 4. Export content
        response = client.post(f"/api/content/{content_id}/export", json={
            "format": "pptx",
            "quality": "high"
        })
        assert response.status_code == 200
        assert "download_url" in response.json()
    
    def test_collaboration_workflow(self):
        """Test real-time collaboration workflow"""
        # 1. Create content
        response = client.post("/api/content/generate", json={
            "prompt": "Create a document about machine learning",
            "content_type": "document"
        })
        content_id = response.json()["id"]
        
        # 2. Start collaboration session
        response = client.post(f"/api/collaboration/{content_id}/join", json={
            "user_id": "user1",
            "user_name": "Test User"
        })
        assert response.status_code == 200
        
        # 3. Send collaboration message
        response = client.post(f"/api/collaboration/{content_id}/message", json={
            "user_id": "user1",
            "message": "Let's add a section about neural networks",
            "action": "suggest"
        })
        assert response.status_code == 200
    
    def test_analytics_workflow(self):
        """Test analytics and metrics workflow"""
        # 1. Generate content
        response = client.post("/api/content/generate", json={
            "prompt": "Create a web page about data science",
            "content_type": "web_page"
        })
        content_id = response.json()["id"]
        
        # 2. Track content view
        response = client.post(f"/api/analytics/{content_id}/track", json={
            "event": "view",
            "user_id": "user1",
            "metadata": {"source": "web"}
        })
        assert response.status_code == 200
        
        # 3. Get analytics
        response = client.get(f"/api/analytics/{content_id}")
        assert response.status_code == 200
        assert "views" in response.json()
    
    def test_error_handling_workflow(self):
        """Test error handling across the system"""
        # 1. Invalid content generation request
        response = client.post("/api/content/generate", json={
            "prompt": "",  # Empty prompt
            "content_type": "invalid_type"
        })
        assert response.status_code == 422
        
        # 2. Non-existent content access
        response = client.get("/api/content/non-existent-id")
        assert response.status_code == 404
        
        # 3. Invalid export format
        response = client.post("/api/content/valid-id/export", json={
            "format": "invalid_format"
        })
        assert response.status_code == 422
    
    def test_performance_under_load(self):
        """Test system performance under concurrent load"""
        import concurrent.futures
        import time
        
        def generate_content():
            response = client.post("/api/content/generate", json={
                "prompt": f"Test content {time.time()}",
                "content_type": "document"
            })
            return response.status_code == 200
        
        # Test concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_content) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8
    
    def test_database_transaction_rollback(self):
        """Test database transaction rollback on errors"""
        # This would test that database transactions are properly rolled back
        # when errors occur during content generation
        pass
    
    def test_cache_invalidation(self):
        """Test cache invalidation when content is updated"""
        # 1. Generate content (should be cached)
        response = client.post("/api/content/generate", json={
            "prompt": "Test cache invalidation",
            "content_type": "document"
        })
        content_id = response.json()["id"]
        
        # 2. Get content (should come from cache)
        response1 = client.get(f"/api/content/{content_id}")
        
        # 3. Update content (should invalidate cache)
        response = client.put(f"/api/content/{content_id}", json={
            "title": "Updated Title"
        })
        
        # 4. Get content again (should not come from cache)
        response2 = client.get(f"/api/content/{content_id}")
        
        # Content should be different
        assert response1.json()["title"] != response2.json()["title"]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make many requests quickly
        for i in range(150):  # Exceed rate limit
            response = client.post("/api/content/generate", json={
                "prompt": f"Rate limit test {i}",
                "content_type": "document"
            })
            
            if response.status_code == 429:  # Rate limited
                break
        
        # Should eventually get rate limited
        assert response.status_code == 429
    
    def test_security_headers(self):
        """Test that security headers are properly set"""
        response = client.get("/health")
        
        # Check for security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers
    
    def test_cors_headers(self):
        """Test CORS headers for cross-origin requests"""
        response = client.options("/api/content/generate", headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST"
        })
        
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

























