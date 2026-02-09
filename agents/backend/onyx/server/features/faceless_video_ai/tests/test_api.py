"""
Integration tests for API
"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from ..api.main import app

client = TestClient(app)


class TestVideoGenerationAPI:
    """Tests for video generation endpoints"""
    
    def test_generate_video_endpoint(self):
        """Test video generation endpoint"""
        response = client.post(
            "/api/v1/generate",
            json={
                "script": {
                    "text": "Test video script",
                    "language": "es"
                }
            }
        )
        assert response.status_code in [200, 202]
        assert "video_id" in response.json()
    
    def test_get_video_status(self):
        """Test get video status"""
        # First create a video
        create_response = client.post(
            "/api/v1/generate",
            json={
                "script": {
                    "text": "Test",
                    "language": "es"
                }
            }
        )
        video_id = create_response.json()["video_id"]
        
        # Then check status
        response = client.get(f"/api/v1/videos/{video_id}/status")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_invalid_request(self):
        """Test invalid request handling"""
        response = client.post(
            "/api/v1/generate",
            json={
                "script": {
                    "text": "",  # Empty text
                    "language": "es"
                }
            }
        )
        assert response.status_code in [400, 422]


class TestHealthEndpoints:
    """Tests for health check endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/admin/health")
        assert response.status_code == 200
        assert "overall_status" in response.json()
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/api/v1/admin/metrics")
        assert response.status_code == 200


class TestRecommendationsAPI:
    """Tests for recommendations endpoint"""
    
    def test_get_recommendations(self):
        """Test recommendations endpoint"""
        response = client.get(
            "/api/v1/recommendations",
            params={
                "script_text": "Test script for marketing video",
                "platform": "youtube",
                "content_type": "marketing"
            }
        )
        assert response.status_code == 200
        assert "video_style" in response.json()
        assert "voice" in response.json()

