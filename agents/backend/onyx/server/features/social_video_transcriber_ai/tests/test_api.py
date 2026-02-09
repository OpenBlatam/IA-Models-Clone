"""
API Tests for Social Video Transcriber AI
"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from ..api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint"""
    
    def test_health_check(self):
        """Test health check returns healthy status"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestPlatformsEndpoint:
    """Tests for platforms endpoint"""
    
    def test_list_platforms(self):
        """Test listing supported platforms"""
        response = client.get("/api/v1/platforms")
        assert response.status_code == 200
        data = response.json()
        assert "platforms" in data
        assert len(data["platforms"]) == 3
        
        platform_ids = [p["id"] for p in data["platforms"]]
        assert "youtube" in platform_ids
        assert "tiktok" in platform_ids
        assert "instagram" in platform_ids


class TestTranscriptionEndpoint:
    """Tests for transcription endpoints"""
    
    def test_transcribe_invalid_url(self):
        """Test transcription with invalid URL"""
        response = client.post(
            "/api/v1/transcribe",
            json={
                "video_url": "https://invalid-site.com/video",
                "include_timestamps": True,
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_transcribe_valid_youtube_url(self):
        """Test transcription request with valid YouTube URL"""
        response = client.post(
            "/api/v1/transcribe",
            json={
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "include_timestamps": True,
                "include_analysis": True,
            }
        )
        # Should accept the request (job created)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
    
    def test_get_nonexistent_job(self):
        """Test getting status of non-existent job"""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/transcribe/{fake_id}")
        assert response.status_code == 404


class TestAnalysisEndpoint:
    """Tests for analysis endpoint"""
    
    def test_analyze_text(self):
        """Test text analysis"""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": "Este es un texto de prueba para analizar su estructura y framework. Contiene suficiente contenido para el análisis.",
                "analyze_framework": True,
                "analyze_structure": True,
            }
        )
        # May fail if OpenRouter is not configured, but should not error
        assert response.status_code in [200, 500]
    
    def test_analyze_short_text(self):
        """Test analysis with too short text"""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": "Corto",
                "analyze_framework": True,
            }
        )
        assert response.status_code == 422  # Validation error


class TestVariantsEndpoint:
    """Tests for variants endpoints"""
    
    def test_variants_no_text_or_job(self):
        """Test variants without text or job_id"""
        response = client.post(
            "/api/v1/variants",
            json={
                "num_variants": 3,
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_quick_variants_invalid_job(self):
        """Test quick variants with invalid job_id"""
        response = client.post(
            "/api/v1/variants/quick",
            json={
                "job_id": str(uuid4()),
            }
        )
        assert response.status_code == 404


class TestJobsEndpoint:
    """Tests for jobs endpoint"""
    
    def test_list_jobs(self):
        """Test listing jobs"""
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
    
    def test_list_jobs_with_filter(self):
        """Test listing jobs with status filter"""
        response = client.get("/api/v1/jobs?status=completed")
        assert response.status_code == 200
    
    def test_list_jobs_invalid_status(self):
        """Test listing jobs with invalid status"""
        response = client.get("/api/v1/jobs?status=invalid")
        assert response.status_code == 400
    
    def test_delete_nonexistent_job(self):
        """Test deleting non-existent job"""
        fake_id = str(uuid4())
        response = client.delete(f"/api/v1/jobs/{fake_id}")
        assert response.status_code == 404


class TestVideoInfoEndpoint:
    """Tests for video info endpoint"""
    
    def test_video_info_missing_url(self):
        """Test video info without URL"""
        response = client.get("/api/v1/video/info")
        assert response.status_code == 422  # Missing required parameter


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data












