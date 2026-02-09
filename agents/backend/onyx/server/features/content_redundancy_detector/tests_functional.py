"""
Functional tests for the refactored system
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_analyze_content():
    """Test content analysis endpoint"""
    test_content = "This is a test text for analysis."
    response = client.post("/analyze", json={"content": test_content})
    assert response.status_code == 200
    data = response.json()
    assert "redundancy_score" in data
    assert "word_count" in data


def test_similarity_check():
    """Test similarity check endpoint"""
    response = client.post("/similarity", json={
        "text1": "First text",
        "text2": "Second text",
        "threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data


def test_quality_assessment():
    """Test quality assessment endpoint"""
    test_content = "Simple text for quality assessment."
    response = client.post("/quality", json={"content": test_content})
    assert response.status_code == 200
    data = response.json()
    assert "readability_score" in data
    assert "quality_rating" in data


def test_stats_endpoint():
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_endpoints" in data
    assert "features" in data


def test_validation_errors():
    """Test input validation"""
    # Empty content
    response = client.post("/analyze", json={"content": ""})
    assert response.status_code == 422
    
    # Missing text
    response = client.post("/similarity", json={"text1": "text", "text2": ""})
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



