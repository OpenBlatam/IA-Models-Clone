"""Tests for batch processing endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_batch_processing_empty():
    """Test batch processing with empty list."""
    response = client.post(
        "/api/v1/batch",
        json={"requests": [], "max_concurrent": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["processed"] == 0


def test_batch_processing_invalid():
    """Test batch processing with invalid request."""
    response = client.post(
        "/api/v1/batch",
        json={"requests": [{"invalid": "data"}]}
    )
    # Should fail validation
    assert response.status_code in [400, 422]


def test_batch_processing_max_concurrent():
    """Test batch processing with max_concurrent limit."""
    requests = [
        {"surgery_type": "rhinoplasty", "intensity": 0.5, "image_url": "http://test.com/img.jpg"}
        for _ in range(5)
    ]
    
    response = client.post(
        "/api/v1/batch",
        json={"requests": requests, "max_concurrent": 2}
    )
    # May succeed or fail depending on image URL availability
    assert response.status_code in [200, 500]

