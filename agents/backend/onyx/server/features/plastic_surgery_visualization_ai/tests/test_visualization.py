"""Tests for visualization endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from PIL import Image
import io

from main import app

client = TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (500, 500), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


def test_get_surgery_types():
    """Test getting available surgery types."""
    response = client.get("/api/v1/surgery-types")
    assert response.status_code == 200
    data = response.json()
    assert "surgery_types" in data
    assert len(data["surgery_types"]) > 0


def test_create_visualization_missing_data():
    """Test creating visualization without image data."""
    payload = {
        "surgery_type": "rhinoplasty",
        "intensity": 0.7
    }
    response = client.post("/api/v1/visualize", json=payload)
    # Should fail validation or return error
    assert response.status_code in [400, 422, 500]


@patch('services.visualization_service.VisualizationService.create_visualization')
def test_create_visualization_success(mock_create):
    """Test successful visualization creation."""
    from api.schemas.visualization import VisualizationResponse, SurgeryType
    from datetime import datetime
    
    mock_response = VisualizationResponse(
        visualization_id="test-id",
        image_url="/api/v1/visualize/test-id",
        surgery_type=SurgeryType.RHINOPLASTY,
        intensity=0.7,
        created_at=datetime.utcnow().isoformat(),
        processing_time=1.5
    )
    mock_create.return_value = mock_response
    
    payload = {
        "surgery_type": "rhinoplasty",
        "intensity": 0.7,
        "image_url": "https://example.com/image.jpg"
    }
    
    response = client.post("/api/v1/visualize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "visualization_id" in data
    assert data["surgery_type"] == "rhinoplasty"


def test_get_nonexistent_visualization():
    """Test getting a visualization that doesn't exist."""
    response = client.get("/api/v1/visualize/nonexistent-id")
    assert response.status_code in [404, 500]


def test_visualization_intensity_validation():
    """Test intensity validation."""
    payload = {
        "surgery_type": "rhinoplasty",
        "intensity": 1.5,  # Invalid: should be 0.0-1.0
        "image_url": "https://example.com/image.jpg"
    }
    response = client.post("/api/v1/visualize", json=payload)
    assert response.status_code == 422  # Validation error

