"""
Tests for API Routes
====================

Unit tests for API route handlers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from ..api.enhancer_api import app
from ..core.enhancer_agent import EnhancerAgent
from ..config.enhancer_config import EnhancerConfig


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    agent = Mock(spec=EnhancerAgent)
    agent.enhance_image = AsyncMock(return_value="task_123")
    agent.enhance_video = AsyncMock(return_value="task_456")
    agent.get_task_status = AsyncMock(return_value={"status": "completed"})
    agent.get_task_result = AsyncMock(return_value={"result": "enhanced"})
    agent.get_stats = Mock(return_value={"total_tasks": 10})
    agent.output_dirs = {"uploads": "/tmp/uploads"}
    agent.config = Mock()
    agent.config.max_file_size_mb = 100
    agent.config.allowed_image_formats = [".jpg", ".png"]
    agent.config.allowed_video_formats = [".mp4"]
    agent.video_processor = Mock()
    agent.video_processor.analyze_video = AsyncMock(return_value={"fps": 30})
    return agent


@pytest.mark.asyncio
async def test_get_task_status(client, mock_agent):
    """Test getting task status."""
    with patch('imagen_video_enhancer_ai.api.routes.task_routes.get_agent', return_value=mock_agent):
        response = client.get("/task/task_123/status")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_get_task_result(client, mock_agent):
    """Test getting task result."""
    with patch('imagen_video_enhancer_ai.api.routes.task_routes.get_agent', return_value=mock_agent):
        response = client.get("/task/task_123/result")
        assert response.status_code == 200
        assert response.json()["result"] == "enhanced"


@pytest.mark.asyncio
async def test_get_stats(client, mock_agent):
    """Test getting stats."""
    with patch('imagen_video_enhancer_ai.api.routes.monitoring_routes.get_agent', return_value=mock_agent):
        response = client.get("/stats")
        assert response.status_code == 200
        assert response.json()["total_tasks"] == 10


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_enhance_image_request_model():
    """Test EnhanceImageRequest model."""
    from ..api.models import EnhanceImageRequest
    
    request = EnhanceImageRequest(
        file_path="/path/to/image.jpg",
        enhancement_type="general",
        priority=5
    )
    
    assert request.file_path == "/path/to/image.jpg"
    assert request.enhancement_type == "general"
    assert request.priority == 5


def test_batch_process_request_model():
    """Test BatchProcessRequest model."""
    from ..api.models import BatchProcessRequest
    
    request = BatchProcessRequest(
        items=[
            {"file_path": "/path/to/image.jpg", "service_type": "enhance_image"}
        ]
    )
    
    assert len(request.items) == 1
    assert request.items[0]["file_path"] == "/path/to/image.jpg"




