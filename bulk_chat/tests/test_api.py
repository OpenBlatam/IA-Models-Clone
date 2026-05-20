"""
Tests for Chat API
==================
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json


@pytest.fixture
def mock_chat_engine():
    """Mock chat engine for testing."""
    engine = MagicMock()
    engine.create_session = AsyncMock()
    engine.get_session = AsyncMock()
    engine.pause_session = AsyncMock()
    engine.resume_session = AsyncMock()
    engine.stop_session = AsyncMock()
    engine.add_user_message = AsyncMock()
    return engine


@pytest.fixture
def client(mock_chat_engine):
    """Create test client."""
    # Import here to avoid circular imports
    from ..api.chat_api import create_chat_app
    
    with patch('bulk_chat.api.chat_api.ContinuousChatEngine', return_value=mock_chat_engine):
        app = create_chat_app()
        return TestClient(app)


@pytest.mark.asyncio
async def test_create_session_endpoint(client, mock_chat_engine):
    """Test POST /api/v1/chat/sessions endpoint."""
    from ..core.chat_session import ChatSession, ChatState
    
    # Mock session
    mock_session = ChatSession(
        session_id="test_session_123",
        user_id="test_user",
        state=ChatState.ACTIVE
    )
    mock_chat_engine.create_session.return_value = mock_session
    
    response = client.post(
        "/api/v1/chat/sessions",
        json={"user_id": "test_user", "initial_message": "Hello"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_123"
    assert data["user_id"] == "test_user"


@pytest.mark.asyncio
async def test_get_session_endpoint(client, mock_chat_engine):
    """Test GET /api/v1/chat/sessions/{session_id} endpoint."""
    from ..core.chat_session import ChatSession, ChatState
    
    mock_session = ChatSession(
        session_id="test_session_123",
        user_id="test_user",
        state=ChatState.ACTIVE
    )
    mock_chat_engine.get_session.return_value = mock_session
    
    response = client.get("/api/v1/chat/sessions/test_session_123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_123"


@pytest.mark.asyncio
async def test_pause_session_endpoint(client, mock_chat_engine):
    """Test POST /api/v1/chat/sessions/{session_id}/pause endpoint."""
    response = client.post(
        "/api/v1/chat/sessions/test_session_123/pause",
        json={"reason": "Testing pause"}
    )
    
    assert response.status_code == 200
    mock_chat_engine.pause_session.assert_called_once()


@pytest.mark.asyncio
async def test_resume_session_endpoint(client, mock_chat_engine):
    """Test POST /api/v1/chat/sessions/{session_id}/resume endpoint."""
    response = client.post("/api/v1/chat/sessions/test_session_123/resume")
    
    assert response.status_code == 200
    mock_chat_engine.resume_session.assert_called_once()


@pytest.mark.asyncio
async def test_stop_session_endpoint(client, mock_chat_engine):
    """Test POST /api/v1/chat/sessions/{session_id}/stop endpoint."""
    response = client.post("/api/v1/chat/sessions/test_session_123/stop")
    
    assert response.status_code == 200
    mock_chat_engine.stop_session.assert_called_once()


@pytest.mark.asyncio
async def test_add_message_endpoint(client, mock_chat_engine):
    """Test POST /api/v1/chat/sessions/{session_id}/messages endpoint."""
    response = client.post(
        "/api/v1/chat/sessions/test_session_123/messages",
        json={"content": "Hello, world!"}
    )
    
    assert response.status_code == 200
    mock_chat_engine.add_user_message.assert_called_once()


@pytest.mark.asyncio
async def test_session_not_found(client, mock_chat_engine):
    """Test error handling for non-existent session."""
    mock_chat_engine.get_session.side_effect = ValueError("Session not found")
    
    response = client.get("/api/v1/chat/sessions/non_existent")
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test GET /health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test GET /api/v1/metrics endpoint."""
    response = client.get("/api/v1/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


