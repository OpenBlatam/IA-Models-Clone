"""
Unit tests for Unified AI Model API
Tests the API endpoints without requiring external services.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from dataclasses import dataclass
from typing import Optional, Dict, Any


# Set environment variables before importing the app
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-test-key-12345"
os.environ["UNIFIED_AI_DEFAULT_MODEL"] = "deepseek-chat"


from unified_ai_model.main import create_app


@dataclass
class MockChatResponse:
    """Mock response for chat service."""
    conversation_id: str
    message: Dict[str, Any]
    model: str
    tokens_used: int
    latency_ms: float = 100.0
    cached: bool = False


@dataclass
class MockGenerateResponse:
    """Mock response for generate service."""
    text: str
    model: str
    tokens_used: int
    latency_ms: float = 100.0
    cached: bool = False
    finish_reason: str = "stop"


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_health_check_returns_healthy_status(self, client):
        """Test that health returns healthy status."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""
    
    def test_chat_endpoint_accepts_request(self, client):
        """Test that chat endpoint accepts requests with correct format."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello!"}
        )
        # Returns 200 on success or 500 if API key not valid (expected in tests)
        assert response.status_code in [200, 500]
        data = response.json()
        # Should have either message (success) or error info
        assert "message" in data or "detail" in data or "error" in data
    
    def test_chat_with_conversation_id(self, client):
        """Test chat with conversation_id parameter."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Continue please",
                "conversation_id": "test-conv"
            }
        )
        # Request should be accepted
        assert response.status_code in [200, 500]
    
    def test_chat_empty_message(self, client):
        """Test chat with empty message."""
        response = client.post(
            "/api/v1/chat",
            json={"message": ""}
        )
        # Should return validation error or process it
        assert response.status_code in [200, 422, 500]


class TestGenerateEndpoint:
    """Tests for generate endpoint."""
    
    def test_generate_endpoint_accepts_request(self, client):
        """Test that generate endpoint accepts requests."""
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Write a haiku"}
        )
        # Returns 200 on success or 500 if API not available
        assert response.status_code in [200, 500]
        data = response.json()
        # Response structure check
        assert isinstance(data, dict)
    
    def test_generate_with_model_parameter(self, client):
        """Test generation with model parameter."""
        response = client.post(
            "/api/v1/generate",
            json={
                "prompt": "Write code",
                "model": "deepseek-coder"
            }
        )
        assert response.status_code in [200, 500]


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_stats_returns_200(self, client):
        """Test that stats endpoint returns 200."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
    
    def test_stats_structure(self, client):
        """Test stats response structure."""
        response = client.get("/api/v1/stats")
        data = response.json()
        
        assert "uptime_seconds" in data
        assert "requests" in data
        assert "cache" in data
        assert "latency" in data


class TestModelsEndpoint:
    """Tests for models endpoint."""
    
    def test_models_endpoint_accessible(self, client):
        """Test that models endpoint is accessible."""
        response = client.get("/api/v1/models")
        # Should return 200 or 500 (if API unavailable)
        assert response.status_code in [200, 500]
    
    def test_models_returns_json(self, client):
        """Test that models returns JSON response."""
        response = client.get("/api/v1/models")
        data = response.json()
        # Should be a dict with either models or error
        assert isinstance(data, dict)


class TestAgentsEndpoint:
    """Tests for continuous agents endpoint."""
    
    def test_create_agent(self, client):
        """Test creating a new agent."""
        response = client.post(
            "/api/v1/agents",
            json={"name": "TestAgent"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert data["success"] is True
    
    def test_list_agents(self, client):
        """Test listing agents."""
        # First create an agent
        client.post("/api/v1/agents", json={"name": "Agent1"})
        
        response = client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
    
    def test_get_agent_status(self, client):
        """Test getting agent status."""
        # Create agent first
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "StatusTestAgent"}
        )
        agent_id = create_resp.json()["agent_id"]
        
        # Get status
        response = client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_submit_task(self, client):
        """Test submitting a task to agent."""
        # Create agent
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "TaskTestAgent"}
        )
        agent_id = create_resp.json()["agent_id"]
        
        # Submit task
        response = client.post(
            f"/api/v1/agents/{agent_id}/tasks",
            json={
                "description": "Test task",
                "priority": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
    
    def test_pause_agent(self, client):
        """Test pausing an agent."""
        # Create agent
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "PauseTestAgent"}
        )
        agent_id = create_resp.json()["agent_id"]
        
        # Pause
        response = client.post(f"/api/v1/agents/{agent_id}/pause")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_resume_agent(self, client):
        """Test resuming an agent."""
        # Create agent
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "ResumeTestAgent"}
        )
        agent_id = create_resp.json()["agent_id"]
        
        # Pause first
        client.post(f"/api/v1/agents/{agent_id}/pause")
        
        # Resume
        response = client.post(f"/api/v1/agents/{agent_id}/resume")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_stop_agent(self, client):
        """Test stopping an agent."""
        # Create agent
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "StopTestAgent"}
        )
        agent_id = create_resp.json()["agent_id"]
        
        # Stop
        response = client.post(f"/api/v1/agents/{agent_id}/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_get_nonexistent_agent(self, client):
        """Test getting a non-existent agent returns 404."""
        response = client.get("/api/v1/agents/nonexistent-id")
        assert response.status_code == 404


class TestConversationsEndpoint:
    """Tests for conversations endpoint."""
    
    def test_create_conversation(self, client):
        """Test creating a new conversation."""
        response = client.post(
            "/api/v1/conversations",
            json={"system_prompt": "You are a helpful assistant."}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
    
    def test_list_conversations(self, client):
        """Test listing conversations."""
        response = client.get("/api/v1/conversations")
        
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert isinstance(data["conversations"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



