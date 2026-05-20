"""
Advanced tests for Unified AI Model
Focuses on Streaming, Agent Execution, and Concurrency
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from unified_ai_model.main import create_app

# Set environment variables
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-test-key-12345"

@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

class TestStreamingAPI:
    """Tests for Streaming API endpoints."""

    def test_chat_stream_format(self, client):
        """Test that streaming endpoint returns correct SSE format."""
        
        # Mock the chat service to return an async generator
        with patch('unified_ai_model.core.chat_service.ChatService.chat_stream') as mock_stream:
            async def mock_generator(*args, **kwargs):
                yield {"chunk": "Hello", "finish_reason": None}
                yield {"chunk": " World", "finish_reason": None}
                yield {"chunk": "", "finish_reason": "stop"}
            
            mock_stream.return_value = mock_generator()
            
            response = client.post(
                "/api/v1/chat/stream",
                json={"message": "Test stream", "stream": True}
            )
            
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            # Check SSE format
            content = response.text
            assert "data: " in content
            assert "[DONE]" in content

class TestAgentExecutionFlow:
    """Tests for Agent execution logic."""

    def test_agent_task_submission_and_status(self, client):
        """Test full flow: Create Agent -> Submit Task -> Check Status."""
        
        # 1. Create Agent
        create_resp = client.post(
            "/api/v1/agents",
            json={"name": "FlowAgent", "enable_parallel": True}
        )
        assert create_resp.status_code == 200
        agent_id = create_resp.json()["agent_id"]
        
        # 2. Submit Task
        task_resp = client.post(
            f"/api/v1/agents/{agent_id}/tasks",
            json={"description": "Do something", "priority": 10}
        )
        assert task_resp.status_code == 200
        task_id = task_resp.json()["task_id"]
        
        # 3. Check Agent Status (should have task in queue or processing)
        status_resp = client.get(f"/api/v1/agents/{agent_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        
        # Verify metrics exist
        assert "metrics" in status_data
        assert "queue_size" in status_data

    def test_agent_control_flow(self, client):
        """Test Pause/Resume/Stop flow."""
        # Create
        create_resp = client.post("/api/v1/agents", json={"name": "ControlAgent"})
        agent_id = create_resp.json()["agent_id"]
        
        # Pause
        client.post(f"/api/v1/agents/{agent_id}/pause")
        status_paused = client.get(f"/api/v1/agents/{agent_id}").json()
        assert status_paused["status"] == "paused"
        
        # Resume
        client.post(f"/api/v1/agents/{agent_id}/resume")
        status_resumed = client.get(f"/api/v1/agents/{agent_id}").json()
        assert status_resumed["status"] == "running" # or idle
        
        # Stop
        client.post(f"/api/v1/agents/{agent_id}/stop")
        status_stopped = client.get(f"/api/v1/agents/{agent_id}").json()
        assert status_stopped["status"] == "stopped" or status_stopped["status"] == "terminated"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
