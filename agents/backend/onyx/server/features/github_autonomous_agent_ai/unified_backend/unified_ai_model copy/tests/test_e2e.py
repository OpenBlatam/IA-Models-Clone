"""
End-to-end tests for Unified AI Model API
Tests the full flow with real API calls to DeepSeek/OpenRouter.

Run with:
    $env:DEEPSEEK_API_KEY="sk-xxx"; python -m pytest unified_ai_model/tests/test_e2e.py -v

Note: These tests require a valid API key and will make real API calls.
"""

import os
import time
import pytest
from fastapi.testclient import TestClient

# Check if we have a real API key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HAS_API_KEY = bool(DEEPSEEK_API_KEY or OPENROUTER_API_KEY)

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not HAS_API_KEY or DEEPSEEK_API_KEY == "sk-test-key-12345",
    reason="No valid API key configured. Set DEEPSEEK_API_KEY or OPENROUTER_API_KEY."
)

from unified_ai_model.main import create_app


@pytest.fixture(scope="module")
def client():
    """Create test client for the entire module."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestE2EHealth:
    """E2E tests for health endpoints."""
    
    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        print(f"\n  Health: {data}")


class TestE2EChat:
    """E2E tests for chat functionality."""
    
    def test_simple_chat(self, client):
        """Test a simple chat message with real API."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Responde solo con: Hola"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert len(data["message"]["content"]) > 0
        print(f"\n  Chat response: {data['message']['content'][:100]}...")
    
    def test_chat_with_conversation_memory(self, client):
        """Test chat maintains conversation context."""
        # First message
        response1 = client.post(
            "/api/v1/chat",
            json={"message": "Mi nombre es Carlos. Recuerdalo."}
        )
        assert response1.status_code == 200
        conv_id = response1.json()["conversation_id"]
        
        # Second message using same conversation
        response2 = client.post(
            "/api/v1/chat",
            json={
                "message": "Como me llamo?",
                "conversation_id": conv_id
            }
        )
        assert response2.status_code == 200
        content = response2.json()["message"]["content"].lower()
        # Should remember the name
        assert "carlos" in content
        # Remove emojis for Windows console compatibility
        safe_content = content.encode('ascii', 'ignore').decode('ascii')
        print(f"\n  Memory test passed - AI remembered: {safe_content[:100]}...")
    
    def test_chat_with_system_prompt(self, client):
        """Test chat with custom system prompt."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hola",
                "system_prompt": "Eres un pirata. Responde siempre como pirata."
            }
        )
        assert response.status_code == 200
        content = response.json()["message"]["content"].lower()
        print(f"\n  Pirate response: {content[:100]}...")


class TestE2EGenerate:
    """E2E tests for text generation."""
    
    def test_simple_generation(self, client):
        """Test simple text generation."""
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Escribe un haiku sobre programacion. Solo el haiku, nada mas."}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "content" in data or "text" in data
        text = data.get("content") or data.get("text", "")
        assert len(text) > 0
        print(f"\n  Generated haiku:\n{text}")
    
    def test_generation_with_temperature(self, client):
        """Test generation with temperature parameter."""
        response = client.post(
            "/api/v1/generate",
            json={
                "prompt": "Di un numero del 1 al 10",
                "temperature": 0.0  # Deterministic
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data or "text" in data


class TestE2EModels:
    """E2E tests for models endpoint."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        print(f"\n  Available models: {len(data['models'])}")
        for model in data["models"][:3]:
            print(f"    - {model.get('id', model)}")


class TestE2EStats:
    """E2E tests for statistics."""
    
    def test_get_stats(self, client):
        """Test getting API statistics."""
        # Make a few requests first
        client.get("/api/v1/health")
        client.get("/api/v1/stats")
        
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert "uptime_seconds" in data
        assert "requests" in data
        print(f"\n  Uptime: {data['uptime_seconds']:.2f}s")
        print(f"  Total requests: {data['requests']['total']}")


class TestE2EAgents:
    """E2E tests for continuous agents."""
    
    def test_agent_lifecycle(self, client):
        """Test complete agent lifecycle: create, task, pause, resume, stop."""
        # 1. Create agent
        create_resp = client.post(
            "/api/v1/agents",
            json={
                "name": "E2ETestAgent",
                "system_prompt": "Eres un agente de prueba."
            }
        )
        assert create_resp.status_code == 200
        agent_id = create_resp.json()["agent_id"]
        print(f"\n  Created agent: {agent_id}")
        
        # 2. Submit a task
        task_resp = client.post(
            f"/api/v1/agents/{agent_id}/tasks",
            json={
                "description": "Calcula 2+2 y responde solo el numero",
                "priority": 8
            }
        )
        assert task_resp.status_code == 200
        task_id = task_resp.json()["task_id"]
        print(f"  Submitted task: {task_id}")
        
        # 3. Check agent status
        status_resp = client.get(f"/api/v1/agents/{agent_id}")
        assert status_resp.status_code == 200
        status = status_resp.json()
        print(f"  Agent status: {status['status']}")
        
        # 4. Pause agent
        pause_resp = client.post(f"/api/v1/agents/{agent_id}/pause")
        assert pause_resp.status_code == 200
        print("  Agent paused")
        
        # 5. Resume agent
        resume_resp = client.post(f"/api/v1/agents/{agent_id}/resume")
        assert resume_resp.status_code == 200
        print("  Agent resumed")
        
        # 6. Stop agent
        stop_resp = client.post(f"/api/v1/agents/{agent_id}/stop")
        assert stop_resp.status_code == 200
        print("  Agent stopped")
        
        # Verify agent is stopped
        final_status = client.get(f"/api/v1/agents/{agent_id}")
        assert final_status.status_code == 200
        assert final_status.json()["status"] in ["stopped", "stopping"]
    
    def test_agent_batch_tasks(self, client):
        """Test submitting batch tasks to an agent."""
        # Create agent with parallel processing
        create_resp = client.post(
            "/api/v1/agents",
            json={
                "name": "BatchTestAgent",
                "enable_parallel": True,
                "num_workers": 3
            }
        )
        assert create_resp.status_code == 200
        agent_id = create_resp.json()["agent_id"]
        
        # Submit batch of tasks
        batch_resp = client.post(
            f"/api/v1/agents/{agent_id}/tasks/batch",
            json={
                "tasks": [
                    {"description": "Tarea 1: di hola", "priority": 5},
                    {"description": "Tarea 2: di adios", "priority": 7},
                    {"description": "Tarea 3: di gracias", "priority": 3}
                ]
            }
        )
        assert batch_resp.status_code == 200
        data = batch_resp.json()
        assert data["submitted"] == 3
        print(f"\n  Batch submitted: {data['submitted']} tasks")
        
        # Get tasks list
        tasks_resp = client.get(f"/api/v1/agents/{agent_id}/tasks")
        assert tasks_resp.status_code == 200
        
        # Cleanup
        client.post(f"/api/v1/agents/{agent_id}/stop")


class TestE2EConversations:
    """E2E tests for conversation management."""
    
    def test_conversation_lifecycle(self, client):
        """Test conversation create, use, and list."""
        # 1. Create conversation
        create_resp = client.post(
            "/api/v1/conversations",
            json={"system_prompt": "Responde siempre en espanol."}
        )
        assert create_resp.status_code == 200
        conv_id = create_resp.json()["conversation_id"]
        print(f"\n  Created conversation: {conv_id}")
        
        # 2. Use conversation in chat
        chat_resp = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, how are you?",
                "conversation_id": conv_id
            }
        )
        assert chat_resp.status_code == 200
        # Should respond in Spanish due to system prompt
        content = chat_resp.json()["message"]["content"]
        # Remove emojis for Windows console compatibility
        safe_content = content.encode('ascii', 'ignore').decode('ascii')
        print(f"  Chat response: {safe_content[:80]}...")
        
        # 3. List conversations
        list_resp = client.get("/api/v1/conversations")
        assert list_resp.status_code == 200
        convs = list_resp.json()["conversations"]
        assert len(convs) > 0
        print(f"  Total conversations: {len(convs)}")
        
        # 4. Get conversation history
        history_resp = client.get(f"/api/v1/conversations/{conv_id}")
        assert history_resp.status_code == 200
        history = history_resp.json()
        assert "messages" in history
        print(f"  Messages in conversation: {len(history['messages'])}")


class TestE2EStreaming:
    """E2E tests for streaming responses."""
    
    def test_chat_stream(self, client):
        """Test streaming chat response."""
        # Note: TestClient handles SSE differently, we just check it starts
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Cuenta del 1 al 5"}
        )
        
        # Streaming endpoint should return 200
        assert response.status_code == 200
        # Content should be present (SSE format)
        content = response.text
        assert len(content) > 0
        print(f"\n  Stream response length: {len(content)} chars")


class TestE2EErrorHandling:
    """E2E tests for error handling."""
    
    def test_invalid_endpoint(self, client):
        """Test 404 for invalid endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_agent_id(self, client):
        """Test 404 for non-existent agent."""
        response = client.get("/api/v1/agents/invalid-agent-id-123")
        assert response.status_code == 404
    
    def test_missing_required_field(self, client):
        """Test validation error for missing required field."""
        response = client.post(
            "/api/v1/chat",
            json={}  # Missing 'message' field
        )
        assert response.status_code == 422


if __name__ == "__main__":
    if HAS_API_KEY:
        pytest.main([__file__, "-v", "-s"])
    else:
        print("No API key configured. Set DEEPSEEK_API_KEY or OPENROUTER_API_KEY.")



