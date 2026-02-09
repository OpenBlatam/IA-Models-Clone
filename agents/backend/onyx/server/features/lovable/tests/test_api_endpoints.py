import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.orm import Session

from agents.backend.onyx.server.features.lovable.api.app import app, get_db_session, get_agent
from agents.backend.onyx.server.features.lovable.core.lovable_sam3_agent import LovableSAM3Agent

# Mock database session
@pytest.fixture
def mock_db_session():
    session = MagicMock(spec=Session)
    return session

# Mock agent
@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=LovableSAM3Agent)
    agent.running = True
    agent.task_manager = MagicMock()
    
    # Make create_task awaitable
    async def mock_create_task(*args, **kwargs):
        return "test-task-id"
    agent.task_manager.create_task.side_effect = mock_create_task
    
    # Make get_task awaitable
    async def mock_get_task(*args, **kwargs):
        return {"id": "test-task-id", "status": "pending"}
    agent.task_manager.get_task.side_effect = mock_get_task
    
    return agent

@pytest.fixture
def client(mock_db_session, mock_agent):
    # Override DB dependency
    def override_get_db_session():
        try:
            yield mock_db_session
        finally:
            pass
    
    # Override Agent dependency
    def override_get_agent():
        return mock_agent
    
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_agent] = override_get_agent
    
    # Patch init_db to avoid startup errors if lifespan runs (though TestClient might not trigger it fully depending on version)
    with patch("agents.backend.onyx.server.features.lovable.api.app.init_db"):
        with TestClient(app) as test_client:
            yield test_client
    
    app.dependency_overrides.clear()

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health_endpoint(client, mock_agent):
    # For health endpoint, we need to ensure app.state.agent is set if we want to test that path,
    # OR we rely on the fact that health endpoint might check app.state directly.
    # However, our refactored health endpoint checks app.state.agent.
    # TestClient with lifespan should run lifespan.
    # But we are mocking get_agent, which is used by other endpoints.
    # The health endpoint accesses app.state.agent DIRECTLY, not via dependency.
    # So we need to set it on the app.
    
    client.app.state.agent = mock_agent
    
    # Mock internal imports in health endpoint
    with patch("agents.backend.onyx.server.features.lovable.api.app.get_db_session") as mock_get_db:
        mock_get_db.return_value = iter([MagicMock()])
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_publish_chat(client, mock_db_session):
    # Mock ChatService and ChatRepository
    with patch("agents.backend.onyx.server.features.lovable.services.chat_service.ChatService") as MockChatService:
        mock_service = MockChatService.return_value
        mock_service.publish_chat.return_value = {
            "id": "test-chat-id",
            "user_id": "user123",
            "title": "Test Chat",
            "content": "Test Content",
            "is_public": True
        }
        
        with patch("agents.backend.onyx.server.features.lovable.repositories.chat_repository.ChatRepository") as MockChatRepo:
            mock_repo = MockChatRepo.return_value
            mock_repo.create.return_value = MagicMock(id="test-chat-id")
            
            response = client.post("/api/v1/publish", json={
                "user_id": "user123",
                "title": "Test Chat",
                "content": "Test Content",
                "is_public": True
            })
            
            assert response.status_code == 200
            assert response.json()["task_id"] == "test-task-id"

def test_vote_chat(client, mock_db_session):
    chat_id = "test-chat-id"
    
    with patch("agents.backend.onyx.server.features.lovable.services.chat_service.ChatService") as MockChatService:
        mock_service = MockChatService.return_value
        mock_service.chat_repo.get_by_id.return_value = MagicMock(id=chat_id, user_id="owner123")
        
        with patch("agents.backend.onyx.server.features.lovable.services.vote_service.VoteService") as MockVoteService:
            mock_vote_service = MockVoteService.return_value
            mock_vote_service.increment_vote.return_value = {"new_count": 1}
            
            response = client.post(f"/api/v1/chats/{chat_id}/vote", json={
                "user_id": "voter123",
                "vote_type": "upvote",
                "chat_id": chat_id
            })
            
            assert response.status_code == 200
            assert response.json()["task_id"] == "test-task-id"

def test_deprecated_vote_endpoint(client, mock_db_session):
    # Test legacy endpoint
    chat_id = "test-chat-id"
    
    with patch("agents.backend.onyx.server.features.lovable.services.chat_service.ChatService") as MockChatService:
        mock_service = MockChatService.return_value
        mock_service.chat_repo.get_by_id.return_value = MagicMock(id=chat_id, user_id="owner123")
        
        with patch("agents.backend.onyx.server.features.lovable.services.vote_service.VoteService") as MockVoteService:
            mock_vote_service = MockVoteService.return_value
            mock_vote_service.increment_vote.return_value = {"new_count": 1}
            
            response = client.post("/api/v1/vote", json={
                "user_id": "voter123",
                "vote_type": "upvote",
                "chat_id": chat_id
            })
            
            assert response.status_code == 200
