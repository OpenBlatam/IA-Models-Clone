"""
Configuración compartida de pytest y fixtures base para Lovable Community
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Importar módulos del proyecto
import sys
from pathlib import Path as PathLib

# Agregar el directorio raíz al path
project_root = PathLib(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models import Base, PublishedChat, ChatRemix, ChatVote, ChatView
    from services import ChatService, RankingService
    from schemas import (
        PublishChatRequest,
        RemixChatRequest,
        VoteRequest,
        SearchRequest
    )
except ImportError:
    # Intentar imports relativos
    pass


# ============================================================================
# Fixtures de Base de Datos
# ============================================================================

@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Crea una sesión de base de datos en memoria para tests"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def chat_service(db_session: Session) -> ChatService:
    """Crea una instancia de ChatService para tests"""
    return ChatService(db_session)


@pytest.fixture
def ranking_service() -> RankingService:
    """Crea una instancia de RankingService para tests"""
    return RankingService()


# ============================================================================
# Fixtures de Mocks
# ============================================================================

@pytest.fixture
def mock_db_session() -> Mock:
    """Mock de sesión de base de datos"""
    session = Mock(spec=Session)
    session.query = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.refresh = Mock()
    session.delete = Mock()
    return session


@pytest.fixture
def mock_chat_service() -> Mock:
    """Mock del servicio de chats"""
    service = Mock(spec=ChatService)
    service.publish_chat = Mock()
    service.get_chat = Mock(return_value=None)
    service.vote_chat = Mock()
    service.remix_chat = Mock()
    service.search_chats = Mock(return_value=([], 0))
    service.get_top_chats = Mock(return_value=[])
    service.update_chat = Mock()
    service.delete_chat = Mock(return_value=True)
    return service


# ============================================================================
# Fixtures de Datos de Prueba
# ============================================================================

@pytest.fixture
def sample_user_id() -> str:
    """ID de usuario de ejemplo"""
    return "test-user-123"


@pytest.fixture
def sample_chat_id() -> str:
    """ID de chat de ejemplo"""
    import uuid
    return str(uuid.uuid4())


@pytest.fixture
def sample_publish_request() -> Dict[str, Any]:
    """Request de publicación de ejemplo"""
    return {
        "title": "Mi increíble chat sobre IA",
        "description": "Un chat sobre inteligencia artificial",
        "chat_content": '{"messages": [{"role": "user", "content": "Hello"}]}',
        "tags": ["ai", "machine-learning"],
        "is_public": True
    }


@pytest.fixture
def sample_remix_request() -> Dict[str, Any]:
    """Request de remix de ejemplo"""
    return {
        "original_chat_id": "original-chat-123",
        "title": "Remix: Mi increíble chat",
        "description": "Una versión mejorada",
        "chat_content": '{"messages": [{"role": "user", "content": "Hello improved"}]}',
        "tags": ["ai", "remix"]
    }


@pytest.fixture
def sample_vote_request() -> Dict[str, Any]:
    """Request de voto de ejemplo"""
    return {
        "chat_id": "chat-123",
        "vote_type": "upvote"
    }


@pytest.fixture
def sample_chat_data() -> Dict[str, Any]:
    """Datos de chat de ejemplo"""
    return {
        "id": "chat-123",
        "user_id": "user-123",
        "title": "Test Chat",
        "description": "A test chat",
        "chat_content": '{"messages": []}',
        "tags": "ai,test",
        "vote_count": 10,
        "remix_count": 5,
        "view_count": 100,
        "score": 25.5,
        "is_public": True,
        "is_featured": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


# ============================================================================
# Fixtures de FastAPI
# ============================================================================

@pytest.fixture
def test_client():
    """Cliente de prueba para FastAPI"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


# ============================================================================
# Helpers de Utilidad
# ============================================================================

def create_test_chat(
    db_session: Session,
    user_id: str = "test-user",
    title: str = "Test Chat",
    **kwargs
) -> PublishedChat:
    """
    Create a test chat in the database.
    
    Args:
        db_session: Database session
        user_id: User ID (default: "test-user")
        title: Chat title (default: "Test Chat")
        **kwargs: Additional chat fields
        
    Returns:
        PublishedChat instance
        
    Raises:
        ValueError: If db_session is None, or user_id/title is None or empty
    """
    if db_session is None:
        raise ValueError("db_session cannot be None")
    
    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be None or empty")
    
    if not title or not title.strip():
        raise ValueError("title cannot be None or empty")
    
    import uuid
    
    chat = PublishedChat(
        id=str(uuid.uuid4()),
        user_id=user_id.strip(),
        title=title.strip(),
        description=kwargs.get("description", "Test description"),
        chat_content=kwargs.get("chat_content", '{"messages": []}'),
        tags=kwargs.get("tags"),
        is_public=kwargs.get("is_public", True),
        vote_count=max(0, kwargs.get("vote_count", 0)),
        remix_count=max(0, kwargs.get("remix_count", 0)),
        view_count=max(0, kwargs.get("view_count", 0)),
        score=max(0.0, kwargs.get("score", 0.0)),
        created_at=kwargs.get("created_at", datetime.utcnow()),
        updated_at=kwargs.get("updated_at", datetime.utcnow())
    )
    
    db_session.add(chat)
    db_session.commit()
    db_session.refresh(chat)
    
    return chat


def assert_chat_valid(chat: Any) -> None:
    """
    Verify that a chat is valid.
    
    Args:
        chat: Chat object to validate
        
    Raises:
        AssertionError: If chat is invalid
    """
    assert chat is not None, "Chat is None"
    assert hasattr(chat, "id"), "Chat missing id"
    assert hasattr(chat, "title"), "Chat missing title"
    assert hasattr(chat, "user_id"), "Chat missing user_id"
    assert chat.id, "Chat id is empty"
    assert chat.title, "Chat title is empty"
    assert chat.user_id, "Chat user_id is empty"
    
    # Validate counts are non-negative
    if hasattr(chat, "vote_count"):
        assert chat.vote_count >= 0, f"vote_count must be >= 0, got {chat.vote_count}"
    if hasattr(chat, "remix_count"):
        assert chat.remix_count >= 0, f"remix_count must be >= 0, got {chat.remix_count}"
    if hasattr(chat, "view_count"):
        assert chat.view_count >= 0, f"view_count must be >= 0, got {chat.view_count}"
    if hasattr(chat, "score"):
        assert chat.score >= 0.0, f"score must be >= 0.0, got {chat.score}"

