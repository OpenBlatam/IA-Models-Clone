"""
Helpers para crear mocks para Lovable Community
"""

from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime


def create_mock_chat_service(
    chats: Optional[List[Dict[str, Any]]] = None,
    default_chat: Optional[Dict[str, Any]] = None
) -> Mock:
    """Crea un mock del servicio de chats"""
    service = Mock()
    
    # Configurar métodos básicos
    service.publish_chat = Mock()
    service.get_chat = Mock(return_value=default_chat)
    service.vote_chat = Mock()
    service.remix_chat = Mock()
    service.search_chats = Mock(return_value=(chats or [], len(chats) if chats else 0))
    service.get_top_chats = Mock(return_value=chats or [])
    service.update_chat = Mock()
    service.delete_chat = Mock(return_value=True)
    service.get_user_vote = Mock(return_value=None)
    service.get_remixes = Mock(return_value=[])
    service.feature_chat = Mock()
    service.get_user_profile = Mock(return_value={
        "user_id": "test-user",
        "total_chats": 0,
        "total_remixes": 0,
        "total_votes": 0
    })
    service.get_trending_chats = Mock(return_value=[])
    service.get_analytics = Mock(return_value={
        "total_chats": 0,
        "total_users": 0,
        "total_votes": 0
    })
    service.get_chat_stats_detailed = Mock(return_value={
        "chat_id": "test-chat",
        "vote_count": 0,
        "remix_count": 0,
        "view_count": 0,
        "score": 0.0
    })
    
    return service


def create_mock_ranking_service() -> Mock:
    """Crea un mock del servicio de ranking"""
    service = Mock()
    service.calculate_score = Mock(return_value=10.5)
    return service


def create_mock_db_session() -> Mock:
    """Crea un mock de sesión de base de datos"""
    session = Mock()
    session.query = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.refresh = Mock()
    session.delete = Mock()
    session.filter = Mock(return_value=session)
    session.first = Mock(return_value=None)
    session.all = Mock(return_value=[])
    session.count = Mock(return_value=0)
    session.scalar = Mock(return_value=0)
    return session

