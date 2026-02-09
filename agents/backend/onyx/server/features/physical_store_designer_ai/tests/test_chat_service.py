"""
Tests for ChatService
"""

import pytest

from ..services.chat_service import ChatService
from ..core.exceptions import NotFoundError
from ..core.models import ChatMessage


class TestChatService:
    """Test cases for ChatService"""
    
    def test_create_session(self, chat_service: ChatService):
        """Test creating a chat session"""
        session_id = chat_service.create_session()
        assert session_id is not None
        assert len(session_id) > 0
        
        # Verify session exists
        session = chat_service.get_session(session_id)
        assert session is not None
        assert len(session.messages) > 0  # Should have welcome message
    
    def test_get_session_not_found(self, chat_service: ChatService):
        """Test getting non-existent session"""
        with pytest.raises(NotFoundError):
            chat_service.get_session("non_existent_id")
    
    def test_send_message(self, chat_service: ChatService):
        """Test sending a message"""
        session_id = chat_service.create_session()
        
        message = ChatMessage(
            role="user",
            content="Quiero abrir una cafetería"
        )
        
        response = chat_service.send_message(session_id, message)
        assert response is not None
        assert "role" in response
        assert response["role"] == "assistant"
        
        # Verify message was added to session
        session = chat_service.get_session(session_id)
        assert len(session.messages) >= 3  # Welcome + user + assistant
    
    def test_extract_store_info(self, chat_service: ChatService):
        """Test extracting store info from chat"""
        session_id = chat_service.create_session()
        
        # Send messages with store information
        messages = [
            "Quiero abrir una cafetería",
            "Se llamará Café Central",
            "Estilo moderno",
            "Presupuesto medio"
        ]
        
        for msg in messages:
            chat_service.send_message(
                session_id,
                ChatMessage(role="user", content=msg)
            )
        
        # Extract info
        store_info = chat_service.extract_store_info(session_id)
        assert store_info is not None
        assert isinstance(store_info, dict)








