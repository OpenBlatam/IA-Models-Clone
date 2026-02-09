"""
Chat Service - Servicio de chat
"""

from typing import List, Dict, Any, Optional
from .base import BaseChat
from .conversation import ConversationManager
from .message_handler import MessageHandler
from llm.service import LLMService
from db.service import DatabaseService
from context.service import ContextService


class ChatService:
    """Servicio para gestionar chat"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        db_service: Optional[DatabaseService] = None,
        context_service: Optional[ContextService] = None
    ):
        """Inicializa el servicio de chat"""
        self.llm_service = llm_service
        self.db_service = db_service
        self.context_service = context_service
        self.conversation_manager = ConversationManager(db_service)
        self.message_handler = MessageHandler(llm_service, context_service)

    async def send_message(self, message: str, user_id: str, **kwargs) -> Dict[str, Any]:
        """Envía un mensaje y obtiene respuesta"""
        conversation = await self.conversation_manager.get_or_create(user_id)
        response = await self.message_handler.process(message, conversation, **kwargs)
        await self.conversation_manager.add_message(user_id, message, response)
        return response

    async def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene historial de conversación"""
        return await self.conversation_manager.get_history(user_id, limit)

