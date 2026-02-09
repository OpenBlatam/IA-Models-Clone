"""
Chat Main - Funciones base y entry points del módulo de chat

Rol en el Ecosistema IA:
- Conversaciones, historial, procesamiento de mensajes
- Interfaz principal con usuarios, contexto de conversación
- Orquestación de respuestas del sistema de IA
"""

from typing import Optional, List, Dict, Any
from .service import ChatService
from .conversation import ConversationManager
from .message_handler import MessageHandler
from llm.main import get_llm_service
from db.main import get_db_service
from context.main import get_context_service


# Instancia global del servicio
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """
    Obtiene la instancia global del servicio de chat.
    
    Returns:
        ChatService: Servicio de chat
    """
    global _chat_service
    if _chat_service is None:
        llm_service = get_llm_service()
        db_service = get_db_service()
        context_service = get_context_service()
        _chat_service = ChatService(
            llm_service=llm_service,
            db_service=db_service,
            context_service=context_service
        )
    return _chat_service


async def send_message(message: str, user_id: str, **kwargs) -> Dict[str, Any]:
    """
    Envía un mensaje y obtiene respuesta.
    
    Args:
        message: Mensaje del usuario
        user_id: ID del usuario
        **kwargs: Parámetros adicionales
        
    Returns:
        Respuesta del sistema
    """
    service = get_chat_service()
    return await service.send_message(message, user_id, **kwargs)


async def get_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Obtiene historial de conversación.
    
    Args:
        user_id: ID del usuario
        limit: Número máximo de mensajes
        
    Returns:
        Lista de mensajes del historial
    """
    service = get_chat_service()
    return await service.get_history(user_id, limit)


def initialize_chat() -> ChatService:
    """
    Inicializa el sistema de chat.
    
    Returns:
        ChatService: Servicio inicializado
    """
    return get_chat_service()

