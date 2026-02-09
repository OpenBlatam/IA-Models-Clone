"""
Chat Module - Sistema de Chat y Conversación
Gestiona conversaciones, historial de chat, y procesamiento de mensajes.

Rol en el Ecosistema IA:
- Conversaciones, historial, procesamiento de mensajes
- Interfaz principal con usuarios, contexto de conversación
- Orquestación de respuestas del sistema de IA

Reglas de Importación:
- Puede importar: llm, db, context, tracing
- NO debe importar: server (evitar ciclos)
- Usa inyección de dependencias para servicios
"""

from .base import BaseChat
from .service import ChatService
from .conversation import ConversationManager
from .message_handler import MessageHandler
from .repository import ChatRepository
from .main import (
    get_chat_service,
    send_message,
    get_history,
    initialize_chat,
)

__all__ = [
    # Clases principales
    "BaseChat",
    "ChatService",
    "ConversationManager",
    "MessageHandler",
    "ChatRepository",
    # Funciones de acceso rápido
    "get_chat_service",
    "send_message",
    "get_history",
    "initialize_chat",
]

