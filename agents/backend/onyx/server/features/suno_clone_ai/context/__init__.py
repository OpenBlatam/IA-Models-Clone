"""
Context Module - Gestión de Contexto
Gestiona contexto de conversaciones, sesiones, y estado de la aplicación.

Rol en el Ecosistema IA:
- Contexto de conversaciones, sesiones, memoria
- Window de contexto, memoria a largo plazo, RAG
- Gestión del estado y contexto del sistema

Reglas de Importación:
- Puede importar: db, redis, indexing, document_index
- NO debe importar: chat, llm, agents (evitar ciclos)
- Usa inyección de dependencias
"""

from .base import BaseContext
from .service import ContextService
from .manager import ContextManager
from .main import (
    get_context_service,
    get_context,
    update_context,
    initialize_context,
)

__all__ = [
    # Clases principales
    "BaseContext",
    "ContextService",
    "ContextManager",
    # Funciones de acceso rápido
    "get_context_service",
    "get_context",
    "update_context",
    "initialize_context",
]

