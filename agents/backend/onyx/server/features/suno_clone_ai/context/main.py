"""
Context Main - Funciones base y entry points del módulo de contexto

Rol en el Ecosistema IA:
- Contexto de conversaciones, sesiones, memoria
- Window de contexto, memoria a largo plazo, RAG
- Gestión del estado y contexto del sistema
"""

from typing import Optional, Dict, Any
from .service import ContextService
from .manager import ContextManager
from db.main import get_db_service
from redis.main import get_redis_service


# Instancia global del servicio
_context_service: Optional[ContextService] = None


def get_context_service() -> ContextService:
    """
    Obtiene la instancia global del servicio de contexto.
    
    Returns:
        ContextService: Servicio de contexto
    """
    global _context_service
    if _context_service is None:
        db_service = get_db_service()
        redis_service = get_redis_service()
        _context_service = ContextService(
            db_service=db_service,
            redis_service=redis_service
        )
    return _context_service


async def get_context(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene el contexto de una sesión.
    
    Args:
        session_id: ID de la sesión
        
    Returns:
        Contexto de la sesión o None
    """
    service = get_context_service()
    return await service.get_context(session_id)


async def update_context(session_id: str, context: Dict[str, Any]) -> None:
    """
    Actualiza el contexto de una sesión.
    
    Args:
        session_id: ID de la sesión
        context: Nuevo contexto
    """
    service = get_context_service()
    await service.update_context(session_id, context)


def initialize_context() -> ContextService:
    """
    Inicializa el sistema de contexto.
    
    Returns:
        ContextService: Servicio inicializado
    """
    return get_context_service()

