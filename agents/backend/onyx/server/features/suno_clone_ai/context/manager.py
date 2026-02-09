"""
Context Manager - Gestor de contexto
"""

from typing import Dict, Any, Optional
from db.service import DatabaseService
from redis.service import RedisService


class ContextManager:
    """Gestor de contexto de sesiones"""

    def __init__(
        self,
        db_service: Optional[DatabaseService] = None,
        redis_service: Optional[RedisService] = None
    ):
        """Inicializa el gestor de contexto"""
        self.db_service = db_service
        self.redis_service = redis_service
        self._contexts: Dict[str, Dict[str, Any]] = {}

    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto de una sesión"""
        # Intentar obtener de Redis primero
        if self.redis_service:
            context = self.redis_service.get(f"context:{session_id}")
            if context:
                return context

        # Fallback a memoria
        return self._contexts.get(session_id)

    async def update_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Actualiza el contexto de una sesión"""
        self._contexts[session_id] = context

        # Guardar en Redis si está disponible
        if self.redis_service:
            self.redis_service.set(f"context:{session_id}", context, ttl=3600)

