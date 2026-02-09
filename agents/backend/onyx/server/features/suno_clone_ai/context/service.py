"""
Context Service - Servicio de contexto
"""

from typing import Dict, Any, Optional
from .base import BaseContext
from .manager import ContextManager
from db.service import DatabaseService
from redis.service import RedisService


class ContextService:
    """Servicio para gestionar contexto"""

    def __init__(
        self,
        db_service: Optional[DatabaseService] = None,
        redis_service: Optional[RedisService] = None
    ):
        """Inicializa el servicio de contexto"""
        self.manager = ContextManager(db_service, redis_service)

    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto de una sesión"""
        return await self.manager.get_context(session_id)

    async def update_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Actualiza el contexto de una sesión"""
        await self.manager.update_context(session_id, context)

