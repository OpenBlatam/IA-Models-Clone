"""
Base Context - Clase base para contexto
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseContext(ABC):
    """Clase base abstracta para gestión de contexto"""

    @abstractmethod
    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el contexto de una sesión"""
        pass

    @abstractmethod
    async def update_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Actualiza el contexto de una sesión"""
        pass

