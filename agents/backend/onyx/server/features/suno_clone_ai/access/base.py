"""
Base Access Controller - Clase base para controladores de acceso
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAccessController(ABC):
    """Clase base abstracta para controladores de acceso"""

    @abstractmethod
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Verifica si un usuario tiene permiso para una acción"""
        pass

    @abstractmethod
    async def get_roles(self, user_id: str) -> list:
        """Obtiene los roles de un usuario"""
        pass

