"""
Base Access Controller - Clase base para controladores de acceso
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseAccessController(ABC):
    """Clase base abstracta para controladores de acceso"""
    
    @abstractmethod
    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verifica si un usuario tiene permiso para realizar una acción en un recurso
        
        Args:
            user_id: ID del usuario
            resource: Recurso a acceder
            action: Acción a realizar
            context: Contexto adicional
            
        Returns:
            True si tiene permiso, False en caso contrario
        """
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Obtiene todos los permisos de un usuario"""
        pass
    
    @abstractmethod
    async def grant_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Otorga un permiso a un usuario"""
        pass
    
    @abstractmethod
    async def revoke_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Revoca un permiso de un usuario"""
        pass

