"""
Access Service - Servicio principal de gestión de acceso
"""
from typing import List, Optional, Dict, Any
from .base import BaseAccessController
from .rbac import RBACManager
from .policies import PolicyManager


class AccessService:
    """Servicio principal para gestión de control de acceso"""
    
    def __init__(
        self,
        rbac_manager: Optional[RBACManager] = None,
        policy_manager: Optional[PolicyManager] = None
    ):
        self.rbac_manager = rbac_manager or RBACManager()
        self.policy_manager = policy_manager or PolicyManager()
    
    async def check_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Verifica acceso combinando RBAC y políticas"""
        # Verificar RBAC
        rbac_allowed = await self.rbac_manager.check_permission(
            user_id, resource, action, context
        )
        
        # Verificar políticas
        policy_allowed = await self.policy_manager.evaluate(
            user_id, resource, action, context
        )
        
        return rbac_allowed and policy_allowed
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Obtiene todos los permisos de un usuario"""
        return await self.rbac_manager.get_user_permissions(user_id)
    
    async def grant_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Otorga un permiso"""
        return await self.rbac_manager.grant_permission(user_id, resource, action)
    
    async def revoke_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Revoca un permiso"""
        return await self.rbac_manager.revoke_permission(user_id, resource, action)

