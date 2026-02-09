"""
RBAC Manager - Implementación de Role-Based Access Control
"""
from typing import List, Optional, Dict, Any
from .base import BaseAccessController


class RBACManager(BaseAccessController):
    """Gestor de control de acceso basado en roles"""
    
    def __init__(self):
        self.roles: Dict[str, List[str]] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self.permissions: Dict[str, Dict[str, List[str]]] = {}
    
    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Verifica permiso usando RBAC"""
        user_roles = self.user_roles.get(user_id, [])
        
        for role in user_roles:
            role_perms = self.permissions.get(role, {}).get(resource, [])
            if action in role_perms:
                return True
        
        return False
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Obtiene todos los permisos de un usuario"""
        user_roles = self.user_roles.get(user_id, [])
        all_permissions = []
        
        for role in user_roles:
            for resource, actions in self.permissions.get(role, {}).items():
                all_permissions.extend([f"{resource}:{action}" for action in actions])
        
        return list(set(all_permissions))
    
    async def grant_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Otorga un permiso asignando un rol"""
        # Implementación simplificada
        role = f"{resource}_{action}_role"
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
        return True
    
    async def revoke_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Revoca un permiso"""
        role = f"{resource}_{action}_role"
        if user_id in self.user_roles:
            if role in self.user_roles[user_id]:
                self.user_roles[user_id].remove(role)
                return True
        return False

