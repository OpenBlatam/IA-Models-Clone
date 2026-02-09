"""
Models - Modelos de permisos y roles
"""

from typing import List, Set


class Permission:
    """Modelo de permiso"""

    def __init__(self, name: str, resource: str, action: str):
        """Inicializa un permiso"""
        self.name = name
        self.resource = resource
        self.action = action


class Role:
    """Modelo de rol"""

    def __init__(self, name: str, permissions: List[Permission]):
        """Inicializa un rol"""
        self.name = name
        self.permissions = permissions

    def has_permission(self, permission: Permission) -> bool:
        """Verifica si el rol tiene un permiso"""
        return permission in self.permissions

