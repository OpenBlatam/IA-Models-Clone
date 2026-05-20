"""
Permissions - Sistema de permisos granulares
"""

import logging
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permisos disponibles"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SHARE = "share"
    COMMENT = "comment"
    EXPORT = "export"
    IMPORT = "import"


@dataclass
class ResourcePermission:
    """Permisos sobre un recurso"""
    resource_id: str
    resource_type: str
    permissions: Set[Permission]
    user_id: Optional[str] = None
    group_id: Optional[str] = None


class PermissionManager:
    """Gestor de permisos"""

    def __init__(self):
        """Inicializar gestor de permisos"""
        self.permissions: Dict[str, List[ResourcePermission]] = {}
        self.groups: Dict[str, List[str]] = {}  # group_id -> [user_ids]

    def grant_permission(
        self,
        resource_id: str,
        resource_type: str,
        user_id: str,
        permissions: List[Permission]
    ):
        """
        Otorgar permisos a un usuario.

        Args:
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            user_id: ID del usuario
            permissions: Lista de permisos
        """
        key = f"{resource_type}:{resource_id}"
        if key not in self.permissions:
            self.permissions[key] = []
        
        # Buscar permiso existente
        existing = next(
            (p for p in self.permissions[key] if p.user_id == user_id),
            None
        )
        
        if existing:
            existing.permissions.update(permissions)
        else:
            self.permissions[key].append(
                ResourcePermission(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    permissions=set(permissions),
                    user_id=user_id
                )
            )
        
        logger.info(f"Permisos otorgados a {user_id} en {resource_id}")

    def revoke_permission(
        self,
        resource_id: str,
        resource_type: str,
        user_id: str,
        permissions: Optional[List[Permission]] = None
    ):
        """
        Revocar permisos.

        Args:
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            user_id: ID del usuario
            permissions: Permisos a revocar (None para todos)
        """
        key = f"{resource_type}:{resource_id}"
        if key not in self.permissions:
            return
        
        existing = next(
            (p for p in self.permissions[key] if p.user_id == user_id),
            None
        )
        
        if existing:
            if permissions:
                existing.permissions.difference_update(permissions)
            else:
                self.permissions[key].remove(existing)

    def has_permission(
        self,
        resource_id: str,
        resource_type: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Verificar si un usuario tiene un permiso.

        Args:
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            user_id: ID del usuario
            permission: Permiso a verificar

        Returns:
            True si tiene el permiso
        """
        key = f"{resource_type}:{resource_id}"
        if key not in self.permissions:
            return False
        
        # Verificar permisos directos
        for perm in self.permissions[key]:
            if perm.user_id == user_id and permission in perm.permissions:
                return True
        
        # Verificar permisos de grupo
        user_groups = [gid for gid, users in self.groups.items() if user_id in users]
        for group_id in user_groups:
            for perm in self.permissions[key]:
                if perm.group_id == group_id and permission in perm.permissions:
                    return True
        
        return False

    def get_user_permissions(
        self,
        resource_id: str,
        resource_type: str,
        user_id: str
    ) -> List[Permission]:
        """
        Obtener todos los permisos de un usuario.

        Args:
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            user_id: ID del usuario

        Returns:
            Lista de permisos
        """
        key = f"{resource_type}:{resource_id}"
        if key not in self.permissions:
            return []
        
        permissions = set()
        
        # Permisos directos
        for perm in self.permissions[key]:
            if perm.user_id == user_id:
                permissions.update(perm.permissions)
        
        # Permisos de grupo
        user_groups = [gid for gid, users in self.groups.items() if user_id in users]
        for group_id in user_groups:
            for perm in self.permissions[key]:
                if perm.group_id == group_id:
                    permissions.update(perm.permissions)
        
        return list(permissions)

    def create_group(self, group_id: str, user_ids: List[str]):
        """
        Crear grupo de usuarios.

        Args:
            group_id: ID del grupo
            user_ids: IDs de usuarios
        """
        self.groups[group_id] = user_ids
        logger.info(f"Grupo creado: {group_id}")

    def grant_group_permission(
        self,
        resource_id: str,
        resource_type: str,
        group_id: str,
        permissions: List[Permission]
    ):
        """
        Otorgar permisos a un grupo.

        Args:
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            group_id: ID del grupo
            permissions: Lista de permisos
        """
        key = f"{resource_type}:{resource_id}"
        if key not in self.permissions:
            self.permissions[key] = []
        
        existing = next(
            (p for p in self.permissions[key] if p.group_id == group_id),
            None
        )
        
        if existing:
            existing.permissions.update(permissions)
        else:
            self.permissions[key].append(
                ResourcePermission(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    permissions=set(permissions),
                    group_id=group_id
                )
            )






