"""
Sistema de Permisos y Roles
===========================
Gestión de permisos y roles de usuario
"""

from typing import Dict, Any, List, Optional, Set
from uuid import UUID
from enum import Enum
import structlog

logger = structlog.get_logger()


class Permission(str, Enum):
    """Permisos disponibles"""
    # Validaciones
    CREATE_VALIDATION = "create_validation"
    VIEW_VALIDATION = "view_validation"
    DELETE_VALIDATION = "delete_validation"
    EXPORT_VALIDATION = "export_validation"
    
    # Conexiones
    CONNECT_SOCIAL_MEDIA = "connect_social_media"
    DISCONNECT_SOCIAL_MEDIA = "disconnect_social_media"
    VIEW_CONNECTIONS = "view_connections"
    
    # Reportes
    VIEW_REPORTS = "view_reports"
    EXPORT_REPORTS = "export_reports"
    
    # Administración
    VIEW_ALL_VALIDATIONS = "view_all_validations"
    MANAGE_USERS = "manage_users"
    MANAGE_SETTINGS = "manage_settings"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_BACKUPS = "manage_backups"
    
    # Análisis
    VIEW_ANALYTICS = "view_analytics"
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_COMPARISONS = "view_comparisons"


class Role(str, Enum):
    """Roles del sistema"""
    USER = "user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class RolePermissions:
    """Permisos por rol"""
    
    PERMISSIONS = {
        Role.USER: {
            Permission.CREATE_VALIDATION,
            Permission.VIEW_VALIDATION,
            Permission.DELETE_VALIDATION,
            Permission.EXPORT_VALIDATION,
            Permission.CONNECT_SOCIAL_MEDIA,
            Permission.DISCONNECT_SOCIAL_MEDIA,
            Permission.VIEW_CONNECTIONS,
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
            Permission.VIEW_DASHBOARD,
        },
        Role.PREMIUM_USER: {
            Permission.CREATE_VALIDATION,
            Permission.VIEW_VALIDATION,
            Permission.DELETE_VALIDATION,
            Permission.EXPORT_VALIDATION,
            Permission.CONNECT_SOCIAL_MEDIA,
            Permission.DISCONNECT_SOCIAL_MEDIA,
            Permission.VIEW_CONNECTIONS,
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_COMPARISONS,
        },
        Role.ADMIN: {
            # Todos los permisos
            *[p for p in Permission],
        },
        Role.ANALYST: {
            Permission.VIEW_VALIDATION,
            Permission.VIEW_REPORTS,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_COMPARISONS,
            Permission.VIEW_AUDIT_LOGS,
        },
        Role.VIEWER: {
            Permission.VIEW_VALIDATION,
            Permission.VIEW_REPORTS,
            Permission.VIEW_DASHBOARD,
        }
    }
    
    @classmethod
    def get_permissions(cls, role: Role) -> Set[Permission]:
        """
        Obtener permisos de un rol
        
        Args:
            role: Rol
            
        Returns:
            Set de permisos
        """
        return cls.PERMISSIONS.get(role, set())


class PermissionManager:
    """Gestor de permisos"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._user_roles: Dict[UUID, List[Role]] = defaultdict(list)
        self._user_permissions: Dict[UUID, Set[Permission]] = defaultdict(set)
        logger.info("PermissionManager initialized")
    
    def assign_role(
        self,
        user_id: UUID,
        role: Role
    ) -> None:
        """
        Asignar rol a usuario
        
        Args:
            user_id: ID del usuario
            role: Rol a asignar
        """
        if role not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role)
            self._update_permissions(user_id)
            logger.info("Role assigned", user_id=str(user_id), role=role.value)
    
    def remove_role(
        self,
        user_id: UUID,
        role: Role
    ) -> None:
        """
        Remover rol de usuario
        
        Args:
            user_id: ID del usuario
            role: Rol a remover
        """
        if role in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role)
            self._update_permissions(user_id)
            logger.info("Role removed", user_id=str(user_id), role=role.value)
    
    def _update_permissions(self, user_id: UUID) -> None:
        """Actualizar permisos del usuario"""
        permissions = set()
        
        for role in self._user_roles[user_id]:
            permissions.update(RolePermissions.get_permissions(role))
        
        self._user_permissions[user_id] = permissions
    
    def has_permission(
        self,
        user_id: UUID,
        permission: Permission
    ) -> bool:
        """
        Verificar si usuario tiene permiso
        
        Args:
            user_id: ID del usuario
            permission: Permiso a verificar
            
        Returns:
            True si tiene permiso
        """
        return permission in self._user_permissions[user_id]
    
    def get_user_permissions(self, user_id: UUID) -> Set[Permission]:
        """
        Obtener permisos del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Set de permisos
        """
        return self._user_permissions[user_id].copy()
    
    def get_user_roles(self, user_id: UUID) -> List[Role]:
        """
        Obtener roles del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de roles
        """
        return self._user_roles[user_id].copy()
    
    def require_permission(
        self,
        user_id: UUID,
        permission: Permission
    ) -> bool:
        """
        Requerir permiso (lanzar excepción si no tiene)
        
        Args:
            user_id: ID del usuario
            permission: Permiso requerido
            
        Returns:
            True si tiene permiso
            
        Raises:
            PermissionDeniedError si no tiene permiso
        """
        if not self.has_permission(user_id, permission):
            from .exceptions import PsychologicalValidationError
            raise PsychologicalValidationError(
                f"Permission denied: {permission.value}",
                error_code="PERMISSION_DENIED",
                details={"permission": permission.value, "user_id": str(user_id)}
            )
        return True


# Instancia global del gestor de permisos
permission_manager = PermissionManager()




