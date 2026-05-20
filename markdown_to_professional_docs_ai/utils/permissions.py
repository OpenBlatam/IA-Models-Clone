"""Permissions and roles system"""
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Document permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CONVERT = "convert"
    EXPORT = "export"
    ANNOTATE = "annotate"
    MANAGE_VERSIONS = "manage_versions"
    MANAGE_BACKUPS = "manage_backups"


class Role:
    """User role with permissions"""
    
    def __init__(self, name: str, permissions: Set[Permission]):
        """
        Initialize role
        
        Args:
            name: Role name
            permissions: Set of permissions
        """
        self.name = name
        self.permissions = permissions
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission"""
        return permission in self.permissions or Permission.ADMIN in self.permissions


class PermissionManager:
    """Manage permissions and access control"""
    
    def __init__(self):
        # Default roles
        self.roles = {
            "admin": Role("admin", set(Permission)),
            "user": Role("user", {
                Permission.READ,
                Permission.CONVERT,
                Permission.EXPORT,
                Permission.ANNOTATE
            }),
            "viewer": Role("viewer", {Permission.READ}),
            "editor": Role("editor", {
                Permission.READ,
                Permission.WRITE,
                Permission.CONVERT,
                Permission.EXPORT,
                Permission.ANNOTATE,
                Permission.MANAGE_VERSIONS
            })
        }
        
        # User roles mapping
        self.user_roles: Dict[str, str] = {}
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """
        Assign role to user
        
        Args:
            user_id: User identifier
            role_name: Role name
            
        Returns:
            True if assigned, False otherwise
        """
        if role_name in self.roles:
            self.user_roles[user_id] = role_name
            return True
        return False
    
    def get_user_role(self, user_id: str) -> Optional[Role]:
        """Get user role"""
        role_name = self.user_roles.get(user_id, "user")
        return self.roles.get(role_name)
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if user has permission
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        role = self.get_user_role(user_id)
        if role:
            return role.has_permission(permission)
        return False
    
    def create_role(
        self,
        name: str,
        permissions: Set[Permission]
    ) -> Role:
        """
        Create custom role
        
        Args:
            name: Role name
            permissions: Set of permissions
            
        Returns:
            Created role
        """
        role = Role(name, permissions)
        self.roles[name] = role
        return role
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles"""
        return [
            {
                "name": role.name,
                "permissions": [p.value for p in role.permissions]
            }
            for role in self.roles.values()
        ]


# Global permission manager
_permission_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """Get global permission manager"""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager

