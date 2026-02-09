"""
Permission Manager for Flux2 Clothing Changer
=============================================

Advanced permission management system.
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Role:
    """Role definition."""
    role_id: str
    name: str
    permissions: Set[Permission]
    created_at: float = time.time()
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UserRole:
    """User role assignment."""
    user_id: str
    role_id: str
    assigned_at: float = time.time()
    expires_at: Optional[float] = None


class PermissionManager:
    """Advanced permission management system."""
    
    def __init__(self):
        """Initialize permission manager."""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[UserRole]] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
    
    def create_role(
        self,
        role_id: str,
        name: str,
        permissions: Set[Permission],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Role:
        """
        Create role.
        
        Args:
            role_id: Role identifier
            name: Role name
            permissions: Set of permissions
            metadata: Optional metadata
            
        Returns:
            Created role
        """
        role = Role(
            role_id=role_id,
            name=name,
            permissions=permissions,
            metadata=metadata or {},
        )
        
        self.roles[role_id] = role
        logger.info(f"Created role: {role_id}")
        return role
    
    def assign_role(
        self,
        user_id: str,
        role_id: str,
        expires_at: Optional[float] = None,
    ) -> UserRole:
        """
        Assign role to user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            expires_at: Optional expiration timestamp
            
        Returns:
            User role assignment
        """
        if role_id not in self.roles:
            raise ValueError(f"Role not found: {role_id}")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        user_role = UserRole(
            user_id=user_id,
            role_id=role_id,
            expires_at=expires_at,
        )
        
        self.user_roles[user_id].append(user_role)
        logger.info(f"Assigned role {role_id} to user {user_id}")
        return user_role
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None,
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            resource: Optional resource identifier
            
        Returns:
            True if has permission
        """
        # Check resource-specific permissions
        if resource and resource in self.resource_permissions:
            if user_id in self.resource_permissions[resource]:
                if permission in self.resource_permissions[resource][user_id]:
                    return True
        
        # Check role-based permissions
        if user_id not in self.user_roles:
            return False
        
        for user_role in self.user_roles[user_id]:
            # Check expiration
            if user_role.expires_at and user_role.expires_at < time.time():
                continue
            
            role = self.roles.get(user_role.role_id)
            if role and permission in role.permissions:
                return True
        
        return False
    
    def set_resource_permission(
        self,
        resource: str,
        user_id: str,
        permissions: Set[Permission],
    ) -> None:
        """
        Set resource-specific permissions.
        
        Args:
            resource: Resource identifier
            user_id: User identifier
            permissions: Set of permissions
        """
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = {}
        
        self.resource_permissions[resource][user_id] = permissions
        logger.info(f"Set permissions for user {user_id} on resource {resource}")
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Set of permissions
        """
        permissions = set()
        
        if user_id in self.user_roles:
            for user_role in self.user_roles[user_id]:
                if user_role.expires_at and user_role.expires_at < time.time():
                    continue
                
                role = self.roles.get(user_role.role_id)
                if role:
                    permissions.update(role.permissions)
        
        # Add resource-specific permissions
        for resource_perms in self.resource_permissions.values():
            if user_id in resource_perms:
                permissions.update(resource_perms[user_id])
        
        return permissions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get permission manager statistics."""
        return {
            "total_roles": len(self.roles),
            "total_user_assignments": sum(len(roles) for roles in self.user_roles.values()),
            "resources_with_permissions": len(self.resource_permissions),
        }


