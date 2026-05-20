"""
Advanced Permissions System
===========================

Advanced permissions and role-based access control system.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PermissionAction(Enum):
    """Permission actions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Permission:
    """Permission definition."""
    resource: str
    action: PermissionAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.resource, self.action.value))
    
    def __eq__(self, other):
        if not isinstance(other, Permission):
            return False
        return self.resource == other.resource and self.action == other.action


@dataclass
class Role:
    """Role definition."""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        self.permissions.discard(permission)
    
    def has_permission(self, resource: str, action: PermissionAction) -> bool:
        """Check if role has permission."""
        for perm in self.permissions:
            if perm.resource == resource and perm.action == action:
                return True
        return False


@dataclass
class User:
    """User definition."""
    user_id: str
    username: str
    roles: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_role(self, role_name: str):
        """Add role to user."""
        if role_name not in self.roles:
            self.roles.append(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user."""
        if role_name in self.roles:
            self.roles.remove(role_name)
    
    def add_permission(self, permission: Permission):
        """Add direct permission to user."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove direct permission from user."""
        self.permissions.discard(permission)


class AdvancedPermissionsManager:
    """Advanced permissions manager with RBAC."""
    
    def __init__(self):
        """Initialize advanced permissions manager."""
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.resources: Set[str] = set()
    
    def register_role(self, role: Role):
        """
        Register a role.
        
        Args:
            role: Role to register
        """
        self.roles[role.name] = role
        logger.info(f"Registered role: {role.name}")
    
    def get_role(self, name: str) -> Optional[Role]:
        """
        Get role by name.
        
        Args:
            name: Role name
            
        Returns:
            Role or None
        """
        return self.roles.get(name)
    
    def register_user(self, user: User):
        """
        Register a user.
        
        Args:
            user: User to register
        """
        self.users[user.user_id] = user
        logger.info(f"Registered user: {user.user_id}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User or None
        """
        return self.users.get(user_id)
    
    def check_permission(
        self,
        user_id: str,
        resource: str,
        action: PermissionAction,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            user_id: User ID
            resource: Resource name
            action: Permission action
            context: Optional context for conditional permissions
            
        Returns:
            True if user has permission
        """
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Check direct permissions
        for perm in user.permissions:
            if perm.resource == resource and perm.action == action:
                # Check conditions if any
                if perm.conditions and context:
                    if not self._check_conditions(perm.conditions, context):
                        continue
                return True
        
        # Check role permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.has_permission(resource, action):
                return True
        
        return False
    
    def _check_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check permission conditions."""
        for key, expected_value in conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        return True
    
    def grant_permission(
        self,
        user_id: str,
        resource: str,
        action: PermissionAction,
        conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Grant permission to user.
        
        Args:
            user_id: User ID
            resource: Resource name
            action: Permission action
            conditions: Optional conditions
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        permission = Permission(
            resource=resource,
            action=action,
            conditions=conditions or {}
        )
        user.add_permission(permission)
        self.resources.add(resource)
    
    def revoke_permission(
        self,
        user_id: str,
        resource: str,
        action: PermissionAction
    ):
        """
        Revoke permission from user.
        
        Args:
            user_id: User ID
            resource: Resource name
            action: Permission action
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        permission = Permission(resource=resource, action=action)
        user.remove_permission(permission)
    
    def assign_role(self, user_id: str, role_name: str):
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role_name: Role name
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        if role_name not in self.roles:
            raise ValueError(f"Role not found: {role_name}")
        
        user.add_role(role_name)
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for user (including from roles).
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permissions
        """
        user = self.users.get(user_id)
        if not user:
            return set()
        
        permissions = user.permissions.copy()
        
        # Add role permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        
        return permissions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get permissions statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_users": len(self.users),
            "total_roles": len(self.roles),
            "total_resources": len(self.resources),
            "users_with_roles": sum(1 for u in self.users.values() if u.roles),
            "users_with_direct_permissions": sum(1 for u in self.users.values() if u.permissions)
        }



