"""
Role-Based Access Control (RBAC)
================================

RBAC implementation for fine-grained access control.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import uuid

from .types import SecurityLevel, AuditEvent

logger = logging.getLogger(__name__)

class Permission:
    """Permission definition."""
    
    def __init__(self, name: str, description: str, resource: str, action: str, conditions: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.resource = resource
        self.action = action
        self.conditions = conditions or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def matches(self, resource: str, action: str, context: Dict[str, Any] = None) -> bool:
        """Check if permission matches resource and action."""
        if self.resource != resource or self.action != action:
            return False
        
        # Check conditions
        if self.conditions and context:
            for key, expected_value in self.conditions.items():
                if key not in context or context[key] != expected_value:
                    return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "resource": self.resource,
            "action": self.action,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class Role:
    """Role definition."""
    
    def __init__(self, name: str, description: str, level: SecurityLevel = SecurityLevel.MEDIUM):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.level = level
        self.permissions: Set[str] = set()  # Permission IDs
        self.inherited_roles: Set[str] = set()  # Role IDs
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_permission(self, permission_id: str):
        """Add permission to role."""
        self.permissions.add(permission_id)
        self.updated_at = datetime.now()
    
    def remove_permission(self, permission_id: str):
        """Remove permission from role."""
        self.permissions.discard(permission_id)
        self.updated_at = datetime.now()
    
    def inherit_role(self, role_id: str):
        """Inherit permissions from another role."""
        self.inherited_roles.add(role_id)
        self.updated_at = datetime.now()
    
    def uninherit_role(self, role_id: str):
        """Stop inheriting from a role."""
        self.inherited_roles.discard(role_id)
        self.updated_at = datetime.now()
    
    def get_all_permissions(self, role_registry: 'RoleRegistry') -> Set[str]:
        """Get all permissions including inherited ones."""
        all_permissions = self.permissions.copy()
        
        for inherited_role_id in self.inherited_roles:
            inherited_role = role_registry.get_role(inherited_role_id)
            if inherited_role:
                all_permissions.update(inherited_role.get_all_permissions(role_registry))
        
        return all_permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "permissions": list(self.permissions),
            "inherited_roles": list(self.inherited_roles),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class UserRole:
    """User role assignment."""
    
    def __init__(self, user_id: str, role_id: str, assigned_by: str, expires_at: Optional[datetime] = None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.role_id = role_id
        self.assigned_by = assigned_by
        self.assigned_at = datetime.now()
        self.expires_at = expires_at
        self.is_active = True
    
    def is_expired(self) -> bool:
        """Check if role assignment is expired."""
        return self.expires_at and self.expires_at <= datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "role_id": self.role_id,
            "assigned_by": self.assigned_by,
            "assigned_at": self.assigned_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active
        }

class RoleRegistry:
    """Role and permission registry."""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self._lock = asyncio.Lock()
    
    async def create_permission(
        self, 
        name: str, 
        description: str, 
        resource: str, 
        action: str,
        conditions: Dict[str, Any] = None
    ) -> str:
        """Create a new permission."""
        try:
            permission = Permission(name, description, resource, action, conditions)
            
            async with self._lock:
                self.permissions[permission.id] = permission
            
            logger.info(f"Created permission: {name}")
            return permission.id
            
        except Exception as e:
            logger.error(f"Failed to create permission: {str(e)}")
            raise
    
    async def create_role(
        self, 
        name: str, 
        description: str, 
        level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> str:
        """Create a new role."""
        try:
            role = Role(name, description, level)
            
            async with self._lock:
                self.roles[role.id] = role
            
            logger.info(f"Created role: {name}")
            return role.id
            
        except Exception as e:
            logger.error(f"Failed to create role: {str(e)}")
            raise
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        return self.permissions.get(permission_id)
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self.roles.get(role_id)
    
    def get_permission_by_name(self, name: str) -> Optional[Permission]:
        """Get permission by name."""
        for permission in self.permissions.values():
            if permission.name == name:
                return permission
        return None
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        for role in self.roles.values():
            if role.name == name:
                return role
        return None
    
    async def assign_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """Assign permission to role."""
        try:
            async with self._lock:
                role = self.roles.get(role_id)
                permission = self.permissions.get(permission_id)
                
                if not role or not permission:
                    return False
                
                role.add_permission(permission_id)
            
            logger.info(f"Assigned permission {permission_id} to role {role_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign permission to role: {str(e)}")
            return False
    
    async def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """Remove permission from role."""
        try:
            async with self._lock:
                role = self.roles.get(role_id)
                if not role:
                    return False
                
                role.remove_permission(permission_id)
            
            logger.info(f"Removed permission {permission_id} from role {role_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove permission from role: {str(e)}")
            return False
    
    async def inherit_role(self, child_role_id: str, parent_role_id: str) -> bool:
        """Make a role inherit from another role."""
        try:
            async with self._lock:
                child_role = self.roles.get(child_role_id)
                parent_role = self.roles.get(parent_role_id)
                
                if not child_role or not parent_role:
                    return False
                
                # Check for circular inheritance
                if self._would_create_circular_inheritance(child_role_id, parent_role_id):
                    return False
                
                child_role.inherit_role(parent_role_id)
            
            logger.info(f"Role {child_role_id} now inherits from {parent_role_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to inherit role: {str(e)}")
            return False
    
    def _would_create_circular_inheritance(self, child_role_id: str, parent_role_id: str) -> bool:
        """Check if inheritance would create a circular dependency."""
        # Simple check - in a real implementation, you'd want a more thorough check
        parent_role = self.roles.get(parent_role_id)
        if parent_role and child_role_id in parent_role.inherited_roles:
            return True
        return False
    
    def list_permissions(self) -> List[Dict[str, Any]]:
        """List all permissions."""
        return [permission.to_dict() for permission in self.permissions.values()]
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles."""
        return [role.to_dict() for role in self.roles.values()]

class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.role_registry = RoleRegistry()
        self.user_roles: Dict[str, List[UserRole]] = {}  # user_id -> UserRole list
        self._lock = asyncio.Lock()
    
    async def initialize_default_roles(self):
        """Initialize default roles and permissions."""
        try:
            # Create default permissions
            permissions = [
                ("read_agents", "Read business agents", "agents", "read"),
                ("write_agents", "Create/update business agents", "agents", "write"),
                ("delete_agents", "Delete business agents", "agents", "delete"),
                ("execute_agents", "Execute agent capabilities", "agents", "execute"),
                ("read_workflows", "Read workflows", "workflows", "read"),
                ("write_workflows", "Create/update workflows", "workflows", "write"),
                ("delete_workflows", "Delete workflows", "workflows", "delete"),
                ("execute_workflows", "Execute workflows", "workflows", "execute"),
                ("read_documents", "Read documents", "documents", "read"),
                ("write_documents", "Create/update documents", "documents", "write"),
                ("delete_documents", "Delete documents", "documents", "delete"),
                ("admin_system", "System administration", "system", "admin"),
                ("read_metrics", "Read system metrics", "metrics", "read"),
                ("manage_users", "Manage users and roles", "users", "manage")
            ]
            
            for name, description, resource, action in permissions:
                await self.role_registry.create_permission(name, description, resource, action)
            
            # Create default roles
            roles = [
                ("admin", "System Administrator", SecurityLevel.CRITICAL),
                ("manager", "Business Manager", SecurityLevel.HIGH),
                ("analyst", "Business Analyst", SecurityLevel.MEDIUM),
                ("user", "Regular User", SecurityLevel.LOW),
                ("viewer", "Read-only User", SecurityLevel.LOW)
            ]
            
            role_ids = {}
            for name, description, level in roles:
                role_id = await self.role_registry.create_role(name, description, level)
                role_ids[name] = role_id
            
            # Assign permissions to roles
            await self._assign_default_permissions(role_ids)
            
            logger.info("Initialized default roles and permissions")
            
        except Exception as e:
            logger.error(f"Failed to initialize default roles: {str(e)}")
            raise
    
    async def _assign_default_permissions(self, role_ids: Dict[str, str]):
        """Assign default permissions to roles."""
        try:
            # Admin gets all permissions
            admin_role_id = role_ids["admin"]
            for permission in self.role_registry.permissions.values():
                await self.role_registry.assign_permission_to_role(admin_role_id, permission.id)
            
            # Manager permissions
            manager_role_id = role_ids["manager"]
            manager_permissions = [
                "read_agents", "write_agents", "execute_agents",
                "read_workflows", "write_workflows", "execute_workflows",
                "read_documents", "write_documents",
                "read_metrics", "manage_users"
            ]
            for perm_name in manager_permissions:
                permission = self.role_registry.get_permission_by_name(perm_name)
                if permission:
                    await self.role_registry.assign_permission_to_role(manager_role_id, permission.id)
            
            # Analyst permissions
            analyst_role_id = role_ids["analyst"]
            analyst_permissions = [
                "read_agents", "execute_agents",
                "read_workflows", "execute_workflows",
                "read_documents", "write_documents",
                "read_metrics"
            ]
            for perm_name in analyst_permissions:
                permission = self.role_registry.get_permission_by_name(perm_name)
                if permission:
                    await self.role_registry.assign_permission_to_role(analyst_role_id, permission.id)
            
            # User permissions
            user_role_id = role_ids["user"]
            user_permissions = [
                "read_agents", "execute_agents",
                "read_workflows", "execute_workflows",
                "read_documents", "write_documents"
            ]
            for perm_name in user_permissions:
                permission = self.role_registry.get_permission_by_name(perm_name)
                if permission:
                    await self.role_registry.assign_permission_to_role(user_role_id, permission.id)
            
            # Viewer permissions
            viewer_role_id = role_ids["viewer"]
            viewer_permissions = [
                "read_agents", "read_workflows", "read_documents", "read_metrics"
            ]
            for perm_name in viewer_permissions:
                permission = self.role_registry.get_permission_by_name(perm_name)
                if permission:
                    await self.role_registry.assign_permission_to_role(viewer_role_id, permission.id)
            
        except Exception as e:
            logger.error(f"Failed to assign default permissions: {str(e)}")
            raise
    
    async def assign_role_to_user(
        self, 
        user_id: str, 
        role_id: str, 
        assigned_by: str,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Assign role to user."""
        try:
            role = self.role_registry.get_role(role_id)
            if not role:
                return False
            
            user_role = UserRole(user_id, role_id, assigned_by, expires_at)
            
            async with self._lock:
                if user_id not in self.user_roles:
                    self.user_roles[user_id] = []
                self.user_roles[user_id].append(user_role)
            
            logger.info(f"Assigned role {role.name} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role to user: {str(e)}")
            return False
    
    async def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Remove role from user."""
        try:
            async with self._lock:
                if user_id not in self.user_roles:
                    return False
                
                user_roles = self.user_roles[user_id]
                for i, user_role in enumerate(user_roles):
                    if user_role.role_id == role_id and user_role.is_active:
                        user_roles[i].is_active = False
                        logger.info(f"Removed role {role_id} from user {user_id}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove role from user: {str(e)}")
            return False
    
    async def check_permission(
        self, 
        user_id: str, 
        resource: str, 
        action: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """Check if user has permission for resource and action."""
        try:
            async with self._lock:
                if user_id not in self.user_roles:
                    return False
                
                user_roles = self.user_roles[user_id]
                active_roles = [ur for ur in user_roles if ur.is_active and not ur.is_expired()]
                
                # Get all permissions for user's roles
                all_permissions = set()
                for user_role in active_roles:
                    role = self.role_registry.get_role(user_role.role_id)
                    if role:
                        all_permissions.update(role.get_all_permissions(self.role_registry))
                
                # Check if any permission matches
                for permission_id in all_permissions:
                    permission = self.role_registry.get_permission(permission_id)
                    if permission and permission.matches(resource, action, context):
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to check permission: {str(e)}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all permissions for a user."""
        try:
            async with self._lock:
                if user_id not in self.user_roles:
                    return []
                
                user_roles = self.user_roles[user_id]
                active_roles = [ur for ur in user_roles if ur.is_active and not ur.is_expired()]
                
                all_permissions = set()
                for user_role in active_roles:
                    role = self.role_registry.get_role(user_role.role_id)
                    if role:
                        all_permissions.update(role.get_all_permissions(self.role_registry))
                
                permissions = []
                for permission_id in all_permissions:
                    permission = self.role_registry.get_permission(permission_id)
                    if permission:
                        permissions.append(permission.to_dict())
                
                return permissions
                
        except Exception as e:
            logger.error(f"Failed to get user permissions: {str(e)}")
            return []
    
    async def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all roles for a user."""
        try:
            async with self._lock:
                if user_id not in self.user_roles:
                    return []
                
                user_roles = self.user_roles[user_id]
                active_roles = [ur for ur in user_roles if ur.is_active and not ur.is_expired()]
                
                roles = []
                for user_role in active_roles:
                    role = self.role_registry.get_role(user_role.role_id)
                    if role:
                        role_dict = role.to_dict()
                        role_dict["assigned_at"] = user_role.assigned_at.isoformat()
                        role_dict["expires_at"] = user_role.expires_at.isoformat() if user_role.expires_at else None
                        roles.append(role_dict)
                
                return roles
                
        except Exception as e:
            logger.error(f"Failed to get user roles: {str(e)}")
            return []
    
    async def cleanup_expired_roles(self):
        """Clean up expired role assignments."""
        try:
            expired_count = 0
            
            async with self._lock:
                for user_id, user_roles in self.user_roles.items():
                    for user_role in user_roles:
                        if user_role.is_expired() and user_role.is_active:
                            user_role.is_active = False
                            expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired role assignments")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired roles: {str(e)}")

# Global RBAC manager instance
rbac_manager = RBACManager()
