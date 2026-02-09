"""
Access Control Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Set
from datetime import datetime
from uuid import uuid4


class Permission(str, Enum):
    """Permission types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class Role:
    """Role definition"""
    
    def __init__(
        self,
        name: str,
        permissions: Set[Permission],
        description: Optional[str] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.permissions = permissions
        self.description = description
        self.created_at = datetime.utcnow()


class Resource:
    """Resource definition"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        owner_id: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.owner_id = owner_id


class AccessPolicy:
    """Access policy definition"""
    
    def __init__(
        self,
        resource: Resource,
        allowed_roles: List[str],
        allowed_users: Optional[List[str]] = None,
        conditions: Optional[Dict] = None
    ):
        self.id = str(uuid4())
        self.resource = resource
        self.allowed_roles = allowed_roles
        self.allowed_users = allowed_users or []
        self.conditions = conditions or {}
        self.created_at = datetime.utcnow()


class AccessControlBase(ABC):
    """Base interface for access control"""
    
    @abstractmethod
    async def check_permission(
        self,
        user_id: str,
        resource: Resource,
        permission: Permission
    ) -> bool:
        """Check if user has permission on resource"""
        pass
    
    @abstractmethod
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles for a user"""
        pass
    
    @abstractmethod
    async def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign role to user"""
        pass

