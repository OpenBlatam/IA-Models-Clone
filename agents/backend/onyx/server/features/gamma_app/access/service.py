"""
Access Control Service Implementation
"""

from typing import List, Optional
import logging

from .base import (
    AccessControlBase,
    Permission,
    Role,
    Resource,
    AccessPolicy
)

logger = logging.getLogger(__name__)


class AccessControlService(AccessControlBase):
    """Access control service implementation"""
    
    def __init__(self, db=None, cache=None):
        """Initialize access control service"""
        self.db = db
        self.cache = cache
        self._policies: dict = {}
        self._user_roles: dict = {}
    
    async def check_permission(
        self,
        user_id: str,
        resource: Resource,
        permission: Permission
    ) -> bool:
        """Check if user has permission on resource"""
        try:
            # Get user roles
            roles = await self.get_user_roles(user_id)
            
            # Check if user is resource owner
            if resource.owner_id == user_id:
                return True
            
            # Check policies
            resource_key = f"{resource.resource_type}:{resource.resource_id}"
            policy = self._policies.get(resource_key)
            
            if policy:
                # Check role-based access
                user_role_names = [role.name for role in roles]
                if any(role in policy.allowed_roles for role in user_role_names):
                    return True
                
                # Check user-based access
                if user_id in policy.allowed_users:
                    return True
            
            # Check role permissions
            for role in roles:
                if permission in role.permissions:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles for a user"""
        try:
            if user_id in self._user_roles:
                return self._user_roles[user_id]
            
            # TODO: Load from database
            return []
            
        except Exception as e:
            logger.error(f"Error getting user roles: {e}")
            return []
    
    async def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign role to user"""
        try:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = []
            
            self._user_roles[user_id].append(role)
            
            # TODO: Persist to database
            
            return True
            
        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            return False
    
    async def create_policy(self, policy: AccessPolicy) -> bool:
        """Create access policy"""
        try:
            resource_key = f"{policy.resource.resource_type}:{policy.resource.resource_id}"
            self._policies[resource_key] = policy
            return True
        except Exception as e:
            logger.error(f"Error creating policy: {e}")
            return False

