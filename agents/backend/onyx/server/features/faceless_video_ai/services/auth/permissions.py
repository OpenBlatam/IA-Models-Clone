"""
Permission System
"""

from enum import Enum
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Available permissions"""
    # Video permissions
    GENERATE_VIDEO = "generate:video"
    GENERATE_BATCH = "generate:batch"
    VIEW_VIDEOS = "view:videos"
    DELETE_VIDEOS = "delete:videos"
    
    # Template permissions
    USE_TEMPLATES = "use:templates"
    CREATE_TEMPLATES = "create:templates"
    EDIT_TEMPLATES = "edit:templates"
    
    # Admin permissions
    VIEW_ANALYTICS = "view:analytics"
    MANAGE_USERS = "manage:users"
    MANAGE_SETTINGS = "manage:settings"
    
    # API permissions
    USE_API = "use:api"
    UNLIMITED_RATE = "unlimited:rate"


class Role:
    """Role with permissions"""
    
    ROLES = {
        "user": [
            Permission.GENERATE_VIDEO,
            Permission.VIEW_VIDEOS,
            Permission.USE_TEMPLATES,
            Permission.USE_API,
        ],
        "premium": [
            Permission.GENERATE_VIDEO,
            Permission.GENERATE_BATCH,
            Permission.VIEW_VIDEOS,
            Permission.USE_TEMPLATES,
            Permission.USE_API,
        ],
        "admin": [
            Permission.GENERATE_VIDEO,
            Permission.GENERATE_BATCH,
            Permission.VIEW_VIDEOS,
            Permission.DELETE_VIDEOS,
            Permission.USE_TEMPLATES,
            Permission.CREATE_TEMPLATES,
            Permission.EDIT_TEMPLATES,
            Permission.VIEW_ANALYTICS,
            Permission.MANAGE_USERS,
            Permission.MANAGE_SETTINGS,
            Permission.USE_API,
            Permission.UNLIMITED_RATE,
        ],
    }
    
    @classmethod
    def get_permissions(cls, roles: List[str]) -> List[Permission]:
        """Get permissions for roles"""
        permissions = set()
        for role in roles:
            if role in cls.ROLES:
                permissions.update(cls.ROLES[role])
        return list(permissions)
    
    @classmethod
    def has_permission(cls, roles: List[str], permission: Permission) -> bool:
        """Check if roles have permission"""
        permissions = cls.get_permissions(roles)
        return permission in permissions


def check_permission(roles: List[str], permission: Permission) -> bool:
    """Check if user has permission"""
    return Role.has_permission(roles, permission)

