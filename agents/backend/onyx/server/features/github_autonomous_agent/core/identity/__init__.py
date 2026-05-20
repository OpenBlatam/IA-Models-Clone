"""
Identity Management System

Manages AI agent identities and personalities for consistent behavior.
"""

from .base_identity import BaseIdentity, IdentityConfig
from .identity_manager import IdentityManager, get_identity_manager
from .kiro_identity import KiroIdentity

__all__ = [
    "BaseIdentity",
    "IdentityConfig",
    "IdentityManager",
    "get_identity_manager",
    "KiroIdentity",
]


