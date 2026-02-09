"""Middleware para Suno Clone AI"""

from .base_middleware import BaseMiddleware
from .auth_middleware import (
    AuthMiddleware,
    get_current_user,
    require_auth,
    require_role
)

__all__ = [
    "BaseMiddleware",
    "AuthMiddleware",
    "get_current_user",
    "require_auth",
    "require_role",
]

