"""
MCP Auth - Autenticación y autorización para el servidor MCP
============================================================

Lógica de autenticación y verificación de permisos para el servidor MCP.
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

try:
    from .auth import AuthManager, Role
except ImportError:
    AuthManager = None
    Role = None


async def authenticate_request(request: Request, auth_manager: Optional[Any]) -> Optional[str]:
    """Autenticar request si auth está habilitado"""
    if not auth_manager:
        return None
    
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if api_key:
        username = auth_manager.api_keys.get(api_key)
        if username:
            user = auth_manager.users.get(username)
            if user and user.enabled:
                return username
    
    return None


async def check_permission(
    username: Optional[str],
    permission: str,
    auth_manager: Optional[Any]
) -> bool:
    """Verificar permisos"""
    if not auth_manager or not username:
        return True
    
    user = auth_manager.users.get(username)
    if not user:
        return False
    
    if not Role:
        return True
    
    if permission == "execute_command":
        return user.role in [Role.ADMIN, Role.USER]
    elif permission == "read_status":
        return user.role in [Role.ADMIN, Role.USER, Role.READONLY]
    elif permission == "read_metrics":
        return user.role in [Role.ADMIN]
    
    return False


def require_auth(auth_manager: Optional[Any]) -> bool:
    """Verificar si la autenticación es requerida"""
    return auth_manager is not None


def require_admin(auth_manager: Optional[Any], username: Optional[str]) -> bool:
    """Verificar si el usuario es admin"""
    if not auth_manager or not username:
        return False
    
    if not Role:
        return False
    
    user = auth_manager.users.get(username)
    return user and user.role == Role.ADMIN




