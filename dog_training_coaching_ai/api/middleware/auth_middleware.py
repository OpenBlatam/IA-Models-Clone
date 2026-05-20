"""
Authentication Middleware
=========================
"""

from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[str]:
    """Verificar token de autenticación (opcional)."""
    if not credentials:
        return None
    
    # Implementar lógica de verificación de token aquí
    # Por ahora retorna el token
    return credentials.credentials


async def optional_auth(request: Request) -> Optional[str]:
    """Autenticación opcional."""
    credentials = await security(request)
    return await verify_token(credentials)

