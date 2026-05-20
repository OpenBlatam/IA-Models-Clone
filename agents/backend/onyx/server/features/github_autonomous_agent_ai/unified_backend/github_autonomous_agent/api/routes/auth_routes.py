"""
Auth Routes - Rutas para autenticación y gestión de API keys.
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional, List
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services import AuthService, Permission, Role
from config.logging_config import get_logger
from config.di_setup import get_service
from core.error_codes import ErrorCode, create_error_response

router = APIRouter()
logger = get_logger(__name__)


class CreateAPIKeyRequest(BaseModel):
    """Request para crear API key."""
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class CreateAPIKeyResponse(BaseModel):
    """Response de creación de API key."""
    key: str
    key_id: str
    name: str
    permissions: List[str]
    expires_at: Optional[str]


@router.post("/api-keys", response_model=CreateAPIKeyResponse)
@handle_api_errors
async def create_api_key(
    request: CreateAPIKeyRequest,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """
    Crear nueva API key.
    
    Args:
        request: Datos de la API key
        authorization: Header de autorización
        
    Returns:
        API key creada
    """
    try:
        auth_service: AuthService = get_service("auth_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service no disponible")
    
    # Validar permisos
    if authorization:
        # Extraer y validar token
        if not authorization.startswith("Bearer "):
            error = create_error_response(ErrorCode.AUTH_INVALID_TOKEN, http_status=401)
            raise HTTPException(status_code=error.http_status, detail=error.to_dict())
    
    # Validar permisos
    permissions = []
    for perm_str in request.permissions:
        try:
            permissions.append(Permission(perm_str))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Permiso inválido: {perm_str}"
            )
    
    # Crear usuario temporal si no existe (en producción, usar autenticación real)
    user_id = "default_user"  # En producción, obtener del token
    key_plain, api_key = auth_service.create_api_key(
        user_id=user_id,
        name=request.name,
        permissions=permissions,
        expires_in_days=request.expires_in_days
    )
    
    return CreateAPIKeyResponse(
        key=key_plain,
        key_id=api_key.key_id,
        name=api_key.name,
        permissions=[p.value for p in api_key.permissions],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None
    )


@router.get("/api-keys")
@handle_api_errors
async def list_api_keys(
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """
    Listar API keys del usuario.
    
    Args:
        authorization: Header de autorización
        
    Returns:
        Lista de API keys
    """
    try:
        auth_service: AuthService = get_service("auth_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service no disponible")
    
    # En producción, obtener user_id del token
    user_id = "default_user"
    keys = auth_service.get_user_api_keys(user_id)
    
    return {
        "total": len(keys),
        "keys": [
            {
                "key_id": key.key_id,
                "name": key.name,
                "permissions": [p.value for p in key.permissions],
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "is_expired": key.is_expired()
            }
            for key in keys
        ]
    }


@router.delete("/api-keys/{key_id}")
@handle_api_errors
async def revoke_api_key(
    key_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """
    Revocar API key.
    
    Args:
        key_id: ID de la API key
        authorization: Header de autorización
        
    Returns:
        Confirmación
    """
    try:
        auth_service: AuthService = get_service("auth_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service no disponible")
    
    success = auth_service.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key no encontrada")
    
    return {"message": "API key revocada exitosamente"}


@router.get("/auth/stats")
@handle_api_errors
async def get_auth_stats():
    """
    Obtener estadísticas de autenticación.
    
    Returns:
        Estadísticas
    """
    try:
        auth_service: AuthService = get_service("auth_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service no disponible")
    
    return auth_service.get_stats()
