"""
Rutas para Seguridad
=====================

Endpoints para autenticación y autorización.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from ..core.security import get_security_manager, SecurityManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/security",
    tags=["Security"]
)

security = HTTPBearer()


class CreateUserRequest(BaseModel):
    """Request para crear usuario"""
    username: str = Field(..., description="Nombre de usuario")
    password: str = Field(..., description="Contraseña")
    roles: list[str] = Field(["user"], description="Roles")


class LoginRequest(BaseModel):
    """Request para login"""
    username: str = Field(..., description="Nombre de usuario")
    password: str = Field(..., description="Contraseña")


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    manager: SecurityManager = Depends(get_security_manager)
):
    """Crear nuevo usuario"""
    try:
        user = manager.create_user(
            request.username,
            request.password,
            request.roles
        )
        
        return {
            "status": "created",
            "username": user.username,
            "roles": user.roles
        }
    except Exception as e:
        logger.error(f"Error creando usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login(
    request: LoginRequest,
    manager: SecurityManager = Depends(get_security_manager)
):
    """Autenticar usuario"""
    user = manager.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    token = manager.generate_token(user)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "roles": user.roles
    }


@router.post("/api-key/{username}")
async def generate_api_key(
    username: str,
    manager: SecurityManager = Depends(get_security_manager)
):
    """Generar API key para usuario"""
    try:
        api_key = manager.generate_api_key(username)
        return {
            "api_key": api_key,
            "username": username
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/verify")
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    manager: SecurityManager = Depends(get_security_manager)
):
    """Verificar token"""
    token = credentials.credentials
    payload = manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    
    return {
        "valid": True,
        "username": payload.get("username"),
        "roles": payload.get("roles")
    }


@router.post("/verify-api-key")
async def verify_api_key(
    api_key: str = Header(..., alias="X-API-Key"),
    manager: SecurityManager = Depends(get_security_manager)
):
    """Verificar API key"""
    user = manager.authenticate_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="API key inválida")
    
    return {
        "valid": True,
        "username": user.username,
        "roles": user.roles
    }
















