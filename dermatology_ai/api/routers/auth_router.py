"""
Authentication Router - Handles user registration, login, and authentication endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel, EmailStr
import logging

from api.services_locator import get_service
from utils.logger import logger
from services.auth_manager import AuthManager

router = APIRouter(prefix="/dermatology", tags=["auth"])


# Request/Response Models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[dict] = None


def get_auth_manager() -> AuthManager:
    """Get AuthManager service from service locator"""
    try:
        auth_manager = get_service("auth_manager")
        if auth_manager is None:
            # Create a new instance if not in service locator
            auth_manager = AuthManager()
        return auth_manager
    except Exception as e:
        logger.warning(f"AuthManager not found in service locator, creating new instance: {e}")
        return AuthManager()


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """
    Register a new user
    
    Args:
        request: Registration request with email, password, and name
        
    Returns:
        AuthResponse with token and user info
    """
    try:
        auth_manager = get_auth_manager()
        
        # Check if user already exists
        existing_user = auth_manager.get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="El email ya está registrado"
            )
        
        # Create user
        metadata = {"name": request.name}
        user = auth_manager.create_user(
            email=request.email,
            password=request.password,
            metadata=metadata
        )
        
        # Generate token
        token = auth_manager.generate_token(user.id)
        
        return JSONResponse(content={
            "success": True,
            "message": "Usuario registrado exitosamente",
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.metadata.get("name", "") if user.metadata else "",
                "created_at": user.created_at
            }
        })
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in register: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al registrar usuario"
        )


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Login user and get authentication token
    
    Args:
        request: Login request with email and password
        
    Returns:
        AuthResponse with token and user info
    """
    try:
        auth_manager = get_auth_manager()
        
        # Authenticate user
        token = auth_manager.authenticate(request.email, request.password)
        
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Email o contraseña incorrectos"
            )
        
        # Get user info
        user = auth_manager.get_user_by_email(request.email)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Usuario no encontrado"
            )
        
        return JSONResponse(content={
            "success": True,
            "message": "Inicio de sesión exitoso",
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.metadata.get("name", "") if user.metadata else "",
                "created_at": user.created_at,
                "last_login": user.last_login
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al iniciar sesión"
        )


@router.get("/auth/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Get current authenticated user info
    
    Args:
        authorization: Bearer token in Authorization header
        
    Returns:
        User information
    """
    try:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Token de autenticación requerido"
            )
        
        # Extract token from "Bearer <token>"
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Formato de token inválido. Use: Bearer <token>"
            )
        
        token = authorization.replace("Bearer ", "").strip()
        
        auth_manager = get_auth_manager()
        user_id = auth_manager.verify_token(token)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Token inválido o expirado"
            )
        
        user = auth_manager.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="Usuario no encontrado"
            )
        
        return JSONResponse(content={
            "success": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.metadata.get("name", "") if user.metadata else "",
                "created_at": user.created_at,
                "last_login": user.last_login,
                "is_active": user.is_active
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al obtener información del usuario"
        )

