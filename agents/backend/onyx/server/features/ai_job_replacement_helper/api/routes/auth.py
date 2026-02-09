"""
Authentication endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.auth import AuthService

router = APIRouter()
auth_service = AuthService()


@router.post("/register")
async def register(
    email: str,
    username: str,
    password: str
) -> Dict[str, Any]:
    """Registrar nuevo usuario"""
    try:
        user = auth_service.register_user(email, username, password)
        return {
            "user_id": user.id,
            "email": user.email,
            "username": user.username,
            "created_at": user.created_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login(email: str, password: str) -> Dict[str, Any]:
    """Iniciar sesión"""
    try:
        session = auth_service.login(email, password)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user = auth_service.get_user(session.user_id)
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "expires_at": session.expires_at.isoformat(),
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def logout(session_id: str) -> Dict[str, Any]:
    """Cerrar sesión"""
    try:
        success = auth_service.logout(session_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify/{session_id}")
async def verify_session(session_id: str) -> Dict[str, Any]:
    """Verificar sesión"""
    try:
        user = auth_service.verify_session(session_id)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return {
            "valid": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user(user_id: str) -> Dict[str, Any]:
    """Obtener información de usuario"""
    try:
        user = auth_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "created_at": user.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




