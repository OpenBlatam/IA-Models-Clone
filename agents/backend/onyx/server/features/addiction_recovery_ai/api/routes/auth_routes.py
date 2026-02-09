"""
Authentication routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.auth_service import AuthService
    from models.database import DatabaseManager
except ImportError:
    from ...services.auth_service import AuthService
    from ...models.database import DatabaseManager

router = APIRouter()

auth = AuthService()
db_manager = DatabaseManager()


@router.post("/auth/register")
async def register(
    user_id: str = Body(...),
    email: Optional[str] = Body(None),
    password: Optional[str] = Body(None),
    name: Optional[str] = Body(None)
):
    """Registra un nuevo usuario"""
    try:
        user = db_manager.create_user(user_id, email, name)
        
        hashed_password = None
        if password:
            hashed_password = auth.hash_password(password)
        
        token_data = {"sub": user_id, "email": email}
        access_token = auth.create_access_token(data=token_data)
        
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "access_token": access_token,
            "token_type": "bearer",
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando usuario: {str(e)}")


@router.post("/auth/login")
async def login(
    user_id: str = Body(...),
    password: Optional[str] = Body(None)
):
    """Inicia sesión y obtiene token"""
    try:
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        token_data = {"sub": user_id, "email": user.email}
        access_token = auth.create_access_token(data=token_data)
        
        return JSONResponse(content={
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user_id,
            "status": "success"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en login: {str(e)}")



