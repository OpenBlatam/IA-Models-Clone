"""
User management routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime

try:
    from models.database import DatabaseManager
except ImportError:
    from ...models.database import DatabaseManager

router = APIRouter()

db_manager = DatabaseManager()


@router.post("/users/create")
async def create_user(
    user_id: str = Body(...),
    email: Optional[str] = Body(None),
    name: Optional[str] = Body(None)
):
    """Crea un nuevo usuario"""
    try:
        user = db_manager.create_user(user_id, email, name)
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando usuario: {str(e)}")


@router.get("/users/{user_id}")
async def get_user(user_id: str):
    """Obtiene información del usuario"""
    try:
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        return JSONResponse(content={
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "status": "success"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo usuario: {str(e)}")


@router.get("/export/{user_id}")
async def export_user_data(
    user_id: str,
    format: str = Query("json", regex="^(json|csv)$")
):
    """Exporta datos del usuario en formato JSON o CSV"""
    try:
        export_data = {
            "user_id": user_id,
            "format": format,
            "exported_at": datetime.now().isoformat(),
            "data": {
                "message": "Datos de exportación (implementar con datos reales de BD)"
            }
        }
        
        if format == "csv":
            return JSONResponse(content={
                "message": "Exportación CSV (implementar conversión)",
                "data": export_data
            })
        
        return JSONResponse(content=export_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando datos: {str(e)}")



