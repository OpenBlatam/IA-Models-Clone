"""
Idempotency Endpoints
=====================
Endpoints para manejo de idempotencia.
"""

from fastapi import APIRouter, HTTPException, Header, Request
from typing import Dict, Any, Optional
from datetime import datetime

from ...utils.idempotency import (
    generate_idempotency_key,
    check_idempotency,
    store_idempotency_result
)
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/idempotency", tags=["idempotency"])
logger = get_logger(__name__)


@router.post("/check")
async def check_idempotency_endpoint(
    request: Request,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Verificar si una request es idempotente.
    
    Headers:
        Idempotency-Key: Clave de idempotencia (opcional)
    """
    try:
        # Generar clave si no se proporciona
        if not idempotency_key:
            if not data:
                raise HTTPException(status_code=400, detail="Either Idempotency-Key header or request body required")
            
            # Generar desde datos de request
            user_id = request.headers.get("X-User-ID")
            endpoint = request.url.path
            idempotency_key = generate_idempotency_key(data, user_id, endpoint)
        
        # Verificar idempotencia
        cached_result = await check_idempotency(idempotency_key)
        
        if cached_result:
            return {
                "success": True,
                "idempotent": True,
                "cached_result": cached_result,
                "message": "Request is idempotent, returning cached result"
            }
        
        return {
            "success": True,
            "idempotent": False,
            "idempotency_key": idempotency_key,
            "message": "Request is not idempotent, proceed with execution"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking idempotency: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store")
async def store_idempotency_result_endpoint(
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    result: Dict[str, Any] = None,
    ttl: int = 3600
) -> Dict[str, Any]:
    """
    Almacenar resultado de request idempotente.
    
    Headers:
        Idempotency-Key: Clave de idempotencia (requerido)
    """
    try:
        if not result:
            raise HTTPException(status_code=400, detail="Result data required")
        
        await store_idempotency_result(idempotency_key, result, ttl)
        
        return {
            "success": True,
            "message": f"Idempotency result stored for key: {idempotency_key}",
            "ttl": ttl,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing idempotency result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

