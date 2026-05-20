"""
Rutas para Caché Distribuido
==============================

Endpoints para gestión de caché distribuido.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.distributed_cache import get_distributed_cache, DistributedCache

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/cache-distributed",
    tags=["Distributed Cache"]
)


class RegisterNodeRequest(BaseModel):
    """Request para registrar nodo"""
    node_id: str = Field(..., description="ID del nodo")
    url: str = Field(..., description="URL del nodo")
    capacity: int = Field(1000, description="Capacidad del nodo")


class SetCacheRequest(BaseModel):
    """Request para establecer valor en caché"""
    key: str = Field(..., description="Clave")
    value: Any = Field(..., description="Valor")
    ttl: Optional[int] = Field(None, description="Time to live en segundos")


@router.post("/nodes")
async def register_node(
    request: RegisterNodeRequest,
    cache: DistributedCache = Depends(get_distributed_cache)
):
    """Registrar nodo de caché"""
    try:
        cache.register_node(request.node_id, request.url, request.capacity)
        
        return {"status": "registered", "node_id": request.node_id}
    except Exception as e:
        logger.error(f"Error registrando nodo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set")
async def set_cache_value(
    request: SetCacheRequest,
    cache: DistributedCache = Depends(get_distributed_cache)
):
    """Establecer valor en caché distribuido"""
    try:
        success = cache.set(request.key, request.value, request.ttl)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error guardando en caché")
        
        return {"status": "set", "key": request.key}
    except Exception as e:
        logger.error(f"Error guardando en caché: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{key}")
async def get_cache_value(
    key: str,
    cache: DistributedCache = Depends(get_distributed_cache)
):
    """Obtener valor del caché distribuido"""
    value = cache.get(key)
    
    if value is None:
        raise HTTPException(status_code=404, detail="Clave no encontrada")
    
    return {"key": key, "value": value}


@router.get("/stats")
async def get_cache_stats(
    cache: DistributedCache = Depends(get_distributed_cache)
):
    """Obtener estadísticas del caché"""
    stats = cache.get_cache_stats()
    return stats














