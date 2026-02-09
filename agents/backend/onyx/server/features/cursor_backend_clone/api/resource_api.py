"""
Resource API - Endpoints para gestión de recursos
==================================================

Endpoints de API para monitorear y gestionar recursos del sistema.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from ..core.resource_manager import ResourceManager

router = APIRouter(prefix="/api/resources", tags=["Resources"])


@router.get("/memory")
async def get_memory_info(agent) -> Dict[str, Any]:
    """
    Obtener información de memoria del sistema.
    
    Returns:
        Información detallada de uso de memoria
    """
    try:
        if not agent.resource_manager:
            return {"message": "Resource manager not available"}
        
        return agent.resource_manager.get_memory_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cpu")
async def get_cpu_info(agent) -> Dict[str, Any]:
    """
    Obtener información de CPU.
    
    Returns:
        Información detallada de uso de CPU
    """
    try:
        if not agent.resource_manager:
            return {"message": "Resource manager not available"}
        
        return agent.resource_manager.get_cpu_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_info(agent) -> Dict[str, Any]:
    """
    Obtener información completa del sistema.
    
    Returns:
        Información completa de memoria, CPU y recursos
    """
    try:
        if not agent.resource_manager:
            return {"message": "Resource manager not available"}
        
        return agent.resource_manager.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def trigger_cleanup(agent) -> Dict[str, Any]:
    """
    Forzar limpieza de recursos.
    
    Returns:
        Resultado de la limpieza
    """
    try:
        if not agent.resource_manager:
            return {"message": "Resource manager not available", "success": False}
        
        await agent.resource_manager.cleanup()
        return {"success": True, "message": "Cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

