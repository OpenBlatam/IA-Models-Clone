"""
Rutas para Sistema de Plugins
===============================

Endpoints para gestión de plugins.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.plugin_system import get_plugin_manager, PluginManager, PluginInfo

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/plugins",
    tags=["Plugins"]
)


class LoadPluginRequest(BaseModel):
    """Request para cargar plugin"""
    module_path: str = Field(..., description="Ruta del módulo")
    class_name: str = Field(..., description="Nombre de la clase")
    name: str = Field(..., description="Nombre del plugin")
    version: str = Field(..., description="Versión")
    description: str = Field(..., description="Descripción")
    author: str = Field(..., description="Autor")


@router.get("/")
async def list_plugins(
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Listar todos los plugins"""
    return {"plugins": manager.list_plugins()}


@router.post("/load")
async def load_plugin(
    request: LoadPluginRequest,
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Cargar plugin desde módulo"""
    try:
        info = PluginInfo(
            name=request.name,
            version=request.version,
            description=request.description,
            author=request.author
        )
        
        success = manager.load_plugin_from_module(
            request.module_path,
            request.class_name,
            info
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Error cargando plugin")
        
        return {"status": "loaded", "plugin": request.name}
    except Exception as e:
        logger.error(f"Error cargando plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_name}/execute")
async def execute_plugin(
    plugin_name: str,
    data: Dict[str, Any],
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Ejecutar plugin específico"""
    try:
        result = manager.execute_plugin(plugin_name, data)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/execute")
async def execute_pipeline(
    data: Dict[str, Any],
    plugin_names: Optional[List[str]] = None,
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Ejecutar pipeline de plugins"""
    try:
        result = manager.execute_pipeline(data, plugin_names)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error ejecutando pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_name}/enable")
async def enable_plugin(
    plugin_name: str,
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Habilitar plugin"""
    manager.enable_plugin(plugin_name)
    return {"status": "enabled", "plugin": plugin_name}


@router.post("/{plugin_name}/disable")
async def disable_plugin(
    plugin_name: str,
    manager: PluginManager = Depends(get_plugin_manager)
):
    """Deshabilitar plugin"""
    manager.disable_plugin(plugin_name)
    return {"status": "disabled", "plugin": plugin_name}
















