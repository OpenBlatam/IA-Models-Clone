"""
API de Gestión de Modelos

Endpoints para:
- Optimizar modelos
- Versionar modelos
- Comparar modelos
- Listar versiones
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.model_optimizer import get_model_optimizer
from middleware.auth_middleware import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/optimize")
async def optimize_model(
    model_path: str = Body(..., description="Ruta del modelo"),
    optimizations: List[str] = Body(["quantize", "compile"], description="Optimizaciones a aplicar"),
    save_path: Optional[str] = Body(None, description="Ruta para guardar modelo optimizado")
) -> Dict[str, Any]:
    """
    Optimiza un modelo (requiere rol admin).
    """
    try:
        # En producción, esto cargaría el modelo real
        # Por ahora retornamos información
        optimizer = get_model_optimizer()
        
        return {
            "message": "Model optimization started",
            "model_path": model_path,
            "optimizations": optimizations,
            "save_path": save_path or f"{model_path}_optimized"
        }
    except Exception as e:
        logger.error(f"Error optimizing model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error optimizing model: {str(e)}"
        )


@router.post("/versions")
async def save_model_version(
    version: str = Body(..., description="Versión del modelo"),
    model_path: str = Body(..., description="Ruta del modelo"),
    metadata: Dict[str, Any] = Body({}, description="Metadatos adicionales")
) -> Dict[str, Any]:
    """
    Guarda una versión del modelo.
    """
    try:
        optimizer = get_model_optimizer()
        
        # En producción, esto cargaría y guardaría el modelo real
        saved_path = optimizer.save_model_version(
            model=None,  # Se cargaría desde model_path
            version=version,
            metadata=metadata
        )
        
        return {
            "message": "Model version saved",
            "version": version,
            "path": saved_path
        }
    except Exception as e:
        logger.error(f"Error saving model version: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving model version: {str(e)}"
        )


@router.get("/versions")
async def list_model_versions() -> Dict[str, Any]:
    """
    Lista todas las versiones de modelos disponibles.
    """
    try:
        optimizer = get_model_optimizer()
        versions = optimizer.list_versions()
        
        return {
            "versions": versions,
            "total": len(versions)
        }
    except Exception as e:
        logger.error(f"Error listing model versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing versions: {str(e)}"
        )


@router.get("/versions/{version}")
async def get_model_version(version: str) -> Dict[str, Any]:
    """
    Obtiene información de una versión específica.
    """
    try:
        optimizer = get_model_optimizer()
        model, metadata = optimizer.load_model_version(version, model_class=None)
        
        return {
            "version": version,
            "metadata": metadata,
            "loaded": model is not None
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error loading model version: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model version: {str(e)}"
        )

