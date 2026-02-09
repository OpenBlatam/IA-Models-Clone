"""
LLM Models Routes - Rutas para gestión de modelos.

Incluye:
- Listado de modelos disponibles
- Información detallada de modelos
- Validación de modelos
- Recomendaciones de modelos
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from config.di_setup import get_service
from config.logging_config import get_logger
from core.services.llm_service import LLMService
from core.services.llm import (
    get_model_validator,
    ModelCapability,
    ModelInfo
)
from core.constants import ErrorMessages

logger = get_logger(__name__)

router = APIRouter(
    prefix="/llm/models",
    tags=["LLM Models"],
    responses={
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable"}
    }
)


def get_llm_service() -> Optional[LLMService]:
    """Obtener servicio LLM del DI container."""
    try:
        return get_service("llm_service")
    except (ValueError, Exception):
        return None


class ModelValidationRequest(BaseModel):
    """Request para validar un modelo."""
    model_id: str = Field(..., description="ID del modelo a validar")
    required_capability: Optional[str] = Field(
        None,
        description="Capacidad requerida (opcional)"
    )
    require_streaming: bool = Field(
        False,
        description="Si requiere soporte de streaming"
    )


class ModelValidationResponse(BaseModel):
    """Respuesta de validación de modelo."""
    is_valid: bool
    model_id: str
    error_message: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None


class ModelRecommendationRequest(BaseModel):
    """Request para obtener recomendación de modelo."""
    capability: str = Field(..., description="Capacidad requerida")
    prefer_streaming: bool = Field(False, description="Preferir streaming")
    max_cost: Optional[float] = Field(None, description="Costo máximo por 1k tokens")


@router.get("/available")
@handle_api_errors
async def get_available_models(
    llm_service: Optional[LLMService] = Depends(get_llm_service),
    refresh: bool = Query(False, description="Forzar actualización de lista")
):
    """
    Obtener lista de modelos disponibles.
    
    Args:
        refresh: Si forzar actualización desde OpenRouter
        
    Returns:
        Lista de modelos disponibles con información
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    validator = get_model_validator()
    
    # Refrescar si es necesario
    if refresh or not validator.available_models:
        try:
            await validator.fetch_available_models(llm_service)
        except Exception as e:
            logger.error(f"Error al refrescar modelos: {e}")
            if not validator.available_models:
                raise HTTPException(
                    status_code=503,
                    detail="No se pudo obtener lista de modelos"
                )
    
    # Obtener información de modelos
    models_info = []
    for model_id in sorted(validator.available_models):
        model_info = validator.get_model_info(model_id)
        if model_info:
            models_info.append(model_info.to_dict())
    
    return {
        "success": True,
        "models": models_info,
        "total": len(models_info),
        "last_updated": validator.last_fetch.isoformat() if validator.last_fetch else None
    }


@router.get("/{model_id}")
@handle_api_errors
async def get_model_info(
    model_id: str,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener información detallada de un modelo.
    
    Args:
        model_id: ID del modelo
        
    Returns:
        Información detallada del modelo
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    validator = get_model_validator()
    
    # Asegurar que tenemos la lista actualizada
    if not validator.available_models:
        await validator.fetch_available_models(llm_service)
    
    model_info = validator.get_model_info(model_id)
    
    if not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_id}' no encontrado"
        )
    
    return {
        "success": True,
        "model": model_info.to_dict()
    }


@router.post("/validate")
@handle_api_errors
async def validate_model(
    request: ModelValidationRequest,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Validar un modelo y sus capacidades.
    
    Args:
        request: Request de validación
        
    Returns:
        Resultado de la validación
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    validator = get_model_validator()
    
    # Asegurar que tenemos la lista actualizada
    if not validator.available_models:
        await validator.fetch_available_models(llm_service)
    
    # Convertir capacidad si se proporciona
    capability = None
    if request.required_capability:
        try:
            capability = ModelCapability(request.required_capability)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Capacidad inválida: {request.required_capability}"
            )
    
    # Validar modelo
    is_valid, error_message = validator.validate_model(
        model_id=request.model_id,
        required_capability=capability,
        require_streaming=request.require_streaming
    )
    
    model_info = None
    if is_valid:
        model_info = validator.get_model_info(request.model_id)
        if model_info:
            model_info = model_info.to_dict()
    
    return ModelValidationResponse(
        is_valid=is_valid,
        model_id=request.model_id,
        error_message=error_message,
        model_info=model_info
    )


@router.post("/recommend")
@handle_api_errors
async def recommend_model(
    request: ModelRecommendationRequest,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener recomendación de modelo para una capacidad específica.
    
    Args:
        request: Request de recomendación
        
    Returns:
        Modelo recomendado con información
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    validator = get_model_validator()
    
    # Asegurar que tenemos la lista actualizada
    if not validator.available_models:
        await validator.fetch_available_models(llm_service)
    
    # Convertir capacidad
    try:
        capability = ModelCapability(request.capability)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Capacidad inválida: {request.capability}"
        )
    
    # Obtener recomendación
    recommended_model_id = validator.get_recommended_model(
        capability=capability,
        prefer_streaming=request.prefer_streaming,
        max_cost=request.max_cost
    )
    
    if not recommended_model_id:
        return {
            "success": False,
            "message": "No se encontró un modelo adecuado con los criterios especificados"
        }
    
    model_info = validator.get_model_info(recommended_model_id)
    
    return {
        "success": True,
        "recommended_model": recommended_model_id,
        "model_info": model_info.to_dict() if model_info else None,
        "reason": "Modelo recomendado basado en capacidad, costo y disponibilidad"
    }


@router.get("/capabilities/list")
@handle_api_errors
async def list_capabilities():
    """
    Listar todas las capacidades disponibles.
    
    Returns:
        Lista de capacidades con descripciones
    """
    capabilities = {
        "text_generation": "Generación de texto general",
        "code_generation": "Generación de código",
        "code_analysis": "Análisis de código",
        "streaming": "Streaming de respuestas",
        "function_calling": "Llamadas a funciones",
        "embeddings": "Generación de embeddings",
        "vision": "Procesamiento de imágenes",
        "multimodal": "Soporte multimodal (texto + imágenes)"
    }
    
    return {
        "success": True,
        "capabilities": [
            {
                "id": cap.value,
                "name": capabilities.get(cap.value, cap.value),
                "description": capabilities.get(cap.value, "")
            }
            for cap in ModelCapability
        ]
    }


@router.post("/refresh")
@handle_api_errors
async def refresh_models(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Refrescar lista de modelos disponibles desde OpenRouter.
    
    Returns:
        Resultado de la actualización
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    validator = get_model_validator()
    
    try:
        models = await validator.fetch_available_models(llm_service)
        return {
            "success": True,
            "message": "Lista de modelos actualizada",
            "total_models": len(models),
            "last_updated": validator.last_fetch.isoformat() if validator.last_fetch else None
        }
    except Exception as e:
        logger.error(f"Error al refrescar modelos: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al refrescar modelos: {str(e)}"
        )



