"""
Rutas para Data Augmentation
==============================

Endpoints para aumento de datos.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.data_augmentation import (
    get_data_augmentation,
    IntelligentDataAugmentation,
    AugmentationType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/data-augmentation",
    tags=["Data Augmentation"]
)


class AugmentDataRequest(BaseModel):
    """Request para aumentar datos"""
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    augmentation_type: str = Field("text", description="Tipo de aumentación")
    factor: float = Field(2.0, description="Factor de aumentación")
    techniques: Optional[List[str]] = Field(None, description="Técnicas")


@router.post("/augment")
async def augment_data(
    request: AugmentDataRequest,
    system: IntelligentDataAugmentation = Depends(get_data_augmentation)
):
    """Aumentar datos"""
    try:
        aug_type = AugmentationType(request.augmentation_type)
        result = system.augment_data(
            request.data,
            aug_type,
            request.factor,
            request.techniques
        )
        
        return {
            "augmentation_id": result.augmentation_id,
            "original_samples": result.original_samples,
            "augmented_samples": result.augmented_samples,
            "augmentation_type": result.augmentation_type.value,
            "techniques_used": result.techniques_used
        }
    except Exception as e:
        logger.error(f"Error aumentando datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augment-text")
async def augment_text(
    text_data: List[str] = Field(..., description="Textos"),
    techniques: Optional[List[str]] = Field(None, description="Técnicas"),
    system: IntelligentDataAugmentation = Depends(get_data_augmentation)
):
    """Aumentar datos de texto"""
    try:
        augmented = system.augment_text(text_data, techniques)
        
        return {
            "original_count": len(text_data),
            "augmented_count": len(augmented),
            "augmented_texts": augmented
        }
    except Exception as e:
        logger.error(f"Error aumentando texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_augmented(
    original_data: List[Dict[str, Any]] = Field(..., description="Datos originales"),
    augmented_data: List[Dict[str, Any]] = Field(..., description="Datos aumentados"),
    system: IntelligentDataAugmentation = Depends(get_data_augmentation)
):
    """Validar datos aumentados"""
    try:
        validation = system.validate_augmented_data(original_data, augmented_data)
        
        return validation
    except Exception as e:
        logger.error(f"Error validando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


