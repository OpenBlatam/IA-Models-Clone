"""
Rutas para Feature Engineering
================================

Endpoints para feature engineering automático.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.feature_engineering import (
    get_feature_engineering,
    AutomatedFeatureEngineering
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/feature-engineering",
    tags=["Feature Engineering"]
)


class GenerateFeaturesRequest(BaseModel):
    """Request para generar características"""
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    target_column: Optional[str] = Field(None, description="Columna objetivo")


@router.post("/generate")
async def generate_features(
    request: GenerateFeaturesRequest,
    system: AutomatedFeatureEngineering = Depends(get_feature_engineering)
):
    """Generar características automáticamente"""
    try:
        features = system.generate_features(request.data, request.target_column)
        
        return {
            "num_features": len(features),
            "features": [
                {
                    "feature_id": f.feature_id,
                    "feature_name": f.feature_name,
                    "feature_type": f.feature_type.value,
                    "importance": f.importance
                }
                for f in features
            ]
        }
    except Exception as e:
        logger.error(f"Error generando características: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select")
async def select_features(
    features: List[Dict[str, Any]] = Field(..., description="Características"),
    method: str = Field("importance", description="Método"),
    top_k: int = Field(10, description="Top K"),
    system: AutomatedFeatureEngineering = Depends(get_feature_engineering)
):
    """Seleccionar características más importantes"""
    try:
        from ..core.feature_engineering import Feature, FeatureType
        
        feature_objects = [
            Feature(
                feature_id=f.get("feature_id", ""),
                feature_name=f.get("feature_name", ""),
                feature_type=FeatureType(f.get("feature_type", "numerical")),
                importance=f.get("importance", 0.5),
                created_at=datetime.now().isoformat()
            )
            for f in features
        ]
        
        selected = system.select_features(feature_objects, method, top_k)
        
        return {
            "selected_count": len(selected),
            "selected_features": [
                {
                    "feature_id": f.feature_id,
                    "feature_name": f.feature_name,
                    "importance": f.importance
                }
                for f in selected
            ]
        }
    except Exception as e:
        logger.error(f"Error seleccionando características: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactions")
async def create_interactions(
    features: List[Dict[str, Any]] = Field(..., description="Características"),
    max_interactions: int = Field(5, description="Máximo de interacciones"),
    system: AutomatedFeatureEngineering = Depends(get_feature_engineering)
):
    """Crear interacciones entre características"""
    try:
        from ..core.feature_engineering import Feature, FeatureType
        
        feature_objects = [
            Feature(
                feature_id=f.get("feature_id", ""),
                feature_name=f.get("feature_name", ""),
                feature_type=FeatureType(f.get("feature_type", "numerical")),
                importance=f.get("importance", 0.5),
                created_at=datetime.now().isoformat()
            )
            for f in features
        ]
        
        interactions = system.create_interactions(feature_objects, max_interactions)
        
        return {
            "num_interactions": len(interactions),
            "interactions": [
                {
                    "feature_id": i.feature_id,
                    "feature_name": i.feature_name,
                    "importance": i.importance
                }
                for i in interactions
            ]
        }
    except Exception as e:
        logger.error(f"Error creando interacciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

