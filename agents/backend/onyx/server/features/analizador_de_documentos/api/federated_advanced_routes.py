"""
Rutas para Advanced Federated Learning
========================================

Endpoints para federated learning avanzado.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_federated_learning import (
    get_advanced_federated_learning,
    AdvancedFederatedLearning,
    AggregationMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-federated-learning",
    tags=["Advanced Federated Learning"]
)


class RegisterClientRequest(BaseModel):
    """Request para registrar cliente"""
    data_size: int = Field(..., description="Tamaño de datos")
    local_epochs: int = Field(5, description="Épocas locales")


@router.post("/clients/{client_id}/register")
async def register_client(
    client_id: str,
    request: RegisterClientRequest,
    system: AdvancedFederatedLearning = Depends(get_advanced_federated_learning)
):
    """Registrar cliente federado"""
    try:
        client = system.register_client(client_id, request.data_size, request.local_epochs)
        
        return {
            "client_id": client.client_id,
            "data_size": client.data_size,
            "local_epochs": client.local_epochs,
            "status": client.status
        }
    except Exception as e:
        logger.error(f"Error registrando cliente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds")
async def run_federated_round(
    aggregation_method: str = Field("fedavg", description="Método de agregación"),
    selected_clients: Optional[List[str]] = Field(None, description="Clientes seleccionados"),
    system: AdvancedFederatedLearning = Depends(get_advanced_federated_learning)
):
    """Ejecutar ronda de federated learning"""
    try:
        agg_method = AggregationMethod(aggregation_method)
        round_result = system.run_federated_round(agg_method, selected_clients)
        
        return {
            "round_id": round_result.round_id,
            "clients": [
                {"client_id": c.client_id, "data_size": c.data_size}
                for c in round_result.clients
            ],
            "aggregation_method": round_result.aggregation_method.value,
            "global_model_version": round_result.global_model_version,
            "accuracy": round_result.accuracy
        }
    except Exception as e:
        logger.error(f"Error ejecutando ronda: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heterogeneity")
async def analyze_heterogeneity(
    system: AdvancedFederatedLearning = Depends(get_advanced_federated_learning)
):
    """Analizar heterogeneidad de datos"""
    try:
        analysis = system.analyze_heterogeneity()
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


