"""
Rutas para Aprendizaje Federado
=================================

Endpoints para aprendizaje federado.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.federated_learning import (
    get_federated_learning,
    FederatedLearningSystem,
    FederatedRoundStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/federated",
    tags=["Federated Learning"]
)


class RegisterClientRequest(BaseModel):
    """Request para registrar cliente"""
    client_id: str = Field(..., description="ID del cliente")
    url: str = Field(..., description="URL del cliente")


class SubmitUpdateRequest(BaseModel):
    """Request para enviar actualización"""
    model_update: Dict[str, Any] = Field(..., description="Actualización del modelo")


@router.post("/clients")
async def register_client(
    request: RegisterClientRequest,
    system: FederatedLearningSystem = Depends(get_federated_learning)
):
    """Registrar cliente federado"""
    try:
        client = system.register_client(request.client_id, request.url)
        
        return {
            "status": "registered",
            "client_id": client.client_id,
            "url": client.url
        }
    except Exception as e:
        logger.error(f"Error registrando cliente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds")
async def start_round(
    round_id: Optional[str] = None,
    client_ids: Optional[List[str]] = None,
    system: FederatedLearningSystem = Depends(get_federated_learning)
):
    """Iniciar ronda de aprendizaje federado"""
    try:
        round_obj = system.start_round(round_id, client_ids)
        
        return {
            "status": "started",
            "round_id": round_obj.round_id,
            "status": round_obj.status.value,
            "clients": round_obj.clients
        }
    except Exception as e:
        logger.error(f"Error iniciando ronda: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_id}/updates/{client_id}")
async def submit_model_update(
    round_id: str,
    client_id: str,
    request: SubmitUpdateRequest,
    system: FederatedLearningSystem = Depends(get_federated_learning)
):
    """Enviar actualización de modelo"""
    try:
        success = system.submit_model_update(
            round_id,
            client_id,
            request.model_update
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Error enviando actualización")
        
        return {"status": "submitted"}
    except Exception as e:
        logger.error(f"Error enviando actualización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rounds/{round_id}/aggregate")
async def aggregate_models(
    round_id: str,
    system: FederatedLearningSystem = Depends(get_federated_learning)
):
    """Agregar modelos de clientes"""
    try:
        aggregated = system.aggregate_models(round_id)
        
        return {
            "status": "aggregated",
            "round_id": round_id,
            "aggregated_model": aggregated
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error agregando modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_id}/status")
async def get_round_status(
    round_id: str,
    system: FederatedLearningSystem = Depends(get_federated_learning)
):
    """Obtener estado de ronda"""
    status = system.get_round_status(round_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Ronda no encontrada")
    
    return status














