"""
Rutas para Edge Computing
===========================

Endpoints para edge computing.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.edge_computing import get_edge_computing, EdgeComputingSystem

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/edge",
    tags=["Edge Computing"]
)


class RegisterNodeRequest(BaseModel):
    """Request para registrar nodo edge"""
    node_id: str = Field(..., description="ID del nodo")
    location: str = Field(..., description="Ubicación del nodo")
    capacity: int = Field(100, description="Capacidad del nodo")
    latency_ms: float = Field(10.0, description="Latencia en ms")


class SubmitTaskRequest(BaseModel):
    """Request para enviar tarea"""
    task: Dict[str, Any] = Field(..., description="Tarea a procesar")
    preferred_node: Optional[str] = Field(None, description="Nodo preferido")


@router.post("/nodes")
async def register_node(
    request: RegisterNodeRequest,
    system: EdgeComputingSystem = Depends(get_edge_computing)
):
    """Registrar nodo edge"""
    try:
        node = system.register_node(
            request.node_id,
            request.location,
            request.capacity,
            request.latency_ms
        )
        
        return {
            "status": "registered",
            "node_id": node.node_id,
            "location": node.location,
            "status": node.status.value
        }
    except Exception as e:
        logger.error(f"Error registrando nodo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks")
async def submit_task(
    request: SubmitTaskRequest,
    system: EdgeComputingSystem = Depends(get_edge_computing)
):
    """Enviar tarea a nodo edge"""
    try:
        task_id = system.submit_task(request.task, request.preferred_node)
        
        return {"status": "submitted", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error enviando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes")
async def list_nodes(
    system: EdgeComputingSystem = Depends(get_edge_computing)
):
    """Listar todos los nodos edge"""
    nodes = system.list_nodes()
    return {"nodes": nodes}


@router.get("/nodes/{node_id}/status")
async def get_node_status(
    node_id: str,
    system: EdgeComputingSystem = Depends(get_edge_computing)
):
    """Obtener estado de nodo"""
    status = system.get_node_status(node_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Nodo no encontrado")
    
    return status














