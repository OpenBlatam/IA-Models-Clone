"""
Rutas para Agentes de IA
=========================

Endpoints para agentes de IA autónomos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.ai_agent import get_multi_agent_system, MultiAgentSystem

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/agents",
    tags=["AI Agents"]
)


class RegisterAgentRequest(BaseModel):
    """Request para registrar agente"""
    agent_id: str = Field(..., description="ID del agente")
    capabilities: List[str] = Field(..., description="Capacidades del agente")


class AssignTaskRequest(BaseModel):
    """Request para asignar tarea"""
    task_description: str = Field(..., description="Descripción de la tarea")
    required_capabilities: List[str] = Field(..., description="Capacidades requeridas")


@router.post("/register")
async def register_agent(
    request: RegisterAgentRequest,
    system: MultiAgentSystem = Depends(get_multi_agent_system)
):
    """Registrar agente de IA"""
    try:
        agent = system.register_agent(request.agent_id, request.capabilities)
        
        return {
            "status": "registered",
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities
        }
    except Exception as e:
        logger.error(f"Error registrando agente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks")
async def assign_task(
    request: AssignTaskRequest,
    system: MultiAgentSystem = Depends(get_multi_agent_system)
):
    """Asignar tarea a agente"""
    try:
        task_id = system.assign_task(
            request.task_description,
            request.required_capabilities
        )
        
        if task_id is None:
            raise HTTPException(
                status_code=503,
                detail="No hay agentes disponibles con las capacidades requeridas"
            )
        
        return {"status": "assigned", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error asignando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents(
    system: MultiAgentSystem = Depends(get_multi_agent_system)
):
    """Listar todos los agentes"""
    agents = [
        {
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities,
            "status": agent.status.value,
            "tasks_count": len(agent.tasks)
        }
        for agent in system.agents.values()
    ]
    
    return {"agents": agents}














