"""
Rutas para Multi-Agent RL
===========================

Endpoints para multi-agent reinforcement learning.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.multi_agent_rl import (
    get_multi_agent_rl,
    MultiAgentRL,
    MARLAlgorithm
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/multi-agent-rl",
    tags=["Multi-Agent RL"]
)


class CreateEnvironmentRequest(BaseModel):
    """Request para crear ambiente"""
    num_agents: int = Field(..., description="Número de agentes")
    state_space: Dict[str, Any] = Field(..., description="Espacio de estados")
    action_space: Dict[str, Any] = Field(..., description="Espacio de acciones")


@router.post("/environments")
async def create_environment(
    request: CreateEnvironmentRequest,
    system: MultiAgentRL = Depends(get_multi_agent_rl)
):
    """Crear ambiente multi-agente"""
    try:
        environment = system.create_environment(
            request.num_agents,
            request.state_space,
            request.action_space
        )
        
        return {
            "env_id": environment.env_id,
            "num_agents": len(environment.agents),
            "state_space": environment.state_space,
            "action_space": environment.action_space
        }
    except Exception as e:
        logger.error(f"Error creando ambiente: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/environments/{env_id}/train")
async def train_marl(
    env_id: str,
    algorithm: str = Field("qmix", description="Algoritmo"),
    episodes: int = Field(100, description="Número de episodios"),
    system: MultiAgentRL = Depends(get_multi_agent_rl)
):
    """Entrenar agentes multi-agente"""
    try:
        marl_algorithm = MARLAlgorithm(algorithm)
        result = system.train_marl(env_id, marl_algorithm, episodes)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/environments/{env_id}/cooperation")
async def evaluate_cooperation(
    env_id: str,
    system: MultiAgentRL = Depends(get_multi_agent_rl)
):
    """Evaluar cooperación entre agentes"""
    try:
        cooperation = system.evaluate_cooperation(env_id)
        
        return cooperation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando cooperación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


