"""
Rutas para Reinforcement Learning
===================================

Endpoints para reinforcement learning.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.reinforcement_learning import (
    get_rl_agent,
    ReinforcementLearningAgent,
    State,
    ActionType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/reinforcement-learning",
    tags=["Reinforcement Learning"]
)


class SelectActionRequest(BaseModel):
    """Request para seleccionar acción"""
    features: Dict[str, float] = Field(..., description="Características del estado")
    available_actions: List[str] = Field(..., description="Acciones disponibles")


class UpdateQValueRequest(BaseModel):
    """Request para actualizar Q-value"""
    state_features: Dict[str, float] = Field(..., description="Estado actual")
    action: str = Field(..., description="Acción tomada")
    reward: float = Field(..., description="Recompensa")
    next_state_features: Optional[Dict[str, float]] = Field(None, description="Estado siguiente")


class TrainEpisodeRequest(BaseModel):
    """Request para entrenar episodio"""
    states: List[Dict[str, float]] = Field(..., description="Estados")
    actions: List[str] = Field(..., description="Acciones")
    rewards: List[float] = Field(..., description="Recompensas")


@router.post("/select-action")
async def select_action(
    request: SelectActionRequest,
    agent: ReinforcementLearningAgent = Depends(get_rl_agent)
):
    """Seleccionar acción usando política"""
    try:
        state = State(
            state_id=f"state_{len(agent.q_table)}",
            features=request.features,
            timestamp=""
        )
        
        available_actions = [ActionType(a) for a in request.available_actions]
        action = agent.select_action(state, available_actions)
        
        return {
            "action": action.value,
            "epsilon": agent.epsilon
        }
    except Exception as e:
        logger.error(f"Error seleccionando acción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-q-value")
async def update_q_value(
    request: UpdateQValueRequest,
    agent: ReinforcementLearningAgent = Depends(get_rl_agent)
):
    """Actualizar Q-value"""
    try:
        state = State(
            state_id="temp_state",
            features=request.state_features,
            timestamp=""
        )
        
        action = ActionType(request.action)
        
        next_state = None
        if request.next_state_features:
            next_state = State(
                state_id="temp_next_state",
                features=request.next_state_features,
                timestamp=""
            )
        
        agent.update_q_value(state, action, request.reward, next_state)
        
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Error actualizando Q-value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-episode")
async def train_episode(
    request: TrainEpisodeRequest,
    agent: ReinforcementLearningAgent = Depends(get_rl_agent)
):
    """Entrenar con episodio"""
    try:
        states = [
            State(
                state_id=f"state_{i}",
                features=state_features,
                timestamp=""
            )
            for i, state_features in enumerate(request.states)
        ]
        
        actions = [ActionType(a) for a in request.actions]
        
        agent.train_episode(states, actions, request.rewards)
        
        return {"status": "trained"}
    except Exception as e:
        logger.error(f"Error entrenando episodio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy")
async def get_policy(
    agent: ReinforcementLearningAgent = Depends(get_rl_agent)
):
    """Obtener política aprendida"""
    policy = agent.get_policy()
    
    return {
        "policy": policy,
        "epsilon": agent.epsilon,
        "num_states": len(agent.q_table)
    }

