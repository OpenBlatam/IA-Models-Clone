"""
MCP Saga Pattern - Patrón Saga para transacciones distribuidas
===============================================================
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class SagaStepStatus(str, Enum):
    """Estados de paso de saga"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATED = "compensated"
    FAILED = "failed"


class SagaStep(BaseModel):
    """Paso de una saga"""
    step_id: str = Field(..., description="ID único del paso")
    name: str = Field(..., description="Nombre del paso")
    action: Any = Field(..., description="Acción a ejecutar (callable)")
    compensate: Optional[Any] = Field(None, description="Compensación si falla (callable)")
    status: SagaStepStatus = Field(default=SagaStepStatus.PENDING)
    result: Optional[Any] = Field(None, description="Resultado del paso")
    error: Optional[str] = Field(None, description="Error si falló")
    executed_at: Optional[datetime] = None


class Saga(BaseModel):
    """Saga para transacción distribuida"""
    saga_id: str = Field(..., description="ID único de la saga")
    steps: List[SagaStep] = Field(..., description="Pasos de la saga")
    status: str = Field(default="pending", description="Estado de la saga")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class SagaOrchestrator:
    """
    Orquestador de sagas
    
    Ejecuta sagas con compensación automática en caso de fallo.
    """
    
    def __init__(self):
        self._sagas: Dict[str, Saga] = {}
    
    async def execute_saga(self, saga: Saga) -> Dict[str, Any]:
        """
        Ejecuta una saga
        
        Args:
            saga: Saga a ejecutar
            
        Returns:
            Resultado de la saga
        """
        self._sagas[saga.saga_id] = saga
        saga.status = "executing"
        
        executed_steps = []
        
        try:
            for step in saga.steps:
                step.status = SagaStepStatus.EXECUTING
                step.executed_at = datetime.now(timezone.utc)
                
                try:
                    # Ejecutar acción
                    if asyncio.iscoroutinefunction(step.action):
                        result = await step.action()
                    else:
                        result = step.action()
                    
                    step.status = SagaStepStatus.COMPLETED
                    step.result = result
                    executed_steps.append(step)
                    
                    logger.info(f"Saga {saga.saga_id} step {step.step_id} completed")
                    
                except Exception as e:
                    step.status = SagaStepStatus.FAILED
                    step.error = str(e)
                    logger.error(f"Saga {saga.saga_id} step {step.step_id} failed: {e}")
                    
                    # Compensar pasos ejecutados
                    await self._compensate_steps(executed_steps)
                    
                    saga.status = "failed"
                    raise MCPError(f"Saga {saga.saga_id} failed at step {step.step_id}: {e}")
            
            saga.status = "completed"
            saga.completed_at = datetime.now(timezone.utc)
            
            return {
                "saga_id": saga.saga_id,
                "status": "completed",
                "steps": [step.dict() for step in saga.steps],
            }
            
        except Exception as e:
            saga.status = "failed"
            logger.error(f"Saga {saga.saga_id} execution failed: {e}")
            raise
    
    async def _compensate_steps(self, steps: List[SagaStep]):
        """
        Compensa pasos ejecutados
        
        Args:
            steps: Lista de pasos a compensar (en orden inverso)
        """
        # Compensar en orden inverso
        for step in reversed(steps):
            if step.compensate:
                try:
                    if asyncio.iscoroutinefunction(step.compensate):
                        await step.compensate(step.result)
                    else:
                        step.compensate(step.result)
                    
                    step.status = SagaStepStatus.COMPENSATED
                    logger.info(f"Step {step.step_id} compensated")
                    
                except Exception as e:
                    logger.error(f"Error compensating step {step.step_id}: {e}")
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """
        Obtiene una saga
        
        Args:
            saga_id: ID de la saga
            
        Returns:
            Saga o None
        """
        return self._sagas.get(saga_id)

