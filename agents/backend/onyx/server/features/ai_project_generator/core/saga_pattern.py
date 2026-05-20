"""
Saga Pattern - Patrón Saga para Transacciones Distribuidas
=========================================================

Implementación del patrón Saga:
- Orchestration-based sagas
- Choreography-based sagas
- Compensation transactions
- Saga state management
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SagaStatus(str, Enum):
    """Estados de saga"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class SagaStep:
    """Paso de una saga"""
    
    def __init__(
        self,
        step_id: str,
        action: Callable,
        compensation: Optional[Callable] = None
    ) -> None:
        self.step_id = step_id
        self.action = action
        self.compensation = compensation
        self.status = SagaStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None


class Saga:
    """
    Saga para transacciones distribuidas.
    """
    
    def __init__(self, saga_id: str) -> None:
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.PENDING
        self.current_step_index = 0
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
    
    def add_step(
        self,
        step_id: str,
        action: Callable,
        compensation: Optional[Callable] = None
    ) -> None:
        """Agrega paso a la saga"""
        step = SagaStep(step_id, action, compensation)
        self.steps.append(step)
    
    async def execute(self) -> bool:
        """Ejecuta la saga"""
        self.status = SagaStatus.RUNNING
        
        try:
            for i, step in enumerate(self.steps):
                self.current_step_index = i
                step.status = SagaStatus.RUNNING
                
                try:
                    if asyncio.iscoroutinefunction(step.action):
                        step.result = await step.action()
                    else:
                        step.result = step.action()
                    
                    step.status = SagaStatus.COMPLETED
                    logger.info(f"Saga {self.saga_id} step {step.step_id} completed")
                    
                except Exception as e:
                    step.status = SagaStatus.FAILED
                    step.error = str(e)
                    logger.error(f"Saga {self.saga_id} step {step.step_id} failed: {e}")
                    
                    # Compensar pasos anteriores
                    await self.compensate()
                    return False
            
            self.status = SagaStatus.COMPLETED
            self.completed_at = datetime.now()
            logger.info(f"Saga {self.saga_id} completed successfully")
            return True
            
        except Exception as e:
            self.status = SagaStatus.FAILED
            logger.error(f"Saga {self.saga_id} failed: {e}")
            await self.compensate()
            return False
    
    async def compensate(self) -> None:
        """Compensa pasos ejecutados"""
        self.status = SagaStatus.COMPENSATING
        
        # Compensar en orden inverso
        for step in reversed(self.steps[:self.current_step_index + 1]):
            if step.status == SagaStatus.COMPLETED and step.compensation:
                try:
                    if asyncio.iscoroutinefunction(step.compensation):
                        await step.compensation(step.result)
                    else:
                        step.compensation(step.result)
                    
                    logger.info(f"Saga {self.saga_id} step {step.step_id} compensated")
                except Exception as e:
                    logger.error(f"Compensation failed for step {step.step_id}: {e}")
        
        self.status = SagaStatus.COMPENSATED


class SagaOrchestrator:
    """
    Orchestrator para sagas.
    """
    
    def __init__(self) -> None:
        self.sagas: Dict[str, Saga] = {}
    
    def create_saga(self) -> Saga:
        """Crea una nueva saga"""
        saga_id = str(uuid.uuid4())
        saga = Saga(saga_id)
        self.sagas[saga_id] = saga
        return saga
    
    async def execute_saga(self, saga: Saga) -> bool:
        """Ejecuta una saga"""
        return await saga.execute()
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Obtiene una saga"""
        return self.sagas.get(saga_id)


import asyncio


def get_saga_orchestrator() -> SagaOrchestrator:
    """Obtiene orchestrator de sagas"""
    return SagaOrchestrator()

