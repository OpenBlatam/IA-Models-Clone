"""
Saga Pattern System
===================

Sistema de patrón Saga para transacciones distribuidas.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Estado de saga."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class StepStatus(Enum):
    """Estado de paso."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Paso de saga."""
    step_id: str
    name: str
    execute_func: Callable
    compensate_func: Optional[Callable] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Saga:
    """Saga."""
    saga_id: str
    name: str
    steps: List[SagaStep]
    status: SagaStatus = SagaStatus.PENDING
    current_step: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SagaManager:
    """
    Gestor de sagas.
    
    Gestiona ejecución de sagas con compensación.
    """
    
    def __init__(self):
        """Inicializar gestor de sagas."""
        self.sagas: Dict[str, Saga] = {}
    
    def create_saga(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear saga.
        
        Args:
            name: Nombre de la saga
            steps: Lista de pasos con execute_func y opcional compensate_func
            metadata: Metadata adicional
            
        Returns:
            ID de la saga
        """
        saga_id = str(uuid.uuid4())
        
        saga_steps = []
        for i, step_data in enumerate(steps):
            step_id = f"{saga_id}_step_{i}"
            step = SagaStep(
                step_id=step_id,
                name=step_data.get("name", f"Step {i}"),
                execute_func=step_data["execute_func"],
                compensate_func=step_data.get("compensate_func")
            )
            saga_steps.append(step)
        
        saga = Saga(
            saga_id=saga_id,
            name=name,
            steps=saga_steps,
            metadata=metadata or {}
        )
        
        self.sagas[saga_id] = saga
        logger.info(f"Created saga: {name} ({saga_id})")
        
        return saga_id
    
    async def execute_saga(self, saga_id: str) -> Dict[str, Any]:
        """
        Ejecutar saga.
        
        Args:
            saga_id: ID de la saga
            
        Returns:
            Resultado de la saga
        """
        if saga_id not in self.sagas:
            raise ValueError(f"Saga not found: {saga_id}")
        
        saga = self.sagas[saga_id]
        saga.status = SagaStatus.RUNNING
        saga.started_at = datetime.now().isoformat()
        
        try:
            for i, step in enumerate(saga.steps):
                saga.current_step = i
                step.status = StepStatus.RUNNING
                step.started_at = datetime.now().isoformat()
                
                try:
                    # Ejecutar paso
                    if asyncio.iscoroutinefunction(step.execute_func):
                        result = await step.execute_func()
                    else:
                        result = step.execute_func()
                    
                    step.status = StepStatus.COMPLETED
                    step.result = result
                    step.completed_at = datetime.now().isoformat()
                    
                    logger.info(f"Saga {saga.name} step {step.name} completed")
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.now().isoformat()
                    
                    logger.error(f"Saga {saga.name} step {step.name} failed: {e}", exc_info=True)
                    
                    # Compensar pasos anteriores
                    await self._compensate_saga(saga, i - 1)
                    
                    saga.status = SagaStatus.FAILED
                    saga.completed_at = datetime.now().isoformat()
                    
                    return {
                        "saga_id": saga_id,
                        "status": "failed",
                        "failed_step": step.name,
                        "error": str(e)
                    }
            
            saga.status = SagaStatus.COMPLETED
            saga.completed_at = datetime.now().isoformat()
            
            return {
                "saga_id": saga_id,
                "status": "completed",
                "steps": len(saga.steps)
            }
        except Exception as e:
            saga.status = SagaStatus.FAILED
            saga.completed_at = datetime.now().isoformat()
            logger.error(f"Saga {saga.name} failed: {e}", exc_info=True)
            raise
    
    async def _compensate_saga(self, saga: Saga, last_step_index: int) -> None:
        """
        Compensar saga.
        
        Args:
            saga: Saga a compensar
            last_step_index: Índice del último paso a compensar
        """
        saga.status = SagaStatus.COMPENSATING
        
        # Compensar en orden inverso
        for i in range(last_step_index, -1, -1):
            step = saga.steps[i]
            
            if step.status == StepStatus.COMPLETED and step.compensate_func:
                try:
                    step.status = StepStatus.RUNNING
                    
                    if asyncio.iscoroutinefunction(step.compensate_func):
                        await step.compensate_func(step.result)
                    else:
                        step.compensate_func(step.result)
                    
                    step.status = StepStatus.COMPENSATED
                    logger.info(f"Saga {saga.name} step {step.name} compensated")
                except Exception as e:
                    logger.error(f"Error compensating step {step.name}: {e}", exc_info=True)
        
        saga.status = SagaStatus.COMPENSATED
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Obtener saga por ID."""
        return self.sagas.get(saga_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de sagas."""
        status_counts = {}
        for saga in self.sagas.values():
            status_counts[saga.status.value] = status_counts.get(saga.status.value, 0) + 1
        
        return {
            "total_sagas": len(self.sagas),
            "status_counts": status_counts
        }


# Importar asyncio
import asyncio

# Instancia global
_saga_manager: Optional[SagaManager] = None


def get_saga_manager() -> SagaManager:
    """Obtener instancia global del gestor de sagas."""
    global _saga_manager
    if _saga_manager is None:
        _saga_manager = SagaManager()
    return _saga_manager


