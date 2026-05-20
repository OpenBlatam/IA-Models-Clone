"""
Saga Orchestrator - Orquestador de Sagas
==========================================

Sistema de orquestación de sagas para transacciones distribuidas con compensación.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Estado de saga."""
    PENDING = "pending"
    RUNNING = "running"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Step de saga."""
    step_id: str
    name: str
    execute: Callable
    compensate: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Saga:
    """Saga."""
    saga_id: str
    name: str
    steps: List[SagaStep] = field(default_factory=list)
    status: SagaStatus = SagaStatus.PENDING
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SagaOrchestrator:
    """Orquestador de sagas."""
    
    def __init__(self):
        self.sagas: Dict[str, Saga] = {}
        self.saga_history: deque = deque(maxlen=100000)
        self.active_sagas: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def create_saga(
        self,
        saga_id: str,
        name: str,
        steps: Optional[List[SagaStep]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear saga."""
        saga = Saga(
            saga_id=saga_id,
            name=name,
            steps=steps or [],
            metadata=metadata or {},
        )
        
        async def save_saga():
            async with self._lock:
                self.sagas[saga_id] = saga
        
        asyncio.create_task(save_saga())
        
        logger.info(f"Created saga: {saga_id} - {name}")
        return saga_id
    
    def add_step(
        self,
        saga_id: str,
        step_id: str,
        name: str,
        execute: Callable,
        compensate: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar step a saga."""
        step = SagaStep(
            step_id=step_id,
            name=name,
            execute=execute,
            compensate=compensate,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        async def save_step():
            async with self._lock:
                saga = self.sagas.get(saga_id)
                if not saga:
                    raise ValueError(f"Saga {saga_id} not found")
                saga.steps.append(step)
        
        asyncio.create_task(save_step())
        
        logger.info(f"Added step {step_id} to saga {saga_id}")
        return step_id
    
    async def execute_saga(
        self,
        saga_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ejecutar saga."""
        saga = self.sagas.get(saga_id)
        if not saga:
            raise ValueError(f"Saga {saga_id} not found")
        
        # Iniciar ejecución en background
        task = asyncio.create_task(self._run_saga(saga, context or {}))
        self.active_sagas[saga_id] = task
        
        return saga_id
    
    async def _run_saga(self, saga: Saga, context: Dict[str, Any]):
        """Ejecutar saga."""
        saga.status = SagaStatus.RUNNING
        saga.started_at = datetime.now()
        
        try:
            # Ejecutar steps en orden
            for step in saga.steps:
                # Verificar dependencias
                if step.dependencies:
                    for dep_id in step.dependencies:
                        if dep_id not in saga.completed_steps:
                            raise ValueError(f"Dependency {dep_id} not completed")
                
                saga.current_step = step.step_id
                
                # Ejecutar step
                try:
                    if asyncio.iscoroutinefunction(step.execute):
                        result = await step.execute(context)
                    else:
                        result = step.execute(context)
                    
                    # Actualizar contexto
                    if result:
                        context.update(result)
                    
                    saga.completed_steps.append(step.step_id)
                
                except Exception as e:
                    error_msg = f"Step {step.step_id} failed: {str(e)}"
                    logger.error(error_msg)
                    
                    # Iniciar compensación
                    await self._compensate_saga(saga)
                    return
            
            # Todas las operaciones exitosas
            saga.status = SagaStatus.COMPLETED
            saga.completed_at = datetime.now()
        
        except Exception as e:
            error_msg = f"Saga execution failed: {str(e)}"
            logger.error(error_msg)
            saga.status = SagaStatus.FAILED
            saga.error = error_msg
            
            # Intentar compensación
            await self._compensate_saga(saga)
        
        finally:
            async with self._lock:
                self.saga_history.append(saga)
                if saga.saga_id in self.active_sagas:
                    del self.active_sagas[saga.saga_id]
    
    async def _compensate_saga(self, saga: Saga):
        """Compensar saga."""
        saga.status = SagaStatus.COMPENSATING
        
        # Compensar en orden inverso
        for step in reversed(saga.steps):
            if step.step_id in saga.completed_steps and step.compensate:
                try:
                    if asyncio.iscoroutinefunction(step.compensate):
                        await step.compensate()
                    else:
                        step.compensate()
                    
                    logger.info(f"Compensated step {step.step_id} in saga {saga.saga_id}")
                
                except Exception as e:
                    logger.error(f"Compensation failed for step {step.step_id}: {e}")
        
        saga.status = SagaStatus.COMPENSATED
        saga.completed_at = datetime.now()
    
    def get_saga(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de saga."""
        saga = self.sagas.get(saga_id)
        if not saga:
            # Buscar en historial
            for s in self.saga_history:
                if s.saga_id == saga_id:
                    saga = s
                    break
        
        if not saga:
            return None
        
        return {
            "saga_id": saga.saga_id,
            "name": saga.name,
            "status": saga.status.value,
            "current_step": saga.current_step,
            "completed_steps": saga.completed_steps,
            "total_steps": len(saga.steps),
            "started_at": saga.started_at.isoformat() if saga.started_at else None,
            "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
            "error": saga.error,
        }
    
    def get_saga_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de sagas."""
        history = list(self.saga_history)[-limit:]
        
        return [
            {
                "saga_id": s.saga_id,
                "name": s.name,
                "status": s.status.value,
                "completed_steps": len(s.completed_steps),
                "total_steps": len(s.steps),
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in history
        ]
    
    def get_saga_orchestrator_summary(self) -> Dict[str, Any]:
        """Obtener resumen del orquestador."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for saga in self.sagas.values():
            by_status[saga.status.value] += 1
        
        for saga in self.saga_history:
            by_status[saga.status.value] += 1
        
        return {
            "active_sagas": len(self.active_sagas),
            "total_sagas": len(self.sagas),
            "sagas_by_status": dict(by_status),
            "total_history": len(self.saga_history),
        }


