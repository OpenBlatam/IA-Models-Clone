"""
Workflow Engine - Motor de workflows para automatizar procesos complejos
==========================================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Estados de un workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Estados de un step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Step individual en un workflow"""
    id: str
    name: str
    action: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: Optional[float] = None
    condition: Optional[Callable] = None
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """Definición de un workflow"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None


class WorkflowEngine:
    """Motor de workflows para automatizar procesos complejos"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
    def register_workflow(self, workflow: Workflow) -> bool:
        """Registra un nuevo workflow"""
        try:
            # Validar que no haya dependencias circulares
            if self._has_circular_dependencies(workflow.steps):
                logger.error(f"Workflow {workflow.id} tiene dependencias circulares")
                return False
            
            self.workflows[workflow.id] = workflow
            logger.info(f"Workflow {workflow.id} registrado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error registrando workflow {workflow.id}: {e}")
            return False
    
    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Verifica si hay dependencias circulares"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = next((s for s in steps if s.id == step_id), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        for step in steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    return True
        
        return False
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ejecuta un workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} no encontrado")
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        workflow.metadata = context or {}
        
        try:
            # Ejecutar steps en orden de dependencias
            executed_steps = {}
            result = await self._execute_steps(workflow, executed_steps)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            workflow.result = result
            
            self._log_workflow_execution(workflow, True)
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": result,
                "executed_steps": len(executed_steps)
            }
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            
            self._log_workflow_execution(workflow, False)
            logger.error(f"Error ejecutando workflow {workflow_id}: {e}")
            raise
    
    async def _execute_steps(
        self,
        workflow: Workflow,
        executed_steps: Dict[str, Any]
    ) -> Any:
        """Ejecuta los steps del workflow en orden"""
        remaining_steps = {step.id: step for step in workflow.steps}
        result = None
        
        while remaining_steps:
            # Encontrar steps listos para ejecutar (sin dependencias pendientes)
            ready_steps = [
                step for step in remaining_steps.values()
                if all(dep in executed_steps for dep in step.dependencies)
            ]
            
            if not ready_steps:
                # Dependencias circulares o faltantes
                raise ValueError("No hay steps listos para ejecutar")
            
            # Ejecutar steps listos en paralelo
            tasks = [self._execute_step(step, executed_steps) for step in ready_steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for step, step_result in zip(ready_steps, results):
                if isinstance(step_result, Exception):
                    step.status = StepStatus.FAILED
                    step.error = str(step_result)
                    raise step_result
                
                executed_steps[step.id] = step_result
                result = step_result
                remaining_steps.pop(step.id)
        
        return result
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """Ejecuta un step individual"""
        # Verificar condición
        if step.condition and not step.condition(context):
            step.status = StepStatus.SKIPPED
            logger.info(f"Step {step.id} saltado por condición")
            return None
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        # Ejecutar con retry
        last_error = None
        for attempt in range(step.retry_count):
            try:
                if step.timeout:
                    result = await asyncio.wait_for(
                        step.action(context),
                        timeout=step.timeout
                    )
                else:
                    result = await step.action(context)
                
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                step.result = result
                
                # Ejecutar callback de éxito
                if step.on_success:
                    await step.on_success(context, result)
                
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Intento {attempt + 1} fallido para step {step.id}: {e}")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Todos los intentos fallaron
        step.status = StepStatus.FAILED
        step.error = str(last_error)
        step.completed_at = datetime.now()
        
        # Ejecutar callback de fallo
        if step.on_failure:
            await step.on_failure(context, last_error)
        
        raise last_error
    
    def _log_workflow_execution(self, workflow: Workflow, success: bool):
        """Registra la ejecución del workflow"""
        self.workflow_history.append({
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "success": success,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "duration": (
                (workflow.completed_at - workflow.started_at).total_seconds()
                if workflow.started_at and workflow.completed_at
                else None
            ),
            "error": workflow.error
        })
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de un workflow"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status.value,
                    "error": step.error
                }
                for step in workflow.steps
            ],
            "result": workflow.result,
            "error": workflow.error
        }
    
    def get_workflow_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtiene el historial de workflows"""
        history = self.workflow_history
        if workflow_id:
            history = [h for h in history if h["workflow_id"] == workflow_id]
        
        return history[-limit:]
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancela un workflow en ejecución"""
        if workflow_id in self.running_workflows:
            task = self.running_workflows[workflow_id]
            task.cancel()
            del self.running_workflows[workflow_id]
            
            if workflow_id in self.workflows:
                self.workflows[workflow_id].status = WorkflowStatus.CANCELLED
            
            return True
        return False
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """Lista todos los workflows registrados"""
        return [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "steps_count": len(wf.steps),
                "status": wf.status.value
            }
            for wf in self.workflows.values()
        ]




