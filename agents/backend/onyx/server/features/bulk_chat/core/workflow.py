"""
Workflow - Sistema de Workflow/Automation
=========================================

Sistema de workflows y automatización de tareas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Estado de workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Paso de workflow."""
    step_id: str
    name: str
    action: Callable
    condition: Optional[Callable] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Motor de workflows."""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def register_workflow(self, workflow: Workflow):
        """Registrar workflow."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecutar workflow.
        
        Args:
            workflow_id: ID del workflow
            initial_context: Contexto inicial
        
        Returns:
            Resultado de la ejecución
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        workflow.context = initial_context or {}
        workflow.current_step = 0
        
        try:
            for i, step in enumerate(workflow.steps):
                workflow.current_step = i
                
                # Verificar condición
                if step.condition:
                    if not await self._evaluate_condition(step.condition, workflow.context):
                        logger.info(f"Skipping step {step.step_id}: condition not met")
                        continue
                
                # Ejecutar paso
                logger.info(f"Executing step: {step.step_id}")
                
                result = await self._execute_step(step, workflow.context)
                
                # Actualizar contexto
                workflow.context[f"step_{step.step_id}_result"] = result
                
                # Si el resultado es un diccionario, mergear al contexto
                if isinstance(result, dict):
                    workflow.context.update(result)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "context": workflow.context,
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
    
    async def _evaluate_condition(
        self,
        condition: Callable,
        context: Dict[str, Any],
    ) -> bool:
        """Evaluar condición."""
        try:
            if asyncio.iscoroutinefunction(condition):
                return await condition(context)
            else:
                return condition(context)
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
    ) -> Any:
        """Ejecutar paso de workflow."""
        retries = 0
        max_retries = step.retry_count
        
        while retries <= max_retries:
            try:
                if step.timeout:
                    result = await asyncio.wait_for(
                        self._call_action(step.action, context),
                        timeout=step.timeout,
                    )
                else:
                    result = await self._call_action(step.action, context)
                
                return result
                
            except asyncio.TimeoutError:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Step {step.step_id} timed out after {max_retries} retries")
                logger.warning(f"Step {step.step_id} timed out, retrying ({retries}/{max_retries})")
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Step {step.step_id} failed after {max_retries} retries: {e}")
                logger.warning(f"Step {step.step_id} failed, retrying ({retries}/{max_retries}): {e}")
    
    async def _call_action(
        self,
        action: Callable,
        context: Dict[str, Any],
    ) -> Any:
        """Llamar acción."""
        if asyncio.iscoroutinefunction(action):
            return await action(context)
        else:
            return action(context)
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Obtener workflow."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """Listar workflows."""
        return [
            {
                "workflow_id": w.workflow_id,
                "name": w.name,
                "description": w.description,
                "status": w.status.value,
                "steps_count": len(w.steps),
            }
            for w in self.workflows.values()
        ]
































