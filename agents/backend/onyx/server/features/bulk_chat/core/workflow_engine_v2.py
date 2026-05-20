"""
Workflow Engine V2 - Motor de Flujos de Trabajo Avanzado
=========================================================

Sistema avanzado de flujos de trabajo con soporte para bucles, paralelismo y compensación.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Tipo de step."""
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    DELAY = "delay"
    COMPENSATION = "compensation"


class StepStatus(Enum):
    """Estado de step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Estado de workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Step de workflow."""
    step_id: str
    step_type: StepType
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: Optional[float] = None
    compensation_handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Ejecución de workflow."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngineV2:
    """Motor de flujos de trabajo avanzado."""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: deque = deque(maxlen=100000)
        self.active_executions: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str = "",
        steps: Optional[List[WorkflowStep]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear workflow."""
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=steps or [],
            metadata=metadata or {},
        )
        
        async def save_workflow():
            async with self._lock:
                self.workflows[workflow_id] = workflow
        
        asyncio.create_task(save_workflow())
        
        logger.info(f"Created workflow: {workflow_id} - {name}")
        return workflow_id
    
    def add_step(
        self,
        workflow_id: str,
        step_id: str,
        step_type: StepType,
        name: str,
        handler: Callable,
        dependencies: Optional[List[str]] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None,
        compensation_handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar step a workflow."""
        step = WorkflowStep(
            step_id=step_id,
            step_type=step_type,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            retry_count=retry_count,
            timeout=timeout,
            compensation_handler=compensation_handler,
            metadata=metadata or {},
        )
        
        async def save_step():
            async with self._lock:
                workflow = self.workflows.get(workflow_id)
                if not workflow:
                    raise ValueError(f"Workflow {workflow_id} not found")
                workflow.steps.append(step)
        
        asyncio.create_task(save_step())
        
        logger.info(f"Added step {step_id} to workflow {workflow_id}")
        return step_id
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ejecutar workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        exec_id = execution_id or f"exec_{workflow_id}_{datetime.now().timestamp()}"
        
        execution = WorkflowExecution(
            execution_id=exec_id,
            workflow_id=workflow_id,
            context=context or {},
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.executions[exec_id] = execution
        
        # Iniciar ejecución en background
        task = asyncio.create_task(self._run_workflow(execution, workflow))
        self.active_executions[exec_id] = task
        
        return exec_id
    
    async def _run_workflow(self, execution: WorkflowExecution, workflow: Workflow):
        """Ejecutar workflow."""
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Ejecutar steps en orden, respetando dependencias
            steps_to_execute = workflow.steps.copy()
            executed_steps = set()
            
            while steps_to_execute:
                # Buscar steps sin dependencias pendientes
                ready_steps = [
                    s for s in steps_to_execute
                    if all(dep in executed_steps for dep in s.dependencies)
                ]
                
                if not ready_steps:
                    # Deadlock o dependencias circulares
                    raise ValueError("Circular dependency or missing dependencies")
                
                # Ejecutar steps listos (pueden ser en paralelo si son PARALLEL)
                parallel_groups = defaultdict(list)
                for step in ready_steps:
                    if step.step_type == StepType.PARALLEL:
                        parallel_groups[step.step_id].append(step)
                    else:
                        parallel_groups[step.step_id] = [step]
                
                for group_steps in parallel_groups.values():
                    if len(group_steps) > 1:
                        # Ejecutar en paralelo
                        await asyncio.gather(*[
                            self._execute_step(execution, step)
                            for step in group_steps
                        ], return_exceptions=True)
                    else:
                        # Ejecutar secuencialmente
                        await self._execute_step(execution, group_steps[0])
                
                # Marcar como ejecutados
                for step in ready_steps:
                    executed_steps.add(step.step_id)
                    steps_to_execute.remove(step)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            execution.status = WorkflowStatus.FAILED
            execution.error = error_msg
            execution.completed_at = datetime.now()
            
            # Ejecutar compensaciones
            await self._execute_compensations(execution, workflow)
        
        finally:
            async with self._lock:
                self.execution_history.append(execution)
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Ejecutar step."""
        execution.current_step = step.step_id
        
        try:
            if step.step_type == StepType.DELAY:
                delay_seconds = step.metadata.get("delay_seconds", 0)
                await asyncio.sleep(delay_seconds)
            
            elif step.step_type == StepType.CONDITION:
                # Evaluar condición
                condition_result = await self._evaluate_condition(step, execution.context)
                if not condition_result:
                    return
            
            elif step.step_type == StepType.LOOP:
                # Ejecutar loop
                await self._execute_loop(step, execution)
            
            else:
                # Ejecutar handler con retry
                for attempt in range(step.retry_count + 1):
                    try:
                        if step.timeout:
                            result = await asyncio.wait_for(
                                self._call_handler(step.handler, execution.context),
                                timeout=step.timeout
                            )
                        else:
                            result = await self._call_handler(step.handler, execution.context)
                        
                        # Actualizar contexto
                        if result:
                            execution.context.update(result)
                        
                        execution.steps_completed.append(step.step_id)
                        return
                    
                    except Exception as e:
                        if attempt == step.retry_count:
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        except Exception as e:
            execution.steps_failed.append(step.step_id)
            raise
    
    async def _call_handler(self, handler: Callable, context: Dict[str, Any]) -> Any:
        """Llamar handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(context)
        else:
            return handler(context)
    
    async def _evaluate_condition(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Evaluar condición."""
        condition = step.metadata.get("condition")
        if not condition:
            return True
        
        if asyncio.iscoroutinefunction(condition):
            return await condition(context)
        else:
            return condition(context)
    
    async def _execute_loop(self, step: WorkflowStep, execution: WorkflowExecution):
        """Ejecutar loop."""
        loop_count = step.metadata.get("loop_count", 1)
        
        for i in range(loop_count):
            await self._call_handler(step.handler, {
                **execution.context,
                "loop_index": i,
                "loop_count": loop_count,
            })
    
    async def _execute_compensations(self, execution: WorkflowExecution, workflow: Workflow):
        """Ejecutar compensaciones."""
        # Ejecutar compensaciones en orden inverso
        for step in reversed(workflow.steps):
            if step.step_id in execution.steps_completed and step.compensation_handler:
                try:
                    await self._call_handler(step.compensation_handler, execution.context)
                except Exception as e:
                    logger.error(f"Compensation failed for step {step.step_id}: {e}")
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancelar ejecución."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        if execution.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        # Cancelar task si está activo
        if execution_id in self.active_executions:
            self.active_executions[execution_id].cancel()
            del self.active_executions[execution_id]
        
        logger.info(f"Cancelled workflow execution: {execution_id}")
        return True
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_type": s.step_type.value,
                    "name": s.name,
                    "dependencies": s.dependencies,
                }
                for s in workflow.steps
            ],
            "created_at": workflow.created_at.isoformat(),
        }
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de ejecución."""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "current_step": execution.current_step,
            "steps_completed": execution.steps_completed,
            "steps_failed": execution.steps_failed,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error": execution.error,
        }
    
    def get_workflow_engine_v2_summary(self) -> Dict[str, Any]:
        """Obtener resumen del motor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for execution in self.executions.values():
            by_status[execution.status.value] += 1
        
        return {
            "total_workflows": len(self.workflows),
            "total_executions": len(self.executions),
            "active_executions": len(self.active_executions),
            "executions_by_status": dict(by_status),
            "total_history": len(self.execution_history),
        }


