"""
Workflow Engine - Motor de flujos de trabajo para agentes
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json

from .agent_manager import AgentManager
from .event_system import EventSystem

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Estados de un flujo de trabajo."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Estados de un paso del flujo."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Paso de un flujo de trabajo."""
    step_id: str
    name: str
    description: str
    agent_type: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutos
    retry_attempts: int = 3
    condition: Optional[str] = None  # Condición para ejecutar el paso
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Workflow:
    """Flujo de trabajo."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "system"
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowEngine:
    """
    Motor de flujos de trabajo para agentes.
    """
    
    def __init__(self, agent_manager: AgentManager):
        """Inicializar el motor de flujos de trabajo."""
        self.agent_manager = agent_manager
        self.event_system = EventSystem()
        
        # Almacenamiento de flujos
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}
        
        # Configuración
        self.max_concurrent_workflows = 10
        self.workflow_timeout = 3600  # 1 hora
        self.step_timeout = 300  # 5 minutos
        
        # Métricas
        self.metrics = {
            "total_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "active_workflows": 0,
            "average_execution_time": 0.0
        }
        
        logger.info("WorkflowEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de flujos de trabajo."""
        try:
            await self.event_system.initialize()
            logger.info("WorkflowEngine inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar WorkflowEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de flujos de trabajo."""
        try:
            # Cancelar flujos activos
            for workflow_id, task in self.active_workflows.items():
                task.cancel()
            
            await self.event_system.shutdown()
            logger.info("WorkflowEngine cerrado")
        except Exception as e:
            logger.error(f"Error al cerrar WorkflowEngine: {e}")
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        created_by: str = "system",
        parameters: Dict[str, Any] = None
    ) -> str:
        """
        Crear un nuevo flujo de trabajo.
        
        Args:
            name: Nombre del flujo
            description: Descripción del flujo
            steps: Lista de pasos del flujo
            created_by: Usuario que crea el flujo
            parameters: Parámetros globales del flujo
            
        Returns:
            ID del flujo de trabajo creado
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Crear pasos del flujo
            workflow_steps = []
            for step_data in steps:
                step = WorkflowStep(
                    step_id=str(uuid.uuid4()),
                    name=step_data["name"],
                    description=step_data.get("description", ""),
                    agent_type=step_data["agent_type"],
                    task_type=step_data["task_type"],
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", []),
                    timeout=step_data.get("timeout", self.step_timeout),
                    retry_attempts=step_data.get("retry_attempts", 3),
                    condition=step_data.get("condition")
                )
                workflow_steps.append(step)
            
            # Crear flujo de trabajo
            workflow = Workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                steps=workflow_steps,
                created_by=created_by,
                parameters=parameters or {}
            )
            
            # Almacenar flujo
            self.workflows[workflow_id] = workflow
            self.metrics["total_workflows"] += 1
            
            # Emitir evento
            await self.event_system.emit_event("workflow_created", {
                "workflow_id": workflow_id,
                "name": name,
                "created_by": created_by
            })
            
            logger.info(f"Flujo de trabajo {workflow_id} creado exitosamente")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error al crear flujo de trabajo: {e}")
            raise
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """
        Iniciar un flujo de trabajo.
        
        Args:
            workflow_id: ID del flujo de trabajo
            
        Returns:
            True si el flujo se inició exitosamente
        """
        try:
            if workflow_id not in self.workflows:
                logger.warning(f"Flujo de trabajo {workflow_id} no encontrado")
                return False
            
            if workflow_id in self.active_workflows:
                logger.warning(f"Flujo de trabajo {workflow_id} ya está activo")
                return False
            
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                logger.warning("Límite de flujos concurrentes alcanzado")
                return False
            
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Crear tarea asíncrona para ejecutar el flujo
            task = asyncio.create_task(self._execute_workflow(workflow_id))
            self.active_workflows[workflow_id] = task
            
            self.metrics["active_workflows"] += 1
            
            # Emitir evento
            await self.event_system.emit_event("workflow_started", {
                "workflow_id": workflow_id,
                "name": workflow.name
            })
            
            logger.info(f"Flujo de trabajo {workflow_id} iniciado")
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar flujo de trabajo {workflow_id}: {e}")
            return False
    
    async def _execute_workflow(self, workflow_id: str):
        """Ejecutar un flujo de trabajo."""
        try:
            workflow = self.workflows[workflow_id]
            
            # Ejecutar pasos en orden
            for step in workflow.steps:
                if workflow.status != WorkflowStatus.RUNNING:
                    break
                
                # Verificar dependencias
                if not await self._check_step_dependencies(step, workflow):
                    step.status = StepStatus.SKIPPED
                    continue
                
                # Verificar condición
                if step.condition and not await self._evaluate_condition(step.condition, workflow):
                    step.status = StepStatus.SKIPPED
                    continue
                
                # Ejecutar paso
                await self._execute_step(step, workflow)
                
                # Verificar si el flujo debe continuar
                if step.status == StepStatus.FAILED and not step.retry_attempts:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.error = f"Paso {step.name} falló: {step.error}"
                    break
            
            # Finalizar flujo
            await self._finalize_workflow(workflow)
            
        except Exception as e:
            logger.error(f"Error al ejecutar flujo de trabajo {workflow_id}: {e}")
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            await self._finalize_workflow(workflow)
        
        finally:
            # Limpiar flujo activo
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            self.metrics["active_workflows"] -= 1
    
    async def _check_step_dependencies(self, step: WorkflowStep, workflow: Workflow) -> bool:
        """Verificar dependencias de un paso."""
        for dep_id in step.dependencies:
            dep_step = next((s for s in workflow.steps if s.step_id == dep_id), None)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    
    async def _evaluate_condition(self, condition: str, workflow: Workflow) -> bool:
        """Evaluar condición de un paso."""
        try:
            # Reemplazar variables en la condición
            context = {
                "workflow": workflow,
                "steps": {step.step_id: step for step in workflow.steps}
            }
            
            # Evaluar condición (implementación básica)
            # En producción, usar un motor de reglas más robusto
            return eval(condition, {"__builtins__": {}}, context)
            
        except Exception as e:
            logger.error(f"Error al evaluar condición {condition}: {e}")
            return False
    
    async def _execute_step(self, step: WorkflowStep, workflow: Workflow):
        """Ejecutar un paso del flujo."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        for attempt in range(step.retry_attempts):
            try:
                # Enviar tarea al agente
                task_id = await self.agent_manager.submit_task(
                    task_type=step.task_type,
                    parameters=step.parameters,
                    agent_type=step.agent_type,
                    priority=5  # Prioridad alta para flujos de trabajo
                )
                
                if not task_id:
                    raise Exception(f"No se pudo enviar tarea al agente {step.agent_type}")
                
                step.task_id = task_id
                
                # Esperar completación de la tarea
                result = await self._wait_for_task_completion(task_id, step.timeout)
                
                if result["status"] == "completed":
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now()
                    step.result = result["result"]
                    break
                else:
                    raise Exception(result.get("error", "Tarea falló"))
                    
            except Exception as e:
                error_msg = f"Intento {attempt + 1} falló: {e}"
                logger.warning(error_msg)
                
                if attempt == step.retry_attempts - 1:
                    step.status = StepStatus.FAILED
                    step.error = error_msg
                    step.completed_at = datetime.now()
                else:
                    # Esperar antes del siguiente intento
                    await asyncio.sleep(2 ** attempt)  # Backoff exponencial
    
    async def _wait_for_task_completion(self, task_id: str, timeout: int) -> Dict[str, Any]:
        """Esperar completación de una tarea."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.agent_manager.get_task_status(task_id)
            
            if status:
                if status["status"] in ["completed", "failed"]:
                    return status
            
            await asyncio.sleep(1)
        
        return {"status": "timeout", "error": "Tarea excedió el tiempo límite"}
    
    async def _finalize_workflow(self, workflow: Workflow):
        """Finalizar un flujo de trabajo."""
        workflow.completed_at = datetime.now()
        
        if workflow.status == WorkflowStatus.RUNNING:
            # Verificar si todos los pasos están completados
            all_completed = all(step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] 
                              for step in workflow.steps)
            
            if all_completed:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.result = {
                    "steps_results": {step.step_id: step.result for step in workflow.steps if step.result},
                    "execution_time": (workflow.completed_at - workflow.started_at).total_seconds()
                }
                self.metrics["completed_workflows"] += 1
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = "Algunos pasos no se completaron"
                self.metrics["failed_workflows"] += 1
        else:
            self.metrics["failed_workflows"] += 1
        
        # Actualizar métricas de tiempo promedio
        if workflow.status == WorkflowStatus.COMPLETED:
            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
            if self.metrics["completed_workflows"] == 1:
                self.metrics["average_execution_time"] = execution_time
            else:
                self.metrics["average_execution_time"] = (
                    (self.metrics["average_execution_time"] * (self.metrics["completed_workflows"] - 1) + execution_time) /
                    self.metrics["completed_workflows"]
                )
        
        # Emitir evento
        await self.event_system.emit_event("workflow_completed", {
            "workflow_id": workflow.workflow_id,
            "status": workflow.status.value,
            "result": workflow.result,
            "error": workflow.error
        })
        
        logger.info(f"Flujo de trabajo {workflow.workflow_id} finalizado con estado {workflow.status.value}")
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancelar un flujo de trabajo."""
        try:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                return False
            
            # Cancelar tarea asíncrona si está activa
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].cancel()
                del self.active_workflows[workflow_id]
                self.metrics["active_workflows"] -= 1
            
            # Cancelar tareas pendientes
            for step in workflow.steps:
                if step.status == StepStatus.RUNNING and step.task_id:
                    await self.agent_manager.cancel_task(step.task_id)
                elif step.status == StepStatus.PENDING:
                    step.status = StepStatus.SKIPPED
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            
            # Emitir evento
            await self.event_system.emit_event("workflow_cancelled", {
                "workflow_id": workflow_id
            })
            
            logger.info(f"Flujo de trabajo {workflow_id} cancelado")
            return True
            
        except Exception as e:
            logger.error(f"Error al cancelar flujo de trabajo {workflow_id}: {e}")
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un flujo de trabajo."""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "result": step.result,
                    "error": step.error
                }
                for step in workflow.steps
            ],
            "result": workflow.result,
            "error": workflow.error
        }
    
    async def get_all_workflows(self) -> Dict[str, Any]:
        """Obtener todos los flujos de trabajo."""
        workflows_data = {}
        
        for workflow_id, workflow in self.workflows.items():
            workflows_data[workflow_id] = await self.get_workflow_status(workflow_id)
        
        return {
            "total_workflows": len(self.workflows),
            "active_workflows": len(self.active_workflows),
            "workflows": workflows_data,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de flujos de trabajo."""
        return {
            **self.metrics,
            "active_workflows": len(self.active_workflows),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de flujos de trabajo."""
        try:
            return {
                "status": "healthy",
                "total_workflows": len(self.workflows),
                "active_workflows": len(self.active_workflows),
                "max_concurrent_workflows": self.max_concurrent_workflows,
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check del WorkflowEngine: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




