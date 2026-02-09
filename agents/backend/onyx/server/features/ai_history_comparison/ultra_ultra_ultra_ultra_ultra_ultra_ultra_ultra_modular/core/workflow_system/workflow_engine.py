"""
Workflow Engine - Motor de Workflows
===================================

Sistema de workflows con patrones avanzados de orquestación y ejecución asíncrona.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from ..interfaces.base_interfaces import IWorkflow, IComponent

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Estados de workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    """Estados de pasos."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class StepType(Enum):
    """Tipos de pasos."""
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    LOOP = "loop"
    DELAY = "delay"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class WorkflowStep:
    """Paso de workflow."""
    
    def __init__(self, step_id: str, step_type: StepType, 
                 name: str, handler: Callable = None, 
                 config: Dict[str, Any] = None):
        self.step_id = step_id
        self.step_type = step_type
        self.name = name
        self.handler = handler
        self.config = config or {}
        self.status = StepStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.retry_count = 0
        self.max_retries = 3
        self.timeout_seconds = 300
        self.dependencies: List[str] = []
        self.conditions: List[Callable] = []
        self.on_success: List[str] = []
        self.on_failure: List[str] = []
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_dependency(self, step_id: str) -> None:
        """Agregar dependencia."""
        if step_id not in self.dependencies:
            self.dependencies.append(step_id)
    
    def add_condition(self, condition: Callable) -> None:
        """Agregar condición."""
        self.conditions.append(condition)
    
    def add_success_handler(self, step_id: str) -> None:
        """Agregar handler de éxito."""
        if step_id not in self.on_success:
            self.on_success.append(step_id)
    
    def add_failure_handler(self, step_id: str) -> None:
        """Agregar handler de fallo."""
        if step_id not in self.on_failure:
            self.on_failure.append(step_id)
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Verificar si puede ejecutarse."""
        # Verificar dependencias
        for dep in self.dependencies:
            if dep not in completed_steps:
                return False
        
        # Verificar condiciones
        for condition in self.conditions:
            try:
                if not condition():
                    return False
            except Exception as e:
                logger.error(f"Error evaluating condition for step {self.step_id}: {e}")
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del paso."""
        execution_time = None
        if self.started_at and self.completed_at:
            execution_time = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "name": self.name,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": execution_time,
            "has_error": self.error is not None,
            "error_message": str(self.error) if self.error else None
        }


class WorkflowContext:
    """Contexto de workflow."""
    
    def __init__(self, workflow_id: str, initial_data: Any = None):
        self.workflow_id = workflow_id
        self.data = initial_data
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.variables: Dict[str, Any] = {}
        self.global_variables: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.current_step = None
        self.previous_step = None
    
    def set_variable(self, name: str, value: Any) -> None:
        """Establecer variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Obtener variable."""
        return self.variables.get(name, default)
    
    def set_global_variable(self, name: str, value: Any) -> None:
        """Establecer variable global."""
        self.global_variables[name] = value
    
    def get_global_variable(self, name: str, default: Any = None) -> Any:
        """Obtener variable global."""
        return self.global_variables.get(name, default)
    
    def add_execution_record(self, step_id: str, action: str, details: Dict[str, Any] = None) -> None:
        """Agregar registro de ejecución."""
        record = {
            "step_id": step_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        self.execution_history.append(record)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de ejecución."""
        return self.execution_history.copy()


class WorkflowDefinition:
    """Definición de workflow."""
    
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.start_step = None
        self.end_steps: List[str] = []
        self.timeout_seconds = 3600
        self.max_retries = 3
        self.created_at = datetime.utcnow()
        self.version = "1.0.0"
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_step(self, step: WorkflowStep) -> None:
        """Agregar paso."""
        self.steps[step.step_id] = step
    
    def remove_step(self, step_id: str) -> bool:
        """Remover paso."""
        if step_id in self.steps:
            del self.steps[step_id]
            return True
        return False
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Obtener paso."""
        return self.steps.get(step_id)
    
    def get_next_steps(self, current_step_id: str) -> List[WorkflowStep]:
        """Obtener siguientes pasos."""
        current_step = self.get_step(current_step_id)
        if not current_step:
            return []
        
        next_steps = []
        for step_id in current_step.on_success:
            step = self.get_step(step_id)
            if step:
                next_steps.append(step)
        
        return next_steps
    
    def get_failure_steps(self, current_step_id: str) -> List[WorkflowStep]:
        """Obtener pasos de fallo."""
        current_step = self.get_step(current_step_id)
        if not current_step:
            return []
        
        failure_steps = []
        for step_id in current_step.on_failure:
            step = self.get_step(step_id)
            if step:
                failure_steps.append(step)
        
        return failure_steps
    
    def validate(self) -> Dict[str, Any]:
        """Validar definición de workflow."""
        errors = []
        warnings = []
        
        # Verificar que existe al menos un paso
        if not self.steps:
            errors.append("Workflow must have at least one step")
        
        # Verificar que existe un paso de inicio
        if not self.start_step:
            errors.append("Workflow must have a start step")
        elif self.start_step not in self.steps:
            errors.append(f"Start step '{self.start_step}' not found in steps")
        
        # Verificar dependencias
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    errors.append(f"Step '{step_id}' has invalid dependency '{dep}'")
            
            for success_step in step.on_success:
                if success_step not in self.steps:
                    errors.append(f"Step '{step_id}' has invalid success handler '{success_step}'")
            
            for failure_step in step.on_failure:
                if failure_step not in self.steps:
                    errors.append(f"Step '{step_id}' has invalid failure handler '{failure_step}'")
        
        # Verificar pasos de fin
        if not self.end_steps:
            warnings.append("Workflow has no defined end steps")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "step_count": len(self.steps),
            "has_start_step": self.start_step is not None,
            "end_step_count": len(self.end_steps)
        }


class WorkflowInstance:
    """Instancia de workflow."""
    
    def __init__(self, instance_id: str, definition: WorkflowDefinition, 
                 context: WorkflowContext):
        self.instance_id = instance_id
        self.definition = definition
        self.context = context
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.current_step_id = None
        self.completed_steps: List[str] = []
        self.failed_steps: List[str] = []
        self.retry_count = 0
        self.max_retries = definition.max_retries
        self.timeout_seconds = definition.timeout_seconds
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.metadata: Dict[str, Any] = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la instancia."""
        execution_time = None
        if self.started_at and self.completed_at:
            execution_time = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "instance_id": self.instance_id,
            "workflow_id": self.definition.workflow_id,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": execution_time,
            "current_step_id": self.current_step_id,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "has_error": self.error is not None,
            "error_message": str(self.error) if self.error else None
        }


class WorkflowEngine(IWorkflow, IComponent):
    """Motor de workflows."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._instances: Dict[str, WorkflowInstance] = {}
        self._running_instances: Set[str] = set()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._execution_stats: Dict[str, Any] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Inicializar motor de workflows."""
        try:
            # Iniciar tarea de limpieza
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            
            self._initialized = True
            logger.info(f"Workflow engine {self.name} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing workflow engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar motor de workflows."""
        try:
            # Cancelar tarea de limpieza
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cancelar workflows en ejecución
            for instance_id in list(self._running_instances):
                await self.cancel_workflow(instance_id)
            
            self._initialized = False
            logger.info(f"Workflow engine {self.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down workflow engine: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del motor."""
        return self._initialized
    
    @property
    def name(self) -> str:
        return f"WorkflowEngine_{self.name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def start(self, context: Any) -> str:
        """Iniciar workflow."""
        # Este método será implementado por las subclases
        raise NotImplementedError
    
    async def execute_step(self, workflow_id: str, step: str) -> Any:
        """Ejecutar paso del workflow."""
        # Este método será implementado por las subclases
        raise NotImplementedError
    
    async def get_status(self, workflow_id: str) -> str:
        """Obtener estado del workflow."""
        try:
            instance = self._instances.get(workflow_id)
            if instance:
                return instance.status.value
            return "not_found"
            
        except Exception as e:
            logger.error(f"Error getting workflow status {workflow_id}: {e}")
            return "error"
    
    async def register_workflow(self, definition: WorkflowDefinition) -> bool:
        """Registrar definición de workflow."""
        try:
            # Validar definición
            validation = definition.validate()
            if not validation["valid"]:
                logger.error(f"Invalid workflow definition: {validation['errors']}")
                return False
            
            async with self._lock:
                self._definitions[definition.workflow_id] = definition
            
            logger.info(f"Registered workflow definition: {definition.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering workflow definition: {e}")
            return False
    
    async def unregister_workflow(self, workflow_id: str) -> bool:
        """Desregistrar definición de workflow."""
        try:
            async with self._lock:
                if workflow_id in self._definitions:
                    del self._definitions[workflow_id]
                    logger.info(f"Unregistered workflow definition: {workflow_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering workflow definition: {e}")
            return False
    
    async def start_workflow(self, workflow_id: str, initial_data: Any = None) -> str:
        """Iniciar instancia de workflow."""
        try:
            definition = self._definitions.get(workflow_id)
            if not definition:
                raise ValueError(f"Workflow definition {workflow_id} not found")
            
            # Crear instancia
            instance_id = str(uuid.uuid4())
            context = WorkflowContext(instance_id, initial_data)
            instance = WorkflowInstance(instance_id, definition, context)
            
            async with self._lock:
                self._instances[instance_id] = instance
                self._running_instances.add(instance_id)
            
            # Iniciar ejecución
            asyncio.create_task(self._execute_workflow(instance))
            
            logger.info(f"Started workflow instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Error starting workflow {workflow_id}: {e}")
            raise
    
    async def _execute_workflow(self, instance: WorkflowInstance) -> None:
        """Ejecutar workflow."""
        try:
            instance.status = WorkflowStatus.RUNNING
            instance.started_at = datetime.utcnow()
            
            # Ejecutar pasos
            current_step_id = instance.definition.start_step
            while current_step_id:
                step = instance.definition.get_step(current_step_id)
                if not step:
                    raise ValueError(f"Step {current_step_id} not found")
                
                # Verificar si puede ejecutarse
                if not step.can_execute(instance.completed_steps):
                    logger.warning(f"Step {current_step_id} cannot execute, skipping")
                    current_step_id = self._get_next_step(instance, current_step_id, False)
                    continue
                
                # Ejecutar paso
                success = await self._execute_step(instance, step)
                
                if success:
                    instance.completed_steps.append(current_step_id)
                    current_step_id = self._get_next_step(instance, current_step_id, True)
                else:
                    instance.failed_steps.append(current_step_id)
                    current_step_id = self._get_next_step(instance, current_step_id, False)
                
                # Verificar si es un paso de fin
                if current_step_id in instance.definition.end_steps:
                    break
            
            # Completar workflow
            instance.status = WorkflowStatus.COMPLETED
            instance.completed_at = datetime.utcnow()
            
            # Remover de instancias en ejecución
            async with self._lock:
                self._running_instances.discard(instance.instance_id)
            
            # Actualizar estadísticas
            await self._update_execution_stats("completed", instance)
            
            logger.info(f"Workflow instance {instance.instance_id} completed")
            
        except asyncio.CancelledError:
            instance.status = WorkflowStatus.CANCELLED
            instance.completed_at = datetime.utcnow()
            async with self._lock:
                self._running_instances.discard(instance.instance_id)
            logger.info(f"Workflow instance {instance.instance_id} cancelled")
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = e
            instance.completed_at = datetime.utcnow()
            async with self._lock:
                self._running_instances.discard(instance.instance_id)
            await self._update_execution_stats("failed", instance)
            logger.error(f"Workflow instance {instance.instance_id} failed: {e}")
    
    async def _execute_step(self, instance: WorkflowInstance, step: WorkflowStep) -> bool:
        """Ejecutar paso individual."""
        try:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.utcnow()
            instance.current_step_id = step.step_id
            
            # Ejecutar handler del paso
            if step.handler:
                if asyncio.iscoroutinefunction(step.handler):
                    step.result = await asyncio.wait_for(
                        step.handler(instance.context),
                        timeout=step.timeout_seconds
                    )
                else:
                    step.result = step.handler(instance.context)
            
            # Marcar como completado
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            
            # Agregar registro de ejecución
            instance.context.add_execution_record(
                step.step_id, 
                "completed", 
                {"result": step.result}
            )
            
            return True
            
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = Exception(f"Step timeout after {step.timeout_seconds} seconds")
            step.completed_at = datetime.utcnow()
            
            instance.context.add_execution_record(
                step.step_id, 
                "timeout", 
                {"timeout_seconds": step.timeout_seconds}
            )
            
            return False
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = e
            step.completed_at = datetime.utcnow()
            
            instance.context.add_execution_record(
                step.step_id, 
                "failed", 
                {"error": str(e)}
            )
            
            return False
    
    def _get_next_step(self, instance: WorkflowInstance, current_step_id: str, success: bool) -> Optional[str]:
        """Obtener siguiente paso."""
        current_step = instance.definition.get_step(current_step_id)
        if not current_step:
            return None
        
        if success:
            # Usar handlers de éxito
            if current_step.on_success:
                return current_step.on_success[0]  # Tomar el primero
        else:
            # Usar handlers de fallo
            if current_step.on_failure:
                return current_step.on_failure[0]  # Tomar el primero
        
        return None
    
    async def pause_workflow(self, instance_id: str) -> bool:
        """Pausar workflow."""
        try:
            instance = self._instances.get(instance_id)
            if not instance:
                return False
            
            if instance.status == WorkflowStatus.RUNNING:
                instance.status = WorkflowStatus.PAUSED
                logger.info(f"Paused workflow instance: {instance_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error pausing workflow {instance_id}: {e}")
            return False
    
    async def resume_workflow(self, instance_id: str) -> bool:
        """Reanudar workflow."""
        try:
            instance = self._instances.get(instance_id)
            if not instance:
                return False
            
            if instance.status == WorkflowStatus.PAUSED:
                instance.status = WorkflowStatus.RUNNING
                # Reanudar ejecución
                asyncio.create_task(self._execute_workflow(instance))
                logger.info(f"Resumed workflow instance: {instance_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resuming workflow {instance_id}: {e}")
            return False
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancelar workflow."""
        try:
            instance = self._instances.get(instance_id)
            if not instance:
                return False
            
            if instance.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
                instance.status = WorkflowStatus.CANCELLED
                instance.completed_at = datetime.utcnow()
                
                async with self._lock:
                    self._running_instances.discard(instance_id)
                
                logger.info(f"Cancelled workflow instance: {instance_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling workflow {instance_id}: {e}")
            return False
    
    async def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Obtener instancia de workflow."""
        return self._instances.get(instance_id)
    
    async def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Obtener definición de workflow."""
        return self._definitions.get(workflow_id)
    
    async def list_workflow_definitions(self) -> List[str]:
        """Listar definiciones de workflow."""
        return list(self._definitions.keys())
    
    async def list_workflow_instances(self, status: WorkflowStatus = None) -> List[str]:
        """Listar instancias de workflow."""
        if status:
            return [instance_id for instance_id, instance in self._instances.items() 
                   if instance.status == status]
        return list(self._instances.keys())
    
    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de workflows."""
        try:
            status_counts = {}
            for instance in self._instances.values():
                status = instance.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "engine_name": self.name,
                "definition_count": len(self._definitions),
                "instance_count": len(self._instances),
                "running_count": len(self._running_instances),
                "status_counts": status_counts,
                "execution_stats": self._execution_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow stats: {e}")
            return {}
    
    async def _update_execution_stats(self, action: str, instance: WorkflowInstance) -> None:
        """Actualizar estadísticas de ejecución."""
        try:
            if action not in self._execution_stats:
                self._execution_stats[action] = 0
            self._execution_stats[action] += 1
            
            # Estadísticas por workflow
            workflow_key = f"{action}_{instance.definition.workflow_id}"
            if workflow_key not in self._execution_stats:
                self._execution_stats[workflow_key] = 0
            self._execution_stats[workflow_key] += 1
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    async def _cleanup_worker(self) -> None:
        """Worker para limpieza de instancias completadas."""
        try:
            while True:
                await asyncio.sleep(3600)  # Limpiar cada hora
                
                # Limpiar instancias completadas antiguas (más de 24 horas)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                instances_to_remove = []
                
                async with self._lock:
                    for instance_id, instance in self._instances.items():
                        if (instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                            instance.completed_at and instance.completed_at < cutoff_time):
                            instances_to_remove.append(instance_id)
                    
                    for instance_id in instances_to_remove:
                        del self._instances[instance_id]
                
                if instances_to_remove:
                    logger.info(f"Cleaned up {len(instances_to_remove)} old workflow instances")
                    
        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup worker: {e}")




