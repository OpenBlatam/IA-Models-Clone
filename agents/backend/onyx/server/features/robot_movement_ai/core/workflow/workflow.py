"""
Workflow System
===============

Sistema de flujos de trabajo.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Estado de paso."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Paso de flujo de trabajo."""
    step_id: str
    name: str
    func: Callable
    condition: Optional[Callable] = None
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    retry_count: int = 0
    timeout: Optional[float] = None


class Workflow:
    """
    Flujo de trabajo.
    
    Ejecuta pasos en secuencia o paralelo.
    """
    
    def __init__(self, workflow_id: str, name: str):
        """
        Inicializar flujo de trabajo.
        
        Args:
            workflow_id: ID único del flujo
            name: Nombre del flujo
        """
        self.workflow_id = workflow_id
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_step(
        self,
        step_id: str,
        name: str,
        func: Callable,
        condition: Optional[Callable] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ) -> WorkflowStep:
        """
        Agregar paso al flujo.
        
        Args:
            step_id: ID único del paso
            name: Nombre del paso
            func: Función a ejecutar
            condition: Condición para ejecutar (opcional)
            on_success: Callback en éxito (opcional)
            on_failure: Callback en fallo (opcional)
            retry_count: Número de reintentos
            timeout: Timeout en segundos
            
        Returns:
            Paso creado
        """
        step = WorkflowStep(
            step_id=step_id,
            name=name,
            func=func,
            condition=condition,
            on_success=on_success,
            on_failure=on_failure,
            retry_count=retry_count,
            timeout=timeout
        )
        
        self.steps.append(step)
        return step
    
    async def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecutar flujo de trabajo.
        
        Args:
            context: Contexto compartido entre pasos
            
        Returns:
            Resultado de la ejecución
        """
        if context is None:
            context = {}
        
        execution_id = f"{self.workflow_id}_{len(self.execution_history)}"
        logger.info(f"Executing workflow: {self.name} ({execution_id})")
        
        results = {}
        status = "completed"
        
        for step in self.steps:
            # Verificar condición
            if step.condition and not step.condition(context):
                logger.info(f"Skipping step: {step.name} (condition not met)")
                results[step.step_id] = {"status": "skipped"}
                continue
            
            # Ejecutar paso
            try:
                step_status = StepStatus.RUNNING
                result = await self._execute_step(step, context)
                step_status = StepStatus.COMPLETED
                
                results[step.step_id] = {
                    "status": "completed",
                    "result": result
                }
                
                # Ejecutar callback de éxito
                if step.on_success:
                    await self._safe_call(step.on_success, context, result)
            
            except Exception as e:
                step_status = StepStatus.FAILED
                status = "failed"
                
                results[step.step_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                
                logger.error(f"Step failed: {step.name} - {e}")
                
                # Ejecutar callback de fallo
                if step.on_failure:
                    await self._safe_call(step.on_failure, context, e)
                
                # Decidir si continuar o detener
                break
        
        execution_result = {
            "execution_id": execution_id,
            "workflow_id": self.workflow_id,
            "status": status,
            "steps": results
        }
        
        self.execution_history.append(execution_result)
        return execution_result
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """Ejecutar paso individual."""
        import asyncio
        
        if step.timeout:
            if asyncio.iscoroutinefunction(step.func):
                return await asyncio.wait_for(
                    step.func(context),
                    timeout=step.timeout
                )
            else:
                # Para funciones sync, ejecutar en thread
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, step.func, context),
                    timeout=step.timeout
                )
        else:
            if asyncio.iscoroutinefunction(step.func):
                return await step.func(context)
            else:
                return step.func(context)
    
    async def _safe_call(self, func: Callable, *args) -> None:
        """Llamar función de forma segura."""
        try:
            if asyncio.iscoroutinefunction(func):
                await func(*args)
            else:
                func(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de ejecuciones."""
        return self.execution_history[-limit:]


class WorkflowManager:
    """Gestor de flujos de trabajo."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.workflows: Dict[str, Workflow] = {}
    
    def create_workflow(self, workflow_id: str, name: str) -> Workflow:
        """Crear nuevo flujo de trabajo."""
        workflow = Workflow(workflow_id, name)
        self.workflows[workflow_id] = workflow
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Obtener flujo de trabajo."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """Listar todos los flujos."""
        return list(self.workflows.values())


# Instancia global
_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Obtener instancia global del gestor de flujos."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager

