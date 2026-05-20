"""
Workflow - Sistema de Workflow/Pipeline
========================================

Sistema para crear y ejecutar workflows y pipelines de tareas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Estado de un paso"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Paso de un workflow"""
    id: str
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: Optional[float] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Workflow:
    """
    Workflow/Pipeline de tareas.
    
    Ejecuta pasos en orden con manejo de dependencias.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.results: Dict[str, Any] = {}
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status: StepStatus = StepStatus.PENDING
    
    def add_step(
        self,
        step_id: str,
        name: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None,
        **metadata
    ) -> WorkflowStep:
        """
        Agregar paso al workflow.
        
        Args:
            step_id: ID único del paso
            name: Nombre del paso
            func: Función a ejecutar (puede ser async o sync)
            dependencies: Lista de IDs de pasos de los que depende
            retry_count: Número de reintentos
            timeout: Timeout en segundos
            **metadata: Metadata adicional
            
        Returns:
            WorkflowStep creado
        """
        step = WorkflowStep(
            id=step_id,
            name=name,
            func=func,
            dependencies=dependencies or [],
            retry_count=retry_count,
            timeout=timeout,
            metadata=metadata
        )
        
        self.steps[step_id] = step
        logger.debug(f"📋 Step added to workflow {self.name}: {step_id}")
        return step
    
    async def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecutar workflow.
        
        Args:
            context: Contexto inicial para los pasos
            
        Returns:
            Diccionario con resultados de todos los pasos
        """
        self.started_at = datetime.now()
        self.status = StepStatus.RUNNING
        context = context or {}
        
        logger.info(f"🚀 Starting workflow: {self.name}")
        
        # Ordenar pasos por dependencias (topological sort)
        execution_order = self._get_execution_order()
        
        for step_id in execution_order:
            step = self.steps[step_id]
            
            # Verificar dependencias
            if not self._check_dependencies(step):
                step.status = StepStatus.SKIPPED
                logger.warning(f"⏭️ Step skipped due to failed dependencies: {step_id}")
                continue
            
            # Ejecutar paso
            await self._execute_step(step, context)
            
            # Si falla y no hay reintentos, detener workflow
            if step.status == StepStatus.FAILED and step.retry_count == 0:
                self.status = StepStatus.FAILED
                logger.error(f"❌ Workflow failed at step: {step_id}")
                break
        
        # Determinar estado final
        if all(s.status == StepStatus.COMPLETED or s.status == StepStatus.SKIPPED 
               for s in self.steps.values()):
            self.status = StepStatus.COMPLETED
        elif any(s.status == StepStatus.FAILED for s in self.steps.values()):
            self.status = StepStatus.FAILED
        
        self.completed_at = datetime.now()
        
        logger.info(
            f"✅ Workflow completed: {self.name} "
            f"({(self.completed_at - self.started_at).total_seconds():.2f}s)"
        )
        
        return self.results
    
    def _get_execution_order(self) -> List[str]:
        """Obtener orden de ejecución basado en dependencias"""
        # Topological sort simple
        visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in visited:
                return
            
            visited.add(step_id)
            step = self.steps[step_id]
            
            for dep in step.dependencies:
                if dep in self.steps:
                    visit(dep)
            
            order.append(step_id)
        
        for step_id in self.steps.keys():
            visit(step_id)
        
        return order
    
    def _check_dependencies(self, step: WorkflowStep) -> bool:
        """Verificar que todas las dependencias se completaron exitosamente"""
        for dep_id in step.dependencies:
            if dep_id not in self.steps:
                logger.warning(f"Dependency {dep_id} not found for step {step.id}")
                return False
            
            dep_step = self.steps[dep_id]
            if dep_step.status != StepStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> None:
        """Ejecutar un paso del workflow"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        logger.debug(f"🔄 Executing step: {step.name} ({step.id})")
        
        try:
            # Ejecutar con timeout si está configurado
            if step.timeout:
                result = await asyncio.wait_for(
                    self._call_step_func(step, context),
                    timeout=step.timeout
                )
            else:
                result = await self._call_step_func(step, context)
            
            step.result = result
            step.status = StepStatus.COMPLETED
            self.results[step.id] = result
            context[step.id] = result
            
            logger.debug(f"✅ Step completed: {step.name}")
            
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = f"Timeout after {step.timeout}s"
            logger.error(f"⏱️ Step timed out: {step.name}")
            
        except Exception as e:
            # Reintentar si hay reintentos disponibles
            if step.retry_count > 0:
                step.retry_count -= 1
                logger.warning(f"🔄 Retrying step {step.name} ({step.retry_count} retries left)")
                await self._execute_step(step, context)
            else:
                step.status = StepStatus.FAILED
                step.error = str(e)
                logger.error(f"❌ Step failed: {step.name} - {e}")
        
        finally:
            step.completed_at = datetime.now()
    
    async def _call_step_func(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Llamar función del paso"""
        if asyncio.iscoroutinefunction(step.func):
            return await step.func(context)
        else:
            return step.func(context)
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen del workflow"""
        return {
            "name": self.name,
            "status": self.status.value,
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED),
            "failed_steps": sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": (
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at else None
            )
        }




