"""
Pipeline Base Class
===================
Clase base común para todos los pipelines
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
from .base_manager import BaseManager


class PipelineStatus(Enum):
    """Estados de pipeline (común)"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class PipelineStep:
    """Paso de pipeline (común)"""
    id: str
    name: str
    config: Dict[str, Any]
    enabled: bool = True
    timeout: float = 3600.0
    retry_count: int = 3


@dataclass
class PipelineExecution:
    """Ejecución de pipeline (común)"""
    id: str
    pipeline_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING
    current_step: Optional[str] = None
    results: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []


class BasePipeline(BaseManager, ABC):
    """
    Clase base para todos los pipelines
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.steps: List[PipelineStep] = []
        self.executions: List[PipelineExecution] = []
        self.step_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def add_step(
        self,
        step_id: str,
        name: str,
        config: Dict[str, Any],
        enabled: bool = True,
        timeout: float = 3600.0,
        retry_count: int = 3
    ) -> PipelineStep:
        """Agregar paso al pipeline"""
        step = PipelineStep(
            id=step_id,
            name=name,
            config=config,
            enabled=enabled,
            timeout=timeout,
            retry_count=retry_count
        )
        self.steps.append(step)
        return step
    
    def register_step_handler(self, step_id: str, handler: Callable):
        """Registrar handler para paso"""
        self.step_handlers[step_id] = handler
    
    @abstractmethod
    def execute(
        self,
        input_data: Optional[Any] = None
    ) -> PipelineExecution:
        """Ejecutar pipeline (debe ser implementado por subclase)"""
        pass
    
    def _execute_step(
        self,
        step: PipelineStep,
        data: Any,
        execution: PipelineExecution
    ) -> Any:
        """Ejecutar paso individual con retry"""
        handler = self.step_handlers.get(step.id)
        if not handler:
            raise ValueError(f"No handler for step {step.id}")
        
        execution.current_step = step.id
        
        # Retry logic
        last_error = None
        for attempt in range(step.retry_count):
            try:
                result = handler(data, step.config)
                return result
            except Exception as e:
                last_error = e
                if attempt < step.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Step {step.id} failed after {step.retry_count} attempts: {last_error}")
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de ejecución"""
        execution = next((e for e in self.executions if e.id == execution_id), None)
        if not execution:
            return None
        
        return {
            'id': execution.id,
            'status': execution.status.value,
            'current_step': execution.current_step,
            'started_at': execution.started_at,
            'completed_at': execution.completed_at,
            'errors': execution.errors,
            'progress': self._calculate_progress(execution)
        }
    
    def _calculate_progress(self, execution: PipelineExecution) -> float:
        """Calcular progreso de ejecución"""
        if execution.status == PipelineStatus.COMPLETED:
            return 100.0
        elif execution.status == PipelineStatus.FAILED:
            return 0.0
        
        if not self.steps:
            return 0.0
        
        current_index = 0
        if execution.current_step:
            for i, step in enumerate(self.steps):
                if step.id == execution.current_step:
                    current_index = i
                    break
        
        return (current_index / len(self.steps)) * 100
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancelar ejecución"""
        execution = next((e for e in self.executions if e.id == execution_id), None)
        if execution and execution.status == PipelineStatus.RUNNING:
            execution.status = PipelineStatus.CANCELLED
            execution.completed_at = time.time()
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del pipeline"""
        base_stats = super().get_stats()
        
        executions = self.executions
        successful = len([e for e in executions if e.status == PipelineStatus.COMPLETED])
        failed = len([e for e in executions if e.status == PipelineStatus.FAILED])
        
        avg_duration = 0
        if executions:
            durations = [
                (e.completed_at or time.time()) - e.started_at
                for e in executions if e.completed_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            **base_stats,
            'total_steps': len(self.steps),
            'total_executions': len(executions),
            'successful_executions': successful,
            'failed_executions': failed,
            'success_rate': successful / len(executions) if executions else 0,
            'average_duration': avg_duration
        }
    
    def _initialize(self):
        """Inicialización del pipeline"""
        # Puede ser sobrescrito por subclases
        pass
    
    def _shutdown(self):
        """Cierre del pipeline"""
        # Cancelar ejecuciones pendientes
        for execution in self.executions:
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.CANCELLED
                execution.completed_at = time.time()

