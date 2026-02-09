"""
Base Pipeline Classes (optimizado)

Define las clases base y interfaces para el sistema de pipelines modular.
Proporciona la estructura fundamental para todos los tipos de pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class PipelineStatus(str, Enum):
    """Estados de un pipeline."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineContext:
    """
    Contexto de ejecución de un pipeline (optimizado).
    
    Contiene toda la información necesaria para la ejecución de un pipeline.
    """
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


@dataclass
class PipelineResult(Generic[R]):
    """
    Resultado de ejecución de un pipeline (optimizado).
    
    Attributes:
        success: Si la ejecución fue exitosa
        data: Datos de resultado
        error: Mensaje de error si falló
        execution_time: Tiempo de ejecución en segundos
        metadata: Metadata adicional
    """
    success: bool
    data: Optional[R] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC, Generic[T, R]):
    """
    Etapa base de un pipeline (optimizado).
    
    Cada etapa procesa un input y produce un output.
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Inicializar etapa (optimizado).
        
        Args:
            name: Nombre de la etapa
            description: Descripción de la etapa
        """
        self.name = name
        self.description = description or ""
        self.enabled = True
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    @abstractmethod
    async def execute(self, input_data: T, context: PipelineContext) -> R:
        """
        Ejecutar la etapa (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Resultado de la etapa
            
        Raises:
            Exception: Si hay error en la ejecución
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la etapa (optimizado).
        
        Returns:
            Dict con estadísticas
        """
        avg_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )
        
        return {
            "name": self.name,
            "enabled": self.enabled,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time
        }


class BasePipeline(ABC, Generic[T, R]):
    """
    Pipeline base abstracto (optimizado).
    
    Define la interfaz común para todos los tipos de pipelines.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        stages: Optional[List[PipelineStage]] = None
    ):
        """
        Inicializar pipeline (optimizado).
        
        Args:
            name: Nombre del pipeline
            description: Descripción del pipeline
            stages: Lista de etapas (opcional)
        """
        self.name = name
        self.description = description or ""
        self.stages: List[PipelineStage] = stages or []
        self.context: Optional[PipelineContext] = None
        self._middleware: List[Callable] = []
    
    def add_stage(self, stage: PipelineStage) -> 'BasePipeline':
        """
        Agregar etapa al pipeline (optimizado).
        
        Args:
            stage: Etapa a agregar
            
        Returns:
            Self para method chaining
        """
        if not isinstance(stage, PipelineStage):
            raise TypeError(f"Stage must be instance of PipelineStage, got {type(stage)}")
        
        self.stages.append(stage)
        logger.debug(f"Added stage '{stage.name}' to pipeline '{self.name}'")
        return self
    
    def add_middleware(self, middleware: Callable) -> 'BasePipeline':
        """
        Agregar middleware al pipeline (optimizado).
        
        Args:
            middleware: Función middleware
            
        Returns:
            Self para method chaining
        """
        self._middleware.append(middleware)
        return self
    
    @abstractmethod
    async def execute(self, input_data: T) -> PipelineResult[R]:
        """
        Ejecutar el pipeline (optimizado).
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Resultado de la ejecución
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtener información del pipeline (optimizado).
        
        Returns:
            Dict con información del pipeline
        """
        return {
            "name": self.name,
            "description": self.description,
            "stage_count": len(self.stages),
            "stages": [stage.name for stage in self.stages],
            "middleware_count": len(self._middleware),
            "status": self.context.status.value if self.context else None
        }

