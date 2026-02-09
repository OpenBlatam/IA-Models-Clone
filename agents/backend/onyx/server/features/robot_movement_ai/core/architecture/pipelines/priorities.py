"""
Pipeline Priorities
===================

Sistema de prioridades para etapas de pipeline.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import IntEnum

from .stages import PipelineStage

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Prioridad de etapa."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5


@dataclass(frozen=True)
class PrioritizedStage:
    """
    Etapa con prioridad.
    Inmutable para mejor seguridad.
    """
    stage: PipelineStage
    priority: Priority
    weight: float = 1.0


def _sort_by_priority(
    prioritized_stages: List[PrioritizedStage]
) -> List[PrioritizedStage]:
    """
    Ordenar etapas por prioridad (función pura).
    
    Args:
        prioritized_stages: Lista de etapas priorizadas
        
    Returns:
        Etapas ordenadas por prioridad (mayor primero)
    """
    return sorted(
        prioritized_stages,
        key=lambda x: (x.priority.value, x.weight),
        reverse=True
    )


class PriorityScheduler:
    """
    Planificador por prioridades.
    Optimizado con funciones puras y mejor rendimiento.
    """
    
    def __init__(self) -> None:
        """Inicializar planificador."""
        self._stage_priorities: Dict[str, Priority] = {}
        self._stage_weights: Dict[str, float] = {}
    
    def set_priority(
        self,
        stage_name: str,
        priority: Priority,
        weight: float = 1.0
    ) -> None:
        """
        Establecer prioridad de etapa.
        
        Args:
            stage_name: Nombre de la etapa
            priority: Prioridad
            weight: Peso adicional para desempate
        """
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
        
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self._stage_priorities[stage_name] = priority
        self._stage_weights[stage_name] = weight
    
    def get_priority(self, stage_name: str) -> Priority:
        """
        Obtener prioridad de etapa.
        
        Args:
            stage_name: Nombre de la etapa
            
        Returns:
            Prioridad de la etapa o NORMAL por defecto
        """
        return self._stage_priorities.get(stage_name, Priority.NORMAL)
    
    def get_weight(self, stage_name: str) -> float:
        """
        Obtener peso de etapa.
        
        Args:
            stage_name: Nombre de la etapa
            
        Returns:
            Peso de la etapa o 1.0 por defecto
        """
        return self._stage_weights.get(stage_name, 1.0)
    
    def schedule(
        self,
        stages: List[PipelineStage]
    ) -> List[PrioritizedStage]:
        """
        Planificar etapas por prioridad.
        
        Args:
            stages: Lista de etapas
            
        Returns:
            Etapas planificadas ordenadas por prioridad
        """
        if not stages:
            return []
        
        prioritized = [
            PrioritizedStage(
                stage=stage,
                priority=self.get_priority(stage.get_name()),
                weight=self.get_weight(stage.get_name())
            )
            for stage in stages
        ]
        
        return _sort_by_priority(prioritized)


class PriorityPipeline:
    """
    Pipeline con soporte para prioridades.
    Optimizado con mejor validación y rendimiento.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        priority_scheduler: Optional[PriorityScheduler] = None,
        **kwargs
    ) -> None:
        """
        Inicializar pipeline con prioridades.
        
        Args:
            name: Nombre del pipeline
            priority_scheduler: Planificador de prioridades
            **kwargs: Argumentos adicionales
        """
        from .pipeline import Pipeline
        
        self._pipeline = Pipeline(name, **kwargs)
        self.priority_scheduler = priority_scheduler or PriorityScheduler()
        self._original_stages: List[PipelineStage] = []
    
    def add_stage(
        self,
        stage: PipelineStage,
        priority: Priority = Priority.NORMAL,
        weight: float = 1.0
    ) -> 'PriorityPipeline':
        """
        Agregar etapa con prioridad.
        
        Args:
            stage: Etapa
            priority: Prioridad
            weight: Peso adicional para desempate
            
        Returns:
            Self para chaining
        """
        if not stage:
            raise ValueError("Stage cannot be None")
        
        self._original_stages.append(stage)
        self.priority_scheduler.set_priority(
            stage.get_name(),
            priority,
            weight
        )
        
        self._reorder_stages()
        
        return self
    
    def set_stage_priority(
        self,
        stage_name: str,
        priority: Priority,
        weight: Optional[float] = None
    ) -> 'PriorityPipeline':
        """
        Cambiar prioridad de etapa existente.
        
        Args:
            stage_name: Nombre de la etapa
            priority: Nueva prioridad
            weight: Nuevo peso (opcional)
            
        Returns:
            Self para chaining
        """
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
        
        if weight is not None:
            self.priority_scheduler.set_priority(stage_name, priority, weight)
        else:
            current_weight = self.priority_scheduler.get_weight(stage_name)
            self.priority_scheduler.set_priority(
                stage_name,
                priority,
                current_weight
            )
        
        self._reorder_stages()
        
        return self
    
    def _reorder_stages(self) -> None:
        """Reordenar etapas según prioridades."""
        prioritized = self.priority_scheduler.schedule(self._original_stages)
        self._pipeline.stages = [p.stage for p in prioritized]
    
    @property
    def stages(self) -> List[PipelineStage]:
        """Obtener etapas del pipeline."""
        return self._pipeline.stages
