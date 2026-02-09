"""
Pipeline Metrics
================

Sistema de métricas y telemetría para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class StageMetrics:
    """
    Métricas de una etapa.
    Inmutable para mejor seguridad.
    """
    stage_name: str
    execution_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    last_execution_time: Optional[float] = None
    
    @property
    def average_duration(self) -> float:
        """
        Duración promedio.
        
        Returns:
            Duración promedio en segundos
        """
        if self.execution_count == 0:
            return 0.0
        return self.total_duration / self.execution_count
    
    @property
    def success_rate(self) -> float:
        """
        Tasa de éxito.
        
        Returns:
            Tasa de éxito entre 0.0 y 1.0
        """
        if self.execution_count == 0:
            return 0.0
        return (self.execution_count - self.error_count) / self.execution_count


@dataclass(frozen=True)
class PipelineMetrics:
    """
    Métricas de un pipeline completo.
    Inmutable para mejor seguridad.
    """
    pipeline_name: str
    execution_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    
    @property
    def average_duration(self) -> float:
        """
        Duración promedio.
        
        Returns:
            Duración promedio en segundos
        """
        if self.execution_count == 0:
            return 0.0
        return self.total_duration / self.execution_count
    
    @property
    def success_rate(self) -> float:
        """
        Tasa de éxito.
        
        Returns:
            Tasa de éxito entre 0.0 y 1.0
        """
        if self.execution_count == 0:
            return 0.0
        return (self.execution_count - self.error_count) / self.execution_count


def _create_stage_metrics(
    stage_name: str,
    duration: float,
    has_error: bool,
    existing: Optional[StageMetrics] = None
) -> StageMetrics:
    """
    Crear o actualizar métricas de etapa (función pura).
    
    Args:
        stage_name: Nombre de la etapa
        duration: Duración en segundos
        has_error: Si hubo error
        existing: Métricas existentes (opcional)
        
    Returns:
        Nuevas métricas
    """
    if existing:
        return StageMetrics(
            stage_name=stage_name,
            execution_count=existing.execution_count + 1,
            total_duration=existing.total_duration + duration,
            min_duration=min(existing.min_duration, duration),
            max_duration=max(existing.max_duration, duration),
            error_count=existing.error_count + (1 if has_error else 0),
            last_execution_time=time.time()
        )
    
    return StageMetrics(
        stage_name=stage_name,
        execution_count=1,
        total_duration=duration,
        min_duration=duration,
        max_duration=duration,
        error_count=1 if has_error else 0,
        last_execution_time=time.time()
    )


def _create_pipeline_metrics(
    pipeline_name: str,
    duration: float,
    has_error: bool,
    existing: Optional[PipelineMetrics] = None
) -> PipelineMetrics:
    """
    Crear o actualizar métricas de pipeline (función pura).
    
    Args:
        pipeline_name: Nombre del pipeline
        duration: Duración en segundos
        has_error: Si hubo error
        existing: Métricas existentes (opcional)
        
    Returns:
        Nuevas métricas
    """
    if existing:
        return PipelineMetrics(
            pipeline_name=pipeline_name,
            execution_count=existing.execution_count + 1,
            total_duration=existing.total_duration + duration,
            min_duration=min(existing.min_duration, duration),
            max_duration=max(existing.max_duration, duration),
            error_count=existing.error_count + (1 if has_error else 0),
            stage_metrics=existing.stage_metrics
        )
    
    return PipelineMetrics(
        pipeline_name=pipeline_name,
        execution_count=1,
        total_duration=duration,
        min_duration=duration,
        max_duration=duration,
        error_count=1 if has_error else 0,
        stage_metrics={}
    )


class MetricsCollector:
    """
    Colector de métricas para pipelines.
    Optimizado con mejor thread-safety y rendimiento.
    """
    
    def __init__(self) -> None:
        """Inicializar colector."""
        self._pipeline_metrics: Dict[str, PipelineMetrics] = {}
        self._lock = threading.RLock()
    
    def record_stage_metrics(
        self,
        stage_name: str,
        duration: float,
        has_error: bool,
        pipeline_name: str = "default"
    ) -> None:
        """
        Registrar métricas de etapa.
        
        Args:
            stage_name: Nombre de la etapa
            duration: Duración en segundos
            has_error: Si hubo error
            pipeline_name: Nombre del pipeline
        """
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
        
        if duration < 0:
            raise ValueError("Duration cannot be negative")
        
        with self._lock:
            pipeline_metrics = self._pipeline_metrics.get(pipeline_name)
            if not pipeline_metrics:
                pipeline_metrics = PipelineMetrics(pipeline_name)
                self._pipeline_metrics[pipeline_name] = pipeline_metrics
            
            existing_stage = pipeline_metrics.stage_metrics.get(stage_name)
            new_stage_metrics = _create_stage_metrics(
                stage_name, duration, has_error, existing_stage
            )
            
            new_pipeline_metrics = PipelineMetrics(
                pipeline_name=pipeline_metrics.pipeline_name,
                execution_count=pipeline_metrics.execution_count,
                total_duration=pipeline_metrics.total_duration,
                min_duration=pipeline_metrics.min_duration,
                max_duration=pipeline_metrics.max_duration,
                error_count=pipeline_metrics.error_count,
                stage_metrics={
                    **pipeline_metrics.stage_metrics,
                    stage_name: new_stage_metrics
                }
            )
            
            self._pipeline_metrics[pipeline_name] = new_pipeline_metrics
    
    def record_pipeline_metrics(
        self,
        pipeline_name: str,
        duration: float,
        has_error: bool
    ) -> None:
        """
        Registrar métricas de pipeline.
        
        Args:
            pipeline_name: Nombre del pipeline
            duration: Duración en segundos
            has_error: Si hubo error
        """
        if not pipeline_name:
            raise ValueError("Pipeline name cannot be empty")
        
        if duration < 0:
            raise ValueError("Duration cannot be negative")
        
        with self._lock:
            existing = self._pipeline_metrics.get(pipeline_name)
            new_metrics = _create_pipeline_metrics(
                pipeline_name, duration, has_error, existing
            )
            self._pipeline_metrics[pipeline_name] = new_metrics
    
    def get_pipeline_metrics(
        self,
        pipeline_name: str
    ) -> Optional[PipelineMetrics]:
        """
        Obtener métricas de pipeline.
        
        Args:
            pipeline_name: Nombre del pipeline
            
        Returns:
            Métricas o None
        """
        with self._lock:
            return self._pipeline_metrics.get(pipeline_name)
    
    def get_stage_metrics(
        self,
        pipeline_name: str,
        stage_name: str
    ) -> Optional[StageMetrics]:
        """
        Obtener métricas de etapa.
        
        Args:
            pipeline_name: Nombre del pipeline
            stage_name: Nombre de la etapa
            
        Returns:
            Métricas o None
        """
        pipeline_metrics = self.get_pipeline_metrics(pipeline_name)
        if pipeline_metrics:
            return pipeline_metrics.stage_metrics.get(stage_name)
        return None
    
    def get_all_metrics(self) -> Dict[str, PipelineMetrics]:
        """
        Obtener todas las métricas.
        
        Returns:
            Diccionario de métricas (copia)
        """
        with self._lock:
            return dict(self._pipeline_metrics)
    
    def reset_metrics(self, pipeline_name: Optional[str] = None) -> None:
        """
        Resetear métricas.
        
        Args:
            pipeline_name: Nombre del pipeline (None para todos)
        """
        with self._lock:
            if pipeline_name:
                if pipeline_name in self._pipeline_metrics:
                    del self._pipeline_metrics[pipeline_name]
            else:
                self._pipeline_metrics.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas.
        
        Returns:
            Resumen en formato diccionario
        """
        with self._lock:
            summary = {
                "total_pipelines": len(self._pipeline_metrics),
                "pipelines": {}
            }
            
            for name, metrics in self._pipeline_metrics.items():
                summary["pipelines"][name] = {
                    "execution_count": metrics.execution_count,
                    "average_duration": metrics.average_duration,
                    "success_rate": metrics.success_rate,
                    "total_stages": len(metrics.stage_metrics)
                }
            
            return summary
