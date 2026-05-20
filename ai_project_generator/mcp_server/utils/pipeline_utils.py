"""
Pipeline Utilities - Utilidades de pipelines de transformación
===============================================================

Sistema de pipelines para transformación de datos con etapas
composables y manejo de errores.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage:
    """Etapa de pipeline."""
    
    def __init__(
        self,
        name: str,
        func: Callable[[Any], Any],
        error_handler: Optional[Callable[[Exception, Any], Any]] = None,
        skip_on_error: bool = False
    ):
        """
        Inicializar etapa.
        
        Args:
            name: Nombre de la etapa
            func: Función de transformación
            error_handler: Handler de errores (opcional)
            skip_on_error: Si True, continúa en caso de error
        """
        self.name = name
        self.func = func
        self.error_handler = error_handler
        self.skip_on_error = skip_on_error
    
    def execute(self, data: Any) -> Any:
        """
        Ejecutar etapa.
        
        Args:
            data: Datos de entrada
        
        Returns:
            Datos transformados
        
        Raises:
            Exception: Si hay error y skip_on_error es False
        """
        try:
            return self.func(data)
        except Exception as e:
            logger.error(f"Error in pipeline stage '{self.name}': {e}", exc_info=True)
            
            if self.error_handler:
                return self.error_handler(e, data)
            
            if self.skip_on_error:
                logger.warning(f"Skipping stage '{self.name}' due to error")
                return data
            
            raise


class Pipeline:
    """
    Pipeline de transformación de datos.
    
    Permite encadenar múltiples transformaciones de forma secuencial.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Inicializar pipeline.
        
        Args:
            name: Nombre del pipeline (opcional)
        """
        self.name = name or "default"
        self._stages: List[PipelineStage] = []
    
    def add_stage(
        self,
        name: str,
        func: Callable[[Any], Any],
        error_handler: Optional[Callable[[Exception, Any], Any]] = None,
        skip_on_error: bool = False
    ) -> 'Pipeline':
        """
        Agregar etapa al pipeline.
        
        Args:
            name: Nombre de la etapa
            func: Función de transformación
            error_handler: Handler de errores (opcional)
            skip_on_error: Si True, continúa en caso de error
        
        Returns:
            Self para chaining
        
        Example:
            pipeline = Pipeline()
            pipeline.add_stage("validate", validate_data)
            pipeline.add_stage("transform", transform_data)
        """
        stage = PipelineStage(name, func, error_handler, skip_on_error)
        self._stages.append(stage)
        return self
    
    def execute(self, data: Any) -> Any:
        """
        Ejecutar pipeline completo.
        
        Args:
            data: Datos de entrada
        
        Returns:
            Datos transformados
        
        Example:
            result = pipeline.execute(input_data)
        """
        current_data = data
        
        for stage in self._stages:
            current_data = stage.execute(current_data)
        
        return current_data
    
    def __call__(self, data: Any) -> Any:
        """Permite usar pipeline como función."""
        return self.execute(data)


class ParallelPipeline:
    """
    Pipeline paralelo para procesar múltiples items.
    
    Ejecuta el mismo pipeline en múltiples items en paralelo.
    """
    
    def __init__(self, pipeline: Pipeline, max_workers: int = 4):
        """
        Inicializar pipeline paralelo.
        
        Args:
            pipeline: Pipeline a ejecutar
            max_workers: Número máximo de workers
        """
        self.pipeline = pipeline
        self.max_workers = max_workers
    
    def execute(self, items: List[Any]) -> List[Any]:
        """
        Ejecutar pipeline en múltiples items.
        
        Args:
            items: Lista de items a procesar
        
        Returns:
            Lista de items transformados
        """
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.pipeline.execute, items))
        
        return results


class ConditionalPipeline:
    """
    Pipeline condicional.
    
    Ejecuta diferentes pipelines según condición.
    """
    
    def __init__(self):
        """Inicializar pipeline condicional."""
        self._branches: List[tuple[Callable[[Any], bool], Pipeline]] = []
        self._default: Optional[Pipeline] = None
    
    def add_branch(
        self,
        condition: Callable[[Any], bool],
        pipeline: Pipeline
    ) -> 'ConditionalPipeline':
        """
        Agregar rama condicional.
        
        Args:
            condition: Función de condición
            pipeline: Pipeline a ejecutar si condición es True
        
        Returns:
            Self para chaining
        """
        self._branches.append((condition, pipeline))
        return self
    
    def set_default(self, pipeline: Pipeline) -> 'ConditionalPipeline':
        """
        Establecer pipeline por defecto.
        
        Args:
            pipeline: Pipeline a ejecutar si ninguna condición se cumple
        
        Returns:
            Self para chaining
        """
        self._default = pipeline
        return self
    
    def execute(self, data: Any) -> Any:
        """
        Ejecutar pipeline condicional.
        
        Args:
            data: Datos de entrada
        
        Returns:
            Datos transformados
        """
        for condition, pipeline in self._branches:
            if condition(data):
                return pipeline.execute(data)
        
        if self._default:
            return self._default.execute(data)
        
        return data


# Funciones helper comunes

def map_stage(name: str, mapper: Callable[[Any], Any]) -> PipelineStage:
    """
    Crear etapa de mapeo.
    
    Args:
        name: Nombre de la etapa
        mapper: Función de mapeo
    
    Returns:
        PipelineStage
    """
    return PipelineStage(name, mapper)


def filter_stage(name: str, predicate: Callable[[Any], bool]) -> PipelineStage:
    """
    Crear etapa de filtrado.
    
    Args:
        name: Nombre de la etapa
        predicate: Función de predicado
    
    Returns:
        PipelineStage
    """
    def filter_func(data: Any) -> Any:
        if isinstance(data, list):
            return [item for item in data if predicate(item)]
        return data if predicate(data) else None
    
    return PipelineStage(name, filter_func)


def validate_stage(name: str, validator: Callable[[Any], bool]) -> PipelineStage:
    """
    Crear etapa de validación.
    
    Args:
        name: Nombre de la etapa
        validator: Función de validación
    
    Returns:
        PipelineStage
    """
    def validate_func(data: Any) -> Any:
        if not validator(data):
            raise ValueError(f"Validation failed in stage '{name}'")
        return data
    
    return PipelineStage(name, validate_func)


__all__ = [
    "PipelineStage",
    "Pipeline",
    "ParallelPipeline",
    "ConditionalPipeline",
    "map_stage",
    "filter_stage",
    "validate_stage",
]

