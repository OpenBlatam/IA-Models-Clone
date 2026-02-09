"""
Pipeline Executors
==================

Diferentes estrategias de ejecución para pipelines.
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, TypeVar
from abc import ABC, abstractmethod

from .stages import PipelineStage, AsyncPipelineStage
from .context import PipelineContext

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineExecutor(ABC):
    """
    Interfaz base para ejecutores de pipeline.
    """
    
    @abstractmethod
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Ejecutar etapas.
        
        Args:
            stages: Lista de etapas
            data: Datos a procesar
            context: Contexto
            
        Returns:
            Datos procesados
        """
        pass


class SequentialExecutor(PipelineExecutor):
    """
    Ejecutor secuencial (por defecto).
    """
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar etapas secuencialmente."""
        current_data = data
        for stage in stages:
            logger.debug(f"Ejecutando etapa: {stage.get_name()}")
            current_data = stage.process(current_data, context)
        return current_data


class ParallelExecutor(PipelineExecutor):
    """
    Ejecutor paralelo para procesamiento concurrente.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        merge_strategy: Optional[Callable[[List[T]], T]] = None
    ):
        """
        Inicializar ejecutor paralelo.
        
        Args:
            max_workers: Número máximo de workers
            merge_strategy: Estrategia para fusionar resultados
        """
        self.max_workers = max_workers
        self.merge_strategy = merge_strategy or (lambda results: results[0] if results else None)
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar etapas en paralelo."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(stage.process, data, context)
                for stage in stages
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error en etapa paralela: {e}")
                    raise
        
        return self.merge_strategy(results)


class ConditionalExecutor(PipelineExecutor):
    """
    Ejecutor condicional que ejecuta diferentes ramas.
    """
    
    def __init__(
        self,
        condition: Callable[[T, Dict[str, Any]], bool],
        true_stages: List[PipelineStage],
        false_stages: Optional[List[PipelineStage]] = None
    ):
        """
        Inicializar ejecutor condicional.
        
        Args:
            condition: Función de condición
            true_stages: Etapas si condición es verdadera
            false_stages: Etapas si condición es falsa
        """
        self.condition = condition
        self.true_stages = true_stages
        self.false_stages = false_stages or []
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar según condición."""
        context = context or {}
        
        if self.condition(data, context):
            executor = SequentialExecutor()
            return executor.execute(self.true_stages, data, context)
        else:
            if self.false_stages:
                executor = SequentialExecutor()
                return executor.execute(self.false_stages, data, context)
            return data


class AsyncExecutor(PipelineExecutor):
    """
    Ejecutor asíncrono para etapas async.
    """
    
    async def execute_async(
        self,
        stages: List[AsyncPipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar etapas asíncronas."""
        current_data = data
        for stage in stages:
            logger.debug(f"Ejecutando etapa async: {stage.get_name()}")
            current_data = await stage.process_async(current_data, context)
        return current_data
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar (wrapper síncrono)."""
        async_stages = [
            stage for stage in stages
            if isinstance(stage, AsyncPipelineStage)
        ]
        
        if async_stages:
            return asyncio.run(self.execute_async(async_stages, data, context))
        else:
            executor = SequentialExecutor()
            return executor.execute(stages, data, context)


class BatchExecutor(PipelineExecutor):
    """
    Ejecutor para procesamiento por lotes.
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Inicializar ejecutor por lotes.
        
        Args:
            batch_size: Tamaño del lote
        """
        self.batch_size = batch_size
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar en lotes."""
        # Si data es una lista, procesar por lotes
        if isinstance(data, list):
            results = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                executor = SequentialExecutor()
                batch_result = executor.execute(stages, batch, context)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            return results
        else:
            # Si no es lista, ejecutar normalmente
            executor = SequentialExecutor()
            return executor.execute(stages, data, context)


class StreamExecutor(PipelineExecutor):
    """
    Ejecutor para procesamiento de streams.
    """
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar en modo stream."""
        # Si data es un generador o iterable, procesar elemento por elemento
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            executor = SequentialExecutor()
            return (executor.execute(stages, item, context) for item in data)
        else:
            executor = SequentialExecutor()
            return executor.execute(stages, data, context)


class PipelineCompositionExecutor(PipelineExecutor):
    """
    Ejecutor que permite composición de pipelines.
    """
    
    def __init__(self, pipelines: List['Pipeline']):
        """
        Inicializar ejecutor de composición.
        
        Args:
            pipelines: Lista de pipelines a ejecutar
        """
        from .pipeline import Pipeline
        self.pipelines = pipelines
    
    def execute(
        self,
        stages: List[PipelineStage],
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> T:
        """Ejecutar pipelines compuestos."""
        current_data = data
        
        # Primero ejecutar stages locales
        if stages:
            executor = SequentialExecutor()
            current_data = executor.execute(stages, current_data, context)
        
        # Luego ejecutar pipelines compuestos
        for pipeline in self.pipelines:
            current_data = pipeline.process(current_data, context)
        
        return current_data

