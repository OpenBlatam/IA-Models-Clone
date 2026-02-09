"""
Pipeline Decorators
===================

Decoradores para crear etapas de pipeline fácilmente.
"""

import functools
import logging
from typing import Dict, Any, Optional, Callable, TypeVar

from .stages import PipelineStage, FunctionStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


def pipeline_stage(
    name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Decorador para convertir función en etapa de pipeline.
    
    Args:
        name: Nombre de la etapa (opcional)
        description: Descripción (opcional)
    
    Returns:
        Función decorada
    """
    def decorator(func: Callable[[T, Optional[Dict[str, Any]]], T]) -> FunctionStage:
        return FunctionStage(func, name, description)
    return decorator


def async_pipeline_stage(
    name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Decorador para función async (crea wrapper síncrono).
    
    Args:
        name: Nombre de la etapa (opcional)
        description: Descripción (opcional)
    
    Returns:
        Función decorada
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data: T, context: Optional[Dict[str, Any]] = None) -> T:
            import asyncio
            return asyncio.run(func(data, context))
        
        return FunctionStage(wrapper, name or func.__name__, description or func.__doc__)
    return decorator


def validate_stage(
    input_validator: Optional[Callable[[T, Optional[Dict[str, Any]]], bool]] = None,
    output_validator: Optional[Callable[[T, Optional[Dict[str, Any]]], bool]] = None
):
    """
    Decorador para agregar validación a una etapa.
    
    Args:
        input_validator: Validador de entrada
        output_validator: Validador de salida
    
    Returns:
        Función decorada
    """
    def decorator(stage: PipelineStage) -> PipelineStage:
        original_process = stage.process
        
        @functools.wraps(original_process)
        def validated_process(
            data: T,
            context: Optional[Dict[str, Any]] = None
        ) -> T:
            # Validar entrada
            if input_validator and not input_validator(data, context):
                raise ValueError(f"Validación de entrada falló para: {stage.get_name()}")
            
            # Procesar
            result = original_process(data, context)
            
            # Validar salida
            if output_validator and not output_validator(result, context):
                raise ValueError(f"Validación de salida falló para: {stage.get_name()}")
            
            return result
        
        stage.process = validated_process
        return stage
    
    return decorator


def cache_stage(cache: Optional[Dict[str, Any]] = None):
    """
    Decorador para agregar caché a una etapa.
    
    Args:
        cache: Diccionario de caché
    
    Returns:
        Función decorada
    """
    if cache is None:
        cache = {}
    
    def decorator(stage: PipelineStage) -> PipelineStage:
        original_process = stage.process
        
        @functools.wraps(original_process)
        def cached_process(
            data: T,
            context: Optional[Dict[str, Any]] = None
        ) -> T:
            import hashlib
            import pickle
            
            # Generar clave de caché
            cache_key = hashlib.md5(
                pickle.dumps((stage.get_name(), data))
            ).hexdigest()
            
            # Verificar caché
            if cache_key in cache:
                logger.debug(f"Cache hit para: {stage.get_name()}")
                return cache[cache_key]
            
            # Procesar y guardar
            result = original_process(data, context)
            cache[cache_key] = result
            return result
        
        stage.process = cached_process
        return stage
    
    return decorator


def retry_stage(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_exceptions: tuple = (Exception,)
):
    """
    Decorador para agregar reintentos a una etapa.
    
    Args:
        max_retries: Número máximo de reintentos
        retry_delay: Delay entre reintentos
        retry_exceptions: Excepciones que activan reintento
    
    Returns:
        Función decorada
    """
    def decorator(stage: PipelineStage) -> PipelineStage:
        original_process = stage.process
        
        @functools.wraps(original_process)
        def retry_process(
            data: T,
            context: Optional[Dict[str, Any]] = None
        ) -> T:
            import time
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return original_process(data, context)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Reintentando '{stage.get_name()}' "
                            f"(intento {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Falló '{stage.get_name()}' después de "
                            f"{max_retries + 1} intentos"
                        )
            
            raise last_exception
        
        stage.process = retry_process
        return stage
    
    return decorator


def metrics_stage(metrics_collector: Optional[Any] = None):
    """
    Decorador para agregar métricas a una etapa.
    
    Args:
        metrics_collector: Colector de métricas
    
    Returns:
        Función decorada
    """
    def decorator(stage: PipelineStage) -> PipelineStage:
        original_process = stage.process
        
        @functools.wraps(original_process)
        def metrics_process(
            data: T,
            context: Optional[Dict[str, Any]] = None
        ) -> T:
            import time
            
            start_time = time.time()
            error = None
            result = None
            
            try:
                result = original_process(data, context)
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.time() - start_time
                if metrics_collector:
                    metrics_collector.record_stage_metrics(
                        stage.get_name(),
                        duration,
                        error is not None
                    )
            
            return result
        
        stage.process = metrics_process
        return stage
    
    return decorator

