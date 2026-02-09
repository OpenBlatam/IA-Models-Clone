"""
Pipeline Middleware
==================

Middleware común para pipelines: timing, logging, caching, etc.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone
from functools import wraps

from .base import PipelineContext

logger = logging.getLogger(__name__)


def timing_middleware(data: Any, context: PipelineContext) -> Any:
    """
    Middleware para medir tiempo de ejecución.
    
    Args:
        data: Datos del pipeline
        context: Contexto del pipeline
        
    Returns:
        Datos sin modificar
    """
    if not hasattr(context, 'metadata'):
        context.metadata = {}
    
    if "timing" not in context.metadata:
        context.metadata["timing"] = {}
    
    context.metadata["timing"]["start"] = datetime.now(timezone.utc).isoformat()
    return data


def create_logging_middleware(log_level: int = logging.INFO) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware de logging.
    
    Args:
        log_level: Nivel de logging
        
    Returns:
        Función middleware
    """
    def middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware de logging.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos sin modificar
        """
        pipeline_id = getattr(context, 'pipeline_id', 'unknown')
        data_type = type(data).__name__
        
        logger.log(
            log_level,
            f"Pipeline '{pipeline_id}' processing data: {data_type}"
        )
        return data
    
    return middleware


def create_caching_middleware(
    cache: Dict[str, Any],
    cache_key_func: Optional[Callable[[Any, PipelineContext], str]] = None
) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware de caching.
    
    Args:
        cache: Diccionario de caché
        cache_key_func: Función para generar clave de caché
        
    Returns:
        Función middleware
    """
    if not isinstance(cache, dict):
        raise TypeError("cache must be a dictionary")
    
    def middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware de caching.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos desde caché o sin modificar
        """
        if cache_key_func:
            cache_key = cache_key_func(data, context)
        else:
            pipeline_id = getattr(context, 'pipeline_id', 'unknown')
            cache_key = f"{pipeline_id}_{hash(str(data))}"
        
        if cache_key in cache:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cache[cache_key]
        
        logger.debug(f"Cache miss for key: {cache_key}")
        return data
    
    return middleware


def create_error_handling_middleware(
    on_error: Optional[Callable[[Exception, Any, PipelineContext], Any]] = None
) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware de manejo de errores.
    
    Args:
        on_error: Función para manejar errores
        
    Returns:
        Función middleware
    """
    def middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware de manejo de errores.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos procesados
            
        Raises:
            Exception: Si hay error y no hay handler
        """
        try:
            return data
        except Exception as e:
            logger.error(f"Error in middleware: {e}", exc_info=True)
            if on_error:
                return on_error(e, data, context)
            raise
    
    return middleware


def create_validation_middleware(
    validator: Callable[[Any], bool]
) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware de validación.
    
    Args:
        validator: Función validadora
        
    Returns:
        Función middleware
        
    Raises:
        TypeError: Si validator no es callable
    """
    if not callable(validator):
        raise TypeError("validator must be callable")
    
    def middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware de validación.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos validados
            
        Raises:
            ValueError: Si la validación falla
        """
        if not validator(data):
            pipeline_id = getattr(context, 'pipeline_id', 'unknown')
            raise ValueError(
                f"Validation failed in middleware for pipeline '{pipeline_id}'"
            )
        return data
    
    return middleware


def create_transformation_middleware(
    transform: Callable[[Any], Any]
) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware de transformación.
    
    Args:
        transform: Función de transformación
        
    Returns:
        Función middleware
        
    Raises:
        TypeError: Si transform no es callable
    """
    if not callable(transform):
        raise TypeError("transform must be callable")
    
    def middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware de transformación.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos transformados
        """
        return transform(data)
    
    return middleware


def create_async_middleware(
    async_func: Callable[[Any, PipelineContext], Any]
) -> Callable[[Any, PipelineContext], Any]:
    """
    Factory para crear middleware async.
    
    Args:
        async_func: Función async
        
    Returns:
        Función middleware wrapper
        
    Raises:
        TypeError: Si async_func no es callable
    """
    if not callable(async_func):
        raise TypeError("async_func must be callable")
    
    async def async_middleware(data: Any, context: PipelineContext) -> Any:
        """
        Middleware async.
        
        Args:
            data: Datos del pipeline
            context: Contexto del pipeline
            
        Returns:
            Datos procesados
        """
        return await async_func(data, context)
    
    return async_middleware
