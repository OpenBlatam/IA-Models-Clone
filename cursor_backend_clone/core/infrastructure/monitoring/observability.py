"""
Observability - Decoradores y utilidades para observabilidad
===========================================================

Decoradores y utilidades para agregar métricas, logging estructurado,
y tracing a métodos críticos.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Dict, TypeVar
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


def observe_async(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    track_metrics: bool = True
):
    """
    Decorador para agregar observabilidad a funciones async.
    
    Registra:
    - Tiempo de ejecución
    - Éxitos/fallos
    - Argumentos (opcional)
    - Resultados (opcional)
    
    Args:
        operation_name: Nombre de la operación (default: nombre de la función)
        log_args: Si registrar argumentos
        log_result: Si registrar resultado
        track_metrics: Si trackear métricas (requiere Metrics disponible)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_datetime = datetime.now()
            
            # Log inicio
            log_data = {
                "operation": op_name,
                "status": "started",
                "timestamp": start_datetime.isoformat()
            }
            
            if log_args:
                log_data["args"] = str(args)[:200] if args else None
                log_data["kwargs"] = {k: str(v)[:100] for k, v in list(kwargs.items())[:5]}
            
            logger.debug(f"🔍 {op_name} started", extra=log_data)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log éxito
                success_log = {
                    "operation": op_name,
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if log_result:
                    result_str = str(result)
                    success_log["result"] = result_str[:200] if len(result_str) > 200 else result_str
                
                logger.info(
                    f"✅ {op_name} completed in {execution_time:.3f}s",
                    extra=success_log
                )
                
                # Trackear métricas si está disponible
                if track_metrics:
                    try:
                        from .metrics import Metrics
                        # Intentar obtener instancia global de métricas
                        # Esto requiere que Metrics sea un singleton o se pase como parámetro
                        pass  # Implementación futura
                    except ImportError:
                        pass
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error
                error_log = {
                    "operation": op_name,
                    "status": "error",
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.error(
                    f"❌ {op_name} failed after {execution_time:.3f}s: {type(e).__name__}: {str(e)[:100]}",
                    extra=error_log,
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def observe_sync(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False
):
    """
    Decorador para agregar observabilidad a funciones síncronas.
    
    Similar a observe_async pero para funciones síncronas.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_datetime = datetime.now()
            
            log_data = {
                "operation": op_name,
                "status": "started",
                "timestamp": start_datetime.isoformat()
            }
            
            if log_args:
                log_data["args"] = str(args)[:200] if args else None
                log_data["kwargs"] = {k: str(v)[:100] for k, v in list(kwargs.items())[:5]}
            
            logger.debug(f"🔍 {op_name} started", extra=log_data)
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                success_log = {
                    "operation": op_name,
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if log_result:
                    result_str = str(result)
                    success_log["result"] = result_str[:200] if len(result_str) > 200 else result_str
                
                logger.info(
                    f"✅ {op_name} completed in {execution_time:.3f}s",
                    extra=success_log
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                error_log = {
                    "operation": op_name,
                    "status": "error",
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.error(
                    f"❌ {op_name} failed after {execution_time:.3f}s: {type(e).__name__}: {str(e)[:100]}",
                    extra=error_log,
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


@asynccontextmanager
async def observe_context(operation_name: str, **context_data):
    """
    Context manager para observar bloques de código.
    
    Example:
        async with observe_context("process_data", user_id=123):
            # código a observar
            result = await process()
    """
    start_time = time.time()
    start_datetime = datetime.now()
    
    log_data = {
        "operation": operation_name,
        "status": "started",
        "timestamp": start_datetime.isoformat(),
        **context_data
    }
    
    logger.debug(f"🔍 {operation_name} started", extra=log_data)
    
    try:
        yield
        execution_time = time.time() - start_time
        
        success_log = {
            "operation": operation_name,
            "status": "success",
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            **context_data
        }
        
        logger.info(
            f"✅ {operation_name} completed in {execution_time:.3f}s",
            extra=success_log
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        error_log = {
            "operation": operation_name,
            "status": "error",
            "execution_time": execution_time,
            "error_type": type(e).__name__,
            "error_message": str(e)[:200],
            "timestamp": datetime.now().isoformat(),
            **context_data
        }
        
        logger.error(
            f"❌ {operation_name} failed after {execution_time:.3f}s: {type(e).__name__}: {str(e)[:100]}",
            extra=error_log,
            exc_info=True
        )
        
        raise


class OperationTracker:
    """Tracker para operaciones con métricas agregadas"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.success: bool = False
        self.error: Optional[Exception] = None
        self.metadata: Dict[str, Any] = {}
    
    def start(self):
        """Iniciar tracking"""
        self.start_time = time.time()
        return self
    
    def finish(self, success: bool = True, error: Optional[Exception] = None, **metadata):
        """Finalizar tracking"""
        self.end_time = time.time()
        self.success = success
        self.error = error
        self.metadata.update(metadata)
        return self
    
    @property
    def duration(self) -> Optional[float]:
        """Duración de la operación en segundos"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para logging/métricas"""
        return {
            "operation": self.operation_name,
            "duration": self.duration,
            "success": self.success,
            "error_type": type(self.error).__name__ if self.error else None,
            "error_message": str(self.error)[:200] if self.error else None,
            "timestamp": datetime.now().isoformat(),
            **self.metadata
        }




