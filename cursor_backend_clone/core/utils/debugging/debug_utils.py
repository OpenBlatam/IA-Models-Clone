"""
Debug Utils - Utilidades de depuración
=======================================

Utilidades para facilitar la depuración y desarrollo.
"""

import asyncio
import logging
import traceback
import sys
import inspect
from typing import Any, Callable, Optional, Dict
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


def debug_async(
    log_args: bool = True,
    log_result: bool = False,
    log_traceback: bool = False
):
    """
    Decorador para debugging de funciones async.
    
    Args:
        log_args: Si registrar argumentos
        log_result: Si registrar resultado
        log_traceback: Si registrar traceback completo en errores
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            
            if log_args:
                logger.debug(
                    f"🔍 {func_name} called with args={args[:3] if len(args) > 3 else args}, "
                    f"kwargs={dict(list(kwargs.items())[:3]) if len(kwargs) > 3 else kwargs}"
                )
            
            try:
                result = await func(*args, **kwargs)
                
                if log_result:
                    result_str = str(result)
                    logger.debug(
                        f"✅ {func_name} returned: {result_str[:200] if len(result_str) > 200 else result_str}"
                    )
                
                return result
            except Exception as e:
                error_msg = f"❌ {func_name} raised {type(e).__name__}: {str(e)}"
                
                if log_traceback:
                    error_msg += f"\n{traceback.format_exc()}"
                
                logger.error(error_msg)
                raise
        
        return wrapper
    return decorator


def debug_sync(
    log_args: bool = True,
    log_result: bool = False,
    log_traceback: bool = False
):
    """
    Decorador para debugging de funciones síncronas.
    
    Similar a debug_async pero para funciones síncronas.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            
            if log_args:
                logger.debug(
                    f"🔍 {func_name} called with args={args[:3] if len(args) > 3 else args}, "
                    f"kwargs={dict(list(kwargs.items())[:3]) if len(kwargs) > 3 else kwargs}"
                )
            
            try:
                result = func(*args, **kwargs)
                
                if log_result:
                    result_str = str(result)
                    logger.debug(
                        f"✅ {func_name} returned: {result_str[:200] if len(result_str) > 200 else result_str}"
                    )
                
                return result
            except Exception as e:
                error_msg = f"❌ {func_name} raised {type(e).__name__}: {str(e)}"
                
                if log_traceback:
                    error_msg += f"\n{traceback.format_exc()}"
                
                logger.error(error_msg)
                raise
        
        return wrapper
    return decorator


def print_call_stack(limit: int = 10) -> None:
    """
    Imprimir stack de llamadas actual.
    
    Args:
        limit: Número máximo de frames a mostrar
    """
    stack = traceback.extract_stack(limit=limit)
    logger.debug("📚 Call stack:")
    for frame in stack[:-1]:  # Excluir esta función
        logger.debug(f"  {frame.filename}:{frame.lineno} in {frame.name}")


def get_function_info(func: Callable) -> Dict[str, Any]:
    """
    Obtener información detallada de una función.
    
    Args:
        func: Función a analizar
        
    Returns:
        Diccionario con información de la función
    """
    sig = inspect.signature(func)
    
    return {
        "name": func.__name__,
        "module": func.__module__ if hasattr(func, '__module__') else None,
        "docstring": inspect.getdoc(func),
        "parameters": {
            name: {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "kind": str(param.kind)
            }
            for name, param in sig.parameters.items()
        },
        "return_annotation": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None,
        "is_async": inspect.iscoroutinefunction(func),
        "is_generator": inspect.isgeneratorfunction(func)
    }


class DebugContext:
    """Context manager para debugging con información contextual"""
    
    def __init__(self, context_name: str, **context_data):
        self.context_name = context_name
        self.context_data = context_data
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.debug(
            f"🔍 Entering debug context: {self.context_name}",
            extra=self.context_data
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.debug(
                f"✅ Exiting debug context: {self.context_name} (duration: {duration:.3f}s)",
                extra={**self.context_data, "duration": duration}
            )
        else:
            logger.error(
                f"❌ Error in debug context: {self.context_name} (duration: {duration:.3f}s) - "
                f"{exc_type.__name__}: {str(exc_val)}",
                extra={**self.context_data, "duration": duration, "error_type": exc_type.__name__},
                exc_info=True
            )
        
        return False  # No suprimir excepciones


def log_memory_usage(label: str = "") -> None:
    """
    Registrar uso de memoria actual.
    
    Args:
        label: Etiqueta para el log
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug(
            f"💾 Memory usage {label}: "
            f"RSS={mem_info.rss / 1024 / 1024:.2f}MB, "
            f"VMS={mem_info.vms / 1024 / 1024:.2f}MB"
        )
    except ImportError:
        logger.debug(f"💾 Memory usage {label}: psutil not available")


def log_async_tasks() -> None:
    """Registrar información sobre tareas asyncio activas"""
    try:
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        logger.debug(f"🔄 Active async tasks: {len(tasks)}")
        for task in tasks[:10]:  # Mostrar solo las primeras 10
            logger.debug(f"  - {task.get_name()}: {task}")
    except RuntimeError:
        logger.debug("🔄 No event loop active")




