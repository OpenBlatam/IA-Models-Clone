"""
Debug Utilities
===============

Utilidades para debugging y desarrollo.
"""

import inspect
import traceback
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


def debug_print(obj: Any, name: str = "debug") -> None:
    """
    Imprimir información detallada de objeto.
    
    Args:
        obj: Objeto a inspeccionar
        name: Nombre del objeto
    """
    print(f"\n{'='*60}")
    print(f"DEBUG: {name}")
    print(f"{'='*60}")
    print(f"Type: {type(obj)}")
    print(f"Value: {obj}")
    
    if hasattr(obj, '__dict__'):
        print(f"Attributes: {obj.__dict__}")
    
    if hasattr(obj, '__class__'):
        print(f"Class: {obj.__class__}")
        print(f"Methods: {[m for m in dir(obj) if not m.startswith('_')]}")
    
    print(f"{'='*60}\n")


def trace_function(func):
    """
    Decorador para trazar ejecución de función.
    
    Usage:
        @trace_function
        def my_function():
            ...
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n>>> Entering {func.__name__}")
        print(f"    Args: {args}")
        print(f"    Kwargs: {kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"<<< Exiting {func.__name__} (took {duration:.4f}s)")
            print(f"    Result: {result}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"!!! Error in {func.__name__} after {duration:.4f}s")
            print(f"    Error: {e}")
            print(f"    Traceback:\n{traceback.format_exc()}")
            raise
    
    return wrapper


def trace_function_async(func):
    """
    Decorador para trazar ejecución de función async.
    
    Usage:
        @trace_function_async
        async def my_async_function():
            ...
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"\n>>> Entering {func.__name__} (async)")
        print(f"    Args: {args}")
        print(f"    Kwargs: {kwargs}")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"<<< Exiting {func.__name__} (async, took {duration:.4f}s)")
            print(f"    Result: {result}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"!!! Error in {func.__name__} (async) after {duration:.4f}s")
            print(f"    Error: {e}")
            print(f"    Traceback:\n{traceback.format_exc()}")
            raise
    
    return wrapper


def get_call_stack(depth: int = 10) -> List[str]:
    """
    Obtener call stack actual.
    
    Args:
        depth: Profundidad del stack
        
    Returns:
        Lista de frames del stack
    """
    stack = []
    for frame_info in inspect.stack()[1:depth+1]:
        stack.append(
            f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}"
        )
    return stack


def log_call_stack(logger_instance: Optional[logging.Logger] = None) -> None:
    """
    Loggear call stack actual.
    
    Args:
        logger_instance: Logger a usar (default: module logger)
    """
    log = logger_instance or logger
    stack = get_call_stack()
    log.debug("Call stack:")
    for frame in stack:
        log.debug(f"  {frame}")


class DebugContext:
    """
    Context manager para debugging.
    
    Usage:
        with DebugContext("my_operation"):
            # código a debuggear
            ...
    """
    
    def __init__(self, name: str, logger_instance: Optional[logging.Logger] = None):
        """
        Inicializar contexto de debug.
        
        Args:
            name: Nombre del contexto
            logger_instance: Logger a usar
        """
        self.name = name
        self.logger = logger_instance or logger
        self.start_time = None
    
    def __enter__(self):
        """Entrar al contexto."""
        self.start_time = time.time()
        self.logger.debug(f"Entering debug context: {self.name}")
        log_call_stack(self.logger)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto."""
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.debug(f"Exiting debug context: {self.name} (took {duration:.4f}s)")
        else:
            self.logger.error(
                f"Error in debug context: {self.name} after {duration:.4f}s"
            )
            self.logger.error(f"Exception: {exc_type.__name__}: {exc_val}")
            self.logger.error(traceback.format_exception(exc_type, exc_val, exc_tb))
        return False  # No suprimir excepciones


def profile_function(func):
    """
    Decorador para profiling de función.
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    import functools
    import cProfile
    import pstats
    import io
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Generar reporte
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20
            logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper

