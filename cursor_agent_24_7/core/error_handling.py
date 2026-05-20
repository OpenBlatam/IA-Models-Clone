"""
Error Handling Utilities - Manejo centralizado de errores
=========================================================

Utilidades para manejo consistente de errores y logging en todo el sistema.
"""

import logging
import functools
import traceback
from typing import Callable, TypeVar, Optional, Any
from contextlib import contextmanager

# Usar structlog si está disponible
try:
    import structlog
    logger = structlog.get_logger(__name__)
    HAS_STRUCTLOG = True
except ImportError:
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False

T = TypeVar('T')


class ComponentInitializationError(Exception):
    """Error al inicializar un componente."""
    pass


class ConfigurationError(Exception):
    """Error en la configuración."""
    pass


class TaskExecutionError(Exception):
    """Error al ejecutar una tarea."""
    pass


def safe_import(module_name: str, fallback: Optional[Any] = None, logger_instance: Optional[Any] = None):
    """
    Importar módulo de forma segura con fallback.
    
    Args:
        module_name: Nombre del módulo a importar.
        fallback: Valor a retornar si falla la importación.
        logger_instance: Logger a usar (opcional).
    
    Returns:
        Módulo importado o fallback.
    """
    log = logger_instance or logger
    try:
        return __import__(module_name)
    except ImportError as e:
        log.debug(f"Optional module not available: {module_name} ({e})")
        return fallback


def safe_initialize(
    component_name: str,
    init_func: Callable[[], T],
    fallback: Optional[T] = None,
    logger_instance: Optional[Any] = None,
    raise_on_error: bool = False
) -> Optional[T]:
    """
    Inicializar componente de forma segura con manejo de errores.
    
    Args:
        component_name: Nombre del componente (para logging).
        init_func: Función que inicializa el componente.
        fallback: Valor a retornar si falla (default: None).
        logger_instance: Logger a usar (opcional).
        raise_on_error: Si True, lanza excepción en lugar de retornar fallback.
    
    Returns:
        Componente inicializado o fallback.
    
    Raises:
        ComponentInitializationError: Si raise_on_error=True y falla la inicialización.
    """
    log = logger_instance or logger
    try:
        component = init_func()
        log.debug(f"{component_name} initialized successfully")
        return component
    except ImportError as e:
        log.debug(f"Optional component not available: {component_name} ({e})")
        if raise_on_error:
            raise ComponentInitializationError(
                f"Failed to initialize {component_name}: {e}"
            ) from e
        return fallback
    except Exception as e:
        log.warning(f"Error initializing {component_name}: {e}")
        if raise_on_error:
            raise ComponentInitializationError(
                f"Failed to initialize {component_name}: {e}"
            ) from e
        return fallback


@contextmanager
def error_context(operation: str, logger_instance: Optional[Any] = None, reraise: bool = True):
    """
    Context manager para manejo de errores con logging.
    
    Args:
        operation: Descripción de la operación.
        logger_instance: Logger a usar (opcional).
        reraise: Si True, relanza la excepción después de loguearla.
    
    Yields:
        None
    
    Example:
        with error_context("processing task"):
            # código que puede fallar
            process_task()
    """
    log = logger_instance or logger
    try:
        yield
    except Exception as e:
        log.error(
            f"Error in {operation}: {e}",
            exc_info=True,
            extra={"operation": operation, "error": str(e)}
        )
        if reraise:
            raise


def handle_errors(
    operation: str = "operation",
    default_return: Optional[Any] = None,
    logger_instance: Optional[Any] = None,
    reraise: bool = False
):
    """
    Decorador para manejo de errores consistente.
    
    Args:
        operation: Descripción de la operación (para logging).
        default_return: Valor a retornar si falla (si reraise=False).
        logger_instance: Logger a usar (opcional).
        reraise: Si True, relanza la excepción.
    
    Returns:
        Decorador de función.
    
    Example:
        @handle_errors("processing task", default_return=None)
        def process_task():
            # código que puede fallar
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger_instance or logger
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(
                    f"Error in {operation} ({func.__name__}): {e}",
                    exc_info=True,
                    extra={
                        "operation": operation,
                        "function": func.__name__,
                        "error": str(e)
                    }
                )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


async def safe_async_call(
    func: Callable,
    *args,
    operation: str = "async operation",
    default_return: Optional[Any] = None,
    logger_instance: Optional[Any] = None,
    reraise: bool = False,
    **kwargs
) -> Optional[Any]:
    """
    Ejecutar función async de forma segura con manejo de errores.
    
    Args:
        func: Función async a ejecutar.
        *args: Argumentos posicionales.
        operation: Descripción de la operación (para logging).
        default_return: Valor a retornar si falla (si reraise=False).
        logger_instance: Logger a usar (opcional).
        reraise: Si True, relanza la excepción.
        **kwargs: Argumentos con nombre.
    
    Returns:
        Resultado de la función o default_return.
    """
    log = logger_instance or logger
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        log.error(
            f"Error in {operation}: {e}",
            exc_info=True,
            extra={"operation": operation, "error": str(e)}
        )
        if reraise:
            raise
        return default_return


def log_component_status(
    component_name: str,
    is_available: bool,
    logger_instance: Optional[Any] = None
) -> None:
    """
    Loggear estado de un componente.
    
    Args:
        component_name: Nombre del componente.
        is_available: Si el componente está disponible.
        logger_instance: Logger a usar (opcional).
    """
    log = logger_instance or logger
    status = "available" if is_available else "unavailable"
    log.debug(f"Component {component_name} is {status}")


def format_error_summary(error: Exception) -> str:
    """
    Formatear resumen de error para logging.
    
    Args:
        error: Excepción a formatear.
    
    Returns:
        String con resumen del error.
    """
    error_type = type(error).__name__
    error_msg = str(error)
    return f"{error_type}: {error_msg}"


def get_error_traceback(error: Exception) -> str:
    """
    Obtener traceback completo de un error.
    
    Args:
        error: Excepción.
    
    Returns:
        String con traceback completo.
    """
    return ''.join(traceback.format_exception(type(error), error, error.__traceback__))




