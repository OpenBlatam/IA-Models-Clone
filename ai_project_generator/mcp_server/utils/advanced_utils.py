"""
Advanced Utilities - Utilidades avanzadas para MCP Server
==========================================================

Funciones avanzadas para operaciones comunes, decoradores útiles,
y utilidades de optimización.
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None
):
    """
    Decorador para reintentar función en caso de fallo.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff exponencial
        exceptions: Tupla de excepciones a capturar
        on_failure: Función a llamar en cada fallo (opcional)
    
    Example:
        @retry_on_failure(max_attempts=3, delay=1.0)
        def risky_operation():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if on_failure:
                            on_failure(attempt, e, current_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


def timed_operation(operation_name: Optional[str] = None):
    """
    Decorador para medir tiempo de ejecución de una operación.
    
    Args:
        operation_name: Nombre de la operación (default: nombre de función)
    
    Example:
        @timed_operation("database_query")
        def query_database():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"{name} completed in {elapsed:.3f}s")
                return result
            except (ValueError, TypeError, AttributeError) as e:
                elapsed = time.time() - start_time
                logger.error(f"{name} failed after {elapsed:.3f}s: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{name} failed after {elapsed:.3f}s: {type(e).__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None, max_size: int = 128):
    """
    Decorador para cachear resultados de función.
    
    Args:
        ttl: Time to live en segundos (None = sin expiración)
        max_size: Tamaño máximo del cache
    
    Example:
        @cache_result(ttl=3600)
        def expensive_computation(x):
            ...
    """
    cache: Dict[str, tuple] = {}
    cache_times: Dict[str, float] = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            if cache_key in cache:
                if ttl is None or (time.time() - cache_times[cache_key]) < ttl:
                    return cache[cache_key][0]
                else:
                    del cache[cache_key]
                    del cache_times[cache_key]
            
            result = func(*args, **kwargs)
            
            if len(cache) >= max_size:
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[cache_key] = (result, time.time())
            cache_times[cache_key] = time.time()
            
            return result
        return wrapper
    return decorator


@contextmanager
def performance_context(operation_name: str):
    """
    Context manager para medir performance de un bloque de código.
    
    Args:
        operation_name: Nombre de la operación
    
    Example:
        with performance_context("data_processing"):
            process_data()
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"{operation_name} took {elapsed:.3f}s")


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    on_error: Optional[Callable[[Exception], Any]] = None,
    **kwargs
) -> Any:
    """
    Ejecutar función de forma segura con manejo de errores.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        default: Valor por defecto si falla
        on_error: Función a llamar en caso de error
        **kwargs: Argumentos nombrados
    
    Returns:
        Resultado de la función o valor por defecto
    
    Example:
        result = safe_execute(risky_function, arg1, arg2, default="error")
    """
    try:
        return func(*args, **kwargs)
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.debug(f"Error executing {func.__name__}: {type(e).__name__}: {e}")
        if on_error:
            return on_error(e)
        return default
    except Exception as e:
        logger.debug(f"Error executing {func.__name__}: {type(e).__name__}: {e}", exc_info=True)
        if on_error:
            return on_error(e)
        return default


def batch_process(
    items: List[Any],
    batch_size: int = 100,
    processor: Optional[Callable] = None
) -> List[Any]:
    """
    Procesar items en lotes.
    
    Args:
        items: Lista de items a procesar
        batch_size: Tamaño del lote
        processor: Función procesadora (opcional)
    
    Returns:
        Lista de resultados
    
    Example:
        results = batch_process(items, batch_size=50, processor=process_item)
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if processor:
            batch_results = [processor(item) for item in batch]
            results.extend(batch_results)
        else:
            results.extend(batch)
    return results


def merge_dicts(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Fusionar múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a fusionar
        deep: Si True, hace merge profundo
    
    Returns:
        Diccionario fusionado
    
    Example:
        merged = merge_dicts(dict1, dict2, dict3, deep=True)
    """
    if not dicts:
        return {}
    
    result = dicts[0].copy()
    
    for d in dicts[1:]:
        if deep:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value, deep=True)
                else:
                    result[key] = value
        else:
            result.update(d)
    
    return result


def flatten_dict(d: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """
    Aplanar diccionario anidado.
    
    Args:
        d: Diccionario a aplanar
        separator: Separador para keys anidadas
        prefix: Prefijo para keys
    
    Returns:
        Diccionario aplanado
    
    Example:
        flat = flatten_dict({'a': {'b': 1, 'c': 2}}, separator='_')
        # {'a_b': 1, 'a_c': 2}
    """
    items = []
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def group_by(items: List[Any], key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
    """
    Agrupar items por función de clave.
    
    Args:
        items: Lista de items
        key_func: Función para extraer clave
    
    Returns:
        Diccionario agrupado
    
    Example:
        grouped = group_by(users, key_func=lambda u: u.age)
    """
    grouped = defaultdict(list)
    for item in items:
        key = key_func(item)
        grouped[key].append(item)
    return dict(grouped)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Dividir lista en chunks.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño del chunk
    
    Returns:
        Lista de chunks
    
    Example:
        chunks = chunk_list(range(10), chunk_size=3)
        # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_bytes(bytes_value: int) -> str:
    """
    Formatear bytes en formato legible.
    
    Args:
        bytes_value: Valor en bytes
    
    Returns:
        String formateado
    
    Example:
        format_bytes(1024)  # "1.0 KB"
        format_bytes(1048576)  # "1.0 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Duración en segundos
    
    Returns:
        String formateado
    
    Example:
        format_duration(3661)  # "1h 1m 1s"
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def validate_not_none(value: Any, name: str = "value") -> None:
    """
    Validar que un valor no sea None.
    
    Args:
        value: Valor a validar
        name: Nombre del valor (para mensaje de error)
    
    Raises:
        ValueError: Si el valor es None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_not_empty(value: Union[str, List, Dict], name: str = "value") -> None:
    """
    Validar que un valor no esté vacío.
    
    Args:
        value: Valor a validar
        name: Nombre del valor (para mensaje de error)
    
    Raises:
        ValueError: Si el valor está vacío
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")


class RateLimiter:
    """
    Rate limiter simple basado en tiempo.
    """
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Inicializar rate limiter.
        
        Args:
            max_calls: Número máximo de llamadas
            time_window: Ventana de tiempo en segundos
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def is_allowed(self) -> bool:
        """
        Verificar si se permite la llamada.
        
        Returns:
            True si se permite, False en caso contrario
        """
        now = time.time()
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def wait_time(self) -> float:
        """
        Obtener tiempo de espera hasta la próxima llamada permitida.
        
        Returns:
            Tiempo en segundos
        """
        if not self.calls:
            return 0.0
        
        now = time.time()
        oldest_call = min(self.calls)
        elapsed = now - oldest_call
        
        if elapsed >= self.time_window:
            return 0.0
        
        return self.time_window - elapsed


__all__ = [
    "retry_on_failure",
    "timed_operation",
    "cache_result",
    "performance_context",
    "safe_execute",
    "batch_process",
    "merge_dicts",
    "flatten_dict",
    "group_by",
    "chunk_list",
    "format_bytes",
    "format_duration",
    "validate_not_none",
    "validate_not_empty",
    "RateLimiter",
]

