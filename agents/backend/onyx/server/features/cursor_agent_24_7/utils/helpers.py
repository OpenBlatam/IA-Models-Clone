"""
Helpers - Utilidades y funciones auxiliares
===========================================

Funciones auxiliares útiles para el agente con decoradores,
utilidades de formato, y funciones de seguridad.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, List, Dict, TypeVar, Union, Awaitable
from functools import wraps
from datetime import datetime

# Importar utilidades de validación si están disponibles
try:
    from ..core.validation_utils import validate_positive, validate_non_negative
except ImportError:
    # Fallback si no están disponibles
    def validate_positive(value, name="value"):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    def validate_non_negative(value, name="value"):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """
    Decorador para reintentar función en caso de error.
    
    Args:
        max_attempts: Número máximo de intentos (default: 3).
        delay: Delay inicial en segundos (default: 1.0).
        backoff: Factor de multiplicación para el delay (default: 2.0).
        exceptions: Tupla de excepciones a capturar (default: todas).
    
    Returns:
        Decorador que envuelve la función con lógica de reintento.
    
    Raises:
        ValueError: Si los parámetros son inválidos.
    """
    validate_positive(max_attempts, "max_attempts")
    validate_non_negative(delay, "delay")
    validate_positive(backoff, "backoff")
    
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for "
                            f"{func.__name__}: {e}. Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            if last_exception:
                raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for "
                            f"{func.__name__}: {e}. Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            if last_exception:
                raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorador para timeout en función async.
    
    Args:
        seconds: Timeout en segundos.
    
    Returns:
        Decorador que envuelve la función con timeout.
    
    Raises:
        ValueError: Si seconds es inválido.
    """
    validate_positive(seconds, "seconds")
    
    def decorator(func: F) -> F:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("timeout decorator only works with async functions")
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        
        return wrapper  # type: ignore
    
    return decorator


def measure_time(func: F) -> F:
    """
    Decorador para medir tiempo de ejecución.
    
    Returns:
        Decorador que mide y registra el tiempo de ejecución.
    """
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore


def format_bytes(bytes_size: int) -> str:
    """
    Formatear bytes en formato legible (B, KB, MB, GB, TB, PB).
    
    Args:
        bytes_size: Tamaño en bytes.
    
    Returns:
        String formateado con unidad apropiada.
    
    Raises:
        ValueError: Si bytes_size es negativo.
    """
    validate_non_negative(bytes_size, "bytes_size")
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_size)
    
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Duración en segundos.
    
    Returns:
        String formateado (s, m, o h).
    
    Raises:
        ValueError: Si seconds es negativo.
    """
    validate_non_negative(seconds, "seconds")
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def truncate_string(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    Truncar string a longitud máxima.
    
    Args:
        text: Texto a truncar.
        max_length: Longitud máxima (default: 100).
        suffix: Sufijo a agregar si se trunca (default: "...").
    
    Returns:
        Texto truncado si excede max_length, texto original en caso contrario.
    
    Raises:
        ValueError: Si max_length es inválido.
    """
    validate_non_negative(max_length, "max_length")
    
    if not isinstance(text, str):
        return str(text)[:max_length]
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def safe_eval(
    expression: str,
    allowed_names: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Evaluar expresión matemática de forma segura.
    
    WARNING: Esta función es limitada y solo evalúa expresiones matemáticas
    básicas. No usar para código Python arbitrario.
    
    Args:
        expression: Expresión matemática a evaluar.
        allowed_names: Diccionario de nombres permitidos (opcional).
    
    Returns:
        Resultado de la evaluación o None si falla.
    
    Raises:
        TypeError: Si expression no es un string.
    """
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    
    if not expression.strip():
        return None
    
    allowed_names = allowed_names or {}
    
    # Solo permitir caracteres matemáticos seguros
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression.replace(" ", "")):
        logger.warning(f"Unsafe characters detected in expression: {expression}")
        return None
    
    try:
        # Usar eval con contexto restringido
        result = eval(
            expression,
            {"__builtins__": {}},
            allowed_names
        )
        return result
    except Exception as e:
        logger.debug(f"Safe eval failed for '{expression}': {e}")
        return None


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Dividir lista en chunks de tamaño fijo.
    
    Args:
        items: Lista a dividir.
        chunk_size: Tamaño de cada chunk.
    
    Returns:
        Lista de chunks.
    
    Raises:
        ValueError: Si chunk_size es inválido.
    """
    validate_positive(chunk_size, "chunk_size")
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionar múltiples diccionarios.
    
    Los diccionarios posteriores sobrescriben los valores de los anteriores.
    
    Args:
        *dicts: Diccionarios a fusionar.
    
    Returns:
        Diccionario fusionado.
    """
    result: Dict[str, Any] = {}
    for d in dicts:
        if not isinstance(d, dict):
            raise TypeError("All arguments must be dictionaries")
        result.update(d)
    return result


def get_env_var(
    key: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Obtener variable de entorno de forma segura.
    
    Args:
        key: Nombre de la variable de entorno.
        default: Valor por defecto si no existe (opcional).
        required: Si es True, lanza excepción si no existe (default: False).
    
    Returns:
        Valor de la variable de entorno o default.
    
    Raises:
        ValueError: Si required=True y la variable no existe.
    """
    import os
    
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value


def create_timestamp() -> str:
    """
    Crear timestamp legible en formato YYYYMMDD_HHMMSS.
    
    Returns:
        Timestamp como string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_duration(duration_str: str) -> float:
    """
    Parsear duración en formato legible a segundos.
    
    Formatos soportados: '10s', '5m', '2h', '1d', o número en segundos.
    
    Args:
        duration_str: String con duración (ej: "10s", "5m", "2h").
    
    Returns:
        Duración en segundos.
    
    Raises:
        ValueError: Si el formato es inválido.
    """
    if not isinstance(duration_str, str):
        raise TypeError("duration_str must be a string")
    
    duration_str = duration_str.strip().lower()
    
    if not duration_str:
        raise ValueError("duration_str cannot be empty")
    
    try:
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        elif duration_str.endswith('m'):
            return float(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return float(duration_str[:-1]) * 3600
        elif duration_str.endswith('d'):
            return float(duration_str[:-1]) * 86400
        else:
            return float(duration_str)
    except ValueError as e:
        raise ValueError(f"Invalid duration format: {duration_str}") from e


async def run_with_progress(
    func: Callable[..., Union[Any, Awaitable[Any]]],
    total_steps: int = 100,
    update_callback: Optional[Callable[[int], None]] = None
) -> Any:
    """
    Ejecutar función con barra de progreso (si tqdm está disponible).
    
    Args:
        func: Función a ejecutar (puede recibir un objeto de progreso).
        total_steps: Número total de pasos (default: 100).
        update_callback: Callback para actualizar progreso (opcional).
    
    Returns:
        Resultado de la función.
    
    Raises:
        ValueError: Si total_steps es inválido.
    """
    validate_positive(total_steps, "total_steps")
    
    try:
        from tqdm import tqdm
        
        with tqdm(total=total_steps) as pbar:
            if asyncio.iscoroutinefunction(func):
                result = await func(pbar)
            else:
                result = func(pbar)
            pbar.update(total_steps - pbar.n)
            return result
    except ImportError:
        # tqdm no disponible, ejecutar sin progreso
        logger.debug("tqdm not available, running without progress bar")
        if asyncio.iscoroutinefunction(func):
            return await func(None)
        else:
            return func(None)
