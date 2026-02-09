"""
Common Utilities
================
Utilidades comunes para todos los módulos
"""

import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable
from functools import wraps


def generate_id(prefix: str = "") -> str:
    """Generar ID único"""
    timestamp = int(time.time() * 1000)
    random_part = uuid.uuid4().hex[:8]
    return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"


def hash_string(text: str) -> str:
    """Hash de string"""
    return hashlib.sha256(text.encode()).hexdigest()


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para retry automático
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos
        backoff: Factor de backoff exponencial
        exceptions: Excepciones que activan retry
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def timeit(func: Callable):
    """Decorator para medir tiempo de ejecución"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if hasattr(wrapper, 'timings'):
            wrapper.timings.append(elapsed)
        else:
            wrapper.timings = [elapsed]
        return result
    return wrapper


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionar múltiples diccionarios"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Fusión profunda de diccionarios"""
    result = base.copy()
    
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtener valor de diccionario anidado de forma segura
    
    Args:
        data: Diccionario
        path: Ruta en formato 'key.subkey.subsubkey'
        default: Valor por defecto
    """
    keys = path.split('.')
    value = data
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def safe_set(data: Dict[str, Any], path: str, value: Any):
    """
    Establecer valor en diccionario anidado de forma segura
    
    Args:
        data: Diccionario
        path: Ruta en formato 'key.subkey.subsubkey'
        value: Valor a establecer
    """
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Dividir lista en chunks"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Aplanar diccionario anidado"""
    result = {}
    
    def _flatten(obj: Any, prefix: str = ''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj
    
    _flatten(data)
    return result


def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Desaplanar diccionario"""
    result = {}
    
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def format_bytes(bytes_count: int) -> str:
    """Formatear bytes en formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


class Timer:
    """Context manager para medir tiempo"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        return False
    
    def __str__(self):
        if self.elapsed is not None:
            return f"{self.name}: {format_duration(self.elapsed)}"
        return f"{self.name}: Not completed"


class RateLimiter:
    """Rate limiter simple"""
    
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
    
    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Limpiar llamadas antiguas
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.period]
            
            self.calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validar que config tenga todas las keys requeridas"""
    return all(key in config for key in required_keys)


def sanitize_filename(filename: str) -> str:
    """Sanitizar nombre de archivo"""
    import re
    # Remover caracteres no permitidos
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limitar longitud
    if len(filename) > 255:
        filename = filename[:255]
    return filename

