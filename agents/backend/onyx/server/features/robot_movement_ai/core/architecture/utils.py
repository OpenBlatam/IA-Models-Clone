"""
Utilidades generales para Robot Movement AI v2.0
Funciones helper y utilidades comunes
"""

import hashlib
import json
import uuid
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio


def generate_id(prefix: str = "") -> str:
    """
    Generar ID único
    
    Args:
        prefix: Prefijo opcional para el ID
        
    Returns:
        ID único generado
    """
    id_str = str(uuid.uuid4())
    if prefix:
        return f"{prefix}-{id_str}"
    return id_str


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """
    Generar hash de un string
    
    Args:
        value: String a hashear
        algorithm: Algoritmo a usar (md5, sha1, sha256)
        
    Returns:
        Hash hexadecimal
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(value.encode())
    return hash_obj.hexdigest()


def serialize_json(obj: Any) -> str:
    """
    Serializar objeto a JSON con manejo de tipos especiales
    
    Args:
        obj: Objeto a serializar
        
    Returns:
        String JSON
    """
    def default_serializer(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, timedelta):
            return o.total_seconds()
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    return json.dumps(obj, default=default_serializer, indent=2)


def deserialize_json(json_str: str) -> Any:
    """
    Deserializar JSON a objeto
    
    Args:
        json_str: String JSON
        
    Returns:
        Objeto deserializado
    """
    return json.loads(json_str)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para retry de funciones
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos (segundos)
        backoff: Factor de backoff exponencial
        exceptions: Tupla de excepciones a capturar
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorator para timeout de funciones async
    
    Args:
        seconds: Segundos de timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        return wrapper
    return decorator


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Dividir lista en chunks
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
        
    Returns:
        Lista de chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Aplanar diccionario anidado
    
    Args:
        d: Diccionario a aplanar
        parent_key: Clave padre (para recursión)
        sep: Separador para claves
        
    Returns:
        Diccionario aplanado
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge profundo de diccionarios
    
    Args:
        base: Diccionario base
        update: Diccionario con actualizaciones
        
    Returns:
        Diccionario mergeado
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_get(d: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Obtener valor de diccionario anidado de forma segura
    
    Args:
        d: Diccionario
        *keys: Claves anidadas
        default: Valor por defecto si no existe
        
    Returns:
        Valor encontrado o default
    """
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default


def format_bytes(bytes_value: int) -> str:
    """
    Formatear bytes a formato legible
    
    Args:
        bytes_value: Valor en bytes
        
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Formatear duración en segundos a formato legible
    
    Args:
        seconds: Segundos
        
    Returns:
        String formateado (ej: "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.2f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m {secs:.2f}s"


class RateLimiter:
    """Rate limiter simple basado en tiempo"""
    
    def __init__(self, max_calls: int, period: float):
        """
        Inicializar rate limiter
        
        Args:
            max_calls: Número máximo de llamadas
            period: Período en segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
    
    def is_allowed(self) -> bool:
        """Verificar si se permite la llamada"""
        now = time.time()
        
        # Limpiar llamadas antiguas
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period]
        
        if len(self.calls) >= self.max_calls:
            return False
        
        self.calls.append(now)
        return True
    
    def reset(self):
        """Resetear contador"""
        self.calls.clear()


# Alias para compatibilidad
import time as _time_module
time = _time_module




