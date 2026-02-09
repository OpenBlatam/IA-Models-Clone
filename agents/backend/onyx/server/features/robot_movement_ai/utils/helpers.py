"""Helpers - Funciones helper"""

from typing import Any, Dict, List, Optional, TypeVar, Callable, Iterator
from functools import wraps
import time
import hashlib
import json

T = TypeVar('T')


class Helper:
    """Clase con funciones helper"""
    
    @staticmethod
    def safe_get(data: dict, key: str, default: Any = None) -> Any:
        """Obtiene un valor de forma segura"""
        return data.get(key, default)
    
    @staticmethod
    def chunk_list(lst: List[T], size: int) -> List[List[T]]:
        """Divide una lista en chunks"""
        return [lst[i:i + size] for i in range(0, len(lst), size)]
    
    @staticmethod
    def flatten(nested_list: List[List[T]]) -> List[T]:
        """Aplana una lista anidada"""
        return [item for sublist in nested_list for item in sublist]
    
    @staticmethod
    def deep_get(data: Dict[str, Any], path: str, default: Any = None, separator: str = '.') -> Any:
        """
        Obtiene un valor anidado de un diccionario usando path.
        
        Args:
            data: Diccionario
            path: Path separado por puntos (ej: "user.profile.name")
            default: Valor por defecto
            separator: Separador del path
        
        Returns:
            Valor encontrado o default
        """
        keys = path.split(separator)
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    @staticmethod
    def deep_set(data: Dict[str, Any], path: str, value: Any, separator: str = '.'):
        """
        Establece un valor anidado en un diccionario usando path.
        
        Args:
            data: Diccionario
            path: Path separado por puntos
            value: Valor a establecer
            separator: Separador del path
        """
        keys = path.split(separator)
        for key in keys[:-1]:
            data = data.setdefault(key, {})
        data[keys[-1]] = value
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """
        Genera hash de un string.
        
        Args:
            text: Texto a hashear
            algorithm: Algoritmo de hash (md5, sha1, sha256)
        
        Returns:
            Hash hexadecimal
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any], algorithm: str = 'sha256') -> str:
        """
        Genera hash de un diccionario.
        
        Args:
            data: Diccionario
            algorithm: Algoritmo de hash
        
        Returns:
            Hash hexadecimal
        """
        json_str = json.dumps(data, sort_keys=True)
        return Helper.hash_string(json_str, algorithm)
    
    @staticmethod
    def retry_on_exception(max_attempts: int = 3, delay: float = 1.0):
        """
        Decorador para reintentar en caso de excepción.
        
        Args:
            max_attempts: Número máximo de intentos
            delay: Delay entre intentos en segundos
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            time.sleep(delay)
                        else:
                            raise
                if last_exception:
                    raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def batch_process(items: List[T], batch_size: int, processor: Callable[[List[T]], Any]) -> List[Any]:
        """
        Procesa items en batches.
        
        Args:
            items: Lista de items
            batch_size: Tamaño del batch
            processor: Función que procesa un batch
        
        Returns:
            Lista de resultados
        """
        results = []
        for batch in Helper.chunk_list(items, batch_size):
            results.append(processor(batch))
        return results
    
    @staticmethod
    def remove_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remueve valores None de un diccionario.
        
        Args:
            data: Diccionario
        
        Returns:
            Diccionario sin valores None
        """
        return {k: v for k, v in data.items() if v is not None}
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusiona múltiples diccionarios.
        
        Args:
            *dicts: Diccionarios a fusionar
        
        Returns:
            Diccionario fusionado
        """
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    @staticmethod
    def get_nested_keys(data: Dict[str, Any], prefix: str = '') -> List[str]:
        """
        Obtiene todas las keys anidadas de un diccionario.
        
        Args:
            data: Diccionario
            prefix: Prefijo para las keys
        
        Returns:
            Lista de keys
        """
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(Helper.get_nested_keys(value, full_key))
            else:
                keys.append(full_key)
        return keys

