"""
Data Transformation Utilities
==============================
Utilidades avanzadas para transformación de datos.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import json
from functools import reduce

from .logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """Transformador de datos."""
    
    def __init__(self, data: Any):
        """
        Inicializar transformador.
        
        Args:
            data: Datos a transformar
        """
        self.data = data
    
    def map(self, func: Callable[[Any], Any]) -> "DataTransformer":
        """
        Aplicar función de mapeo.
        
        Args:
            func: Función de transformación
            
        Returns:
            Nuevo transformador
        """
        if isinstance(self.data, list):
            self.data = [func(item) for item in self.data]
        elif isinstance(self.data, dict):
            self.data = {k: func(v) for k, v in self.data.items()}
        else:
            self.data = func(self.data)
        
        return self
    
    def filter(self, predicate: Callable[[Any], bool]) -> "DataTransformer":
        """
        Filtrar datos.
        
        Args:
            predicate: Función de filtrado
            
        Returns:
            Nuevo transformador
        """
        if isinstance(self.data, list):
            self.data = [item for item in self.data if predicate(item)]
        elif isinstance(self.data, dict):
            self.data = {k: v for k, v in self.data.items() if predicate(v)}
        
        return self
    
    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        initial: Optional[Any] = None
    ) -> Any:
        """
        Reducir datos.
        
        Args:
            func: Función de reducción
            initial: Valor inicial
            
        Returns:
            Resultado reducido
        """
        if isinstance(self.data, list):
            if initial is None:
                return reduce(func, self.data)
            return reduce(func, self.data, initial)
        
        return self.data
    
    def flatten(self, depth: int = 1) -> "DataTransformer":
        """
        Aplanar estructura anidada.
        
        Args:
            depth: Profundidad de aplanamiento
            
        Returns:
            Nuevo transformador
        """
        def _flatten(items, current_depth=0):
            if current_depth >= depth:
                return items
            
            result = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    result.extend(_flatten(item, current_depth + 1))
                else:
                    result.append(item)
            
            return result
        
        if isinstance(self.data, (list, tuple)):
            self.data = _flatten(self.data)
        
        return self
    
    def group_by(self, key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
        """
        Agrupar datos por clave.
        
        Args:
            key_func: Función para extraer clave
            
        Returns:
            Datos agrupados
        """
        if not isinstance(self.data, list):
            return {}
        
        grouped = {}
        for item in self.data:
            key = key_func(item)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        
        return grouped
    
    def sort_by(
        self,
        key_func: Optional[Callable[[Any], Any]] = None,
        reverse: bool = False
    ) -> "DataTransformer":
        """
        Ordenar datos.
        
        Args:
            key_func: Función para extraer clave de ordenamiento
            reverse: Orden inverso
            
        Returns:
            Nuevo transformador
        """
        if isinstance(self.data, list):
            self.data = sorted(self.data, key=key_func, reverse=reverse)
        
        return self
    
    def select(self, keys: List[str]) -> "DataTransformer":
        """
        Seleccionar campos específicos.
        
        Args:
            keys: Lista de claves a seleccionar
            
        Returns:
            Nuevo transformador
        """
        if isinstance(self.data, list):
            self.data = [
                {k: item.get(k) for k in keys if k in item}
                for item in self.data
                if isinstance(item, dict)
            ]
        elif isinstance(self.data, dict):
            self.data = {k: self.data.get(k) for k in keys if k in self.data}
        
        return self
    
    def rename(self, mapping: Dict[str, str]) -> "DataTransformer":
        """
        Renombrar campos.
        
        Args:
            mapping: Mapeo de nombres antiguos a nuevos
            
        Returns:
            Nuevo transformador
        """
        if isinstance(self.data, list):
            self.data = [
                {mapping.get(k, k): v for k, v in item.items()}
                for item in self.data
                if isinstance(item, dict)
            ]
        elif isinstance(self.data, dict):
            self.data = {mapping.get(k, k): v for k, v in self.data.items()}
        
        return self
    
    def get(self) -> Any:
        """Obtener datos transformados."""
        return self.data


def transform_data(
    data: Any,
    transformations: List[Dict[str, Any]]
) -> Any:
    """
    Aplicar múltiples transformaciones.
    
    Args:
        data: Datos a transformar
        transformations: Lista de transformaciones
        
    Returns:
        Datos transformados
    """
    transformer = DataTransformer(data)
    
    for transform in transformations:
        transform_type = transform.get("type")
        params = transform.get("params", {})
        
        if transform_type == "map":
            transformer.map(params.get("func"))
        elif transform_type == "filter":
            transformer.filter(params.get("predicate"))
        elif transform_type == "select":
            transformer.select(params.get("keys", []))
        elif transform_type == "rename":
            transformer.rename(params.get("mapping", {}))
        elif transform_type == "sort":
            transformer.sort_by(
                key_func=params.get("key_func"),
                reverse=params.get("reverse", False)
            )
        elif transform_type == "flatten":
            transformer.flatten(params.get("depth", 1))
    
    return transformer.get()


def normalize_data(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizar datos según esquema.
    
    Args:
        data: Datos a normalizar
        schema: Esquema de normalización
        
    Returns:
        Datos normalizados
    """
    normalized = {}
    
    for key, value in schema.items():
        source_key = value.get("source", key)
        default = value.get("default")
        transform = value.get("transform")
        
        source_value = data.get(source_key, default)
        
        if transform and callable(transform):
            source_value = transform(source_value)
        
        normalized[key] = source_value
    
    return normalized


def denormalize_data(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Desnormalizar datos según esquema.
    
    Args:
        data: Datos a desnormalizar
        schema: Esquema de desnormalización
        
    Returns:
        Datos desnormalizados
    """
    denormalized = {}
    
    for key, value in schema.items():
        target_key = value.get("target", key)
        transform = value.get("transform")
        
        source_value = data.get(key)
        
        if transform and callable(transform):
            source_value = transform(source_value)
        
        denormalized[target_key] = source_value
    
    return denormalized

