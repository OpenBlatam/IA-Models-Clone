"""
Pipeline Context Management
===========================

Manejo de contexto para pipelines.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import threading
from typing import Dict, Any, Optional
from copy import deepcopy
from contextlib import contextmanager


class PipelineContext:
    """
    Contexto compartido para pipelines.
    
    Thread-safe y con soporte para anidamiento.
    Optimizado con mejor rendimiento y seguridad.
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Inicializar contexto.
        
        Args:
            initial_data: Datos iniciales
        """
        self._data: Dict[str, Any] = initial_data.copy() if initial_data else {}
        self._lock = threading.RLock()
        self._parent: Optional['PipelineContext'] = None
    
    def set(self, key: str, value: Any) -> None:
        """
        Establecer valor.
        
        Args:
            key: Clave
            value: Valor
        """
        if not key:
            raise ValueError("Key cannot be empty")
        
        with self._lock:
            self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor.
        
        Args:
            key: Clave
            default: Valor por defecto
            
        Returns:
            Valor o default
        """
        if not key:
            return default
        
        with self._lock:
            if key in self._data:
                return self._data[key]
            
            if self._parent:
                return self._parent.get(key, default)
            
            return default
    
    def has(self, key: str) -> bool:
        """
        Verificar si existe clave.
        
        Args:
            key: Clave
            
        Returns:
            True si existe
        """
        if not key:
            return False
        
        with self._lock:
            if key in self._data:
                return True
            
            if self._parent:
                return self._parent.has(key)
            
            return False
    
    def remove(self, key: str) -> bool:
        """
        Remover clave.
        
        Args:
            key: Clave
            
        Returns:
            True si se removió, False si no existía
        """
        if not key:
            return False
        
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def clear(self) -> None:
        """Limpiar contexto."""
        with self._lock:
            self._data.clear()
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Actualizar con diccionario.
        
        Args:
            data: Datos a actualizar
        """
        if not data:
            return
        
        with self._lock:
            self._data.update(data)
    
    def copy(self) -> 'PipelineContext':
        """
        Crear copia del contexto.
        
        Returns:
            Nueva copia del contexto
        """
        with self._lock:
            new_context = PipelineContext(deepcopy(self._data))
            if self._parent:
                new_context._parent = self._parent.copy()
            return new_context
    
    def create_child(self) -> 'PipelineContext':
        """
        Crear contexto hijo.
        
        Returns:
            Nuevo contexto hijo
        """
        child = PipelineContext()
        child._parent = self
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con todos los valores
        """
        with self._lock:
            result = deepcopy(self._data)
            if self._parent:
                parent_dict = self._parent.to_dict()
                parent_dict.update(result)
                return parent_dict
            return result
    
    def keys(self) -> set[str]:
        """
        Obtener todas las claves.
        
        Returns:
            Set de claves
        """
        with self._lock:
            keys = set(self._data.keys())
            if self._parent:
                keys.update(self._parent.keys())
            return keys
    
    def values(self) -> Dict[str, Any]:
        """
        Obtener todos los valores como diccionario.
        
        Returns:
            Diccionario con todos los valores
        """
        return self.to_dict()
    
    @contextmanager
    def temporary(self, **kwargs: Any):
        """
        Context manager para valores temporales.
        
        Args:
            **kwargs: Valores temporales a establecer
            
        Yields:
            Contexto con valores temporales
        """
        original_values = {k: self.get(k) for k in kwargs.keys() if self.has(k)}
        
        try:
            self.update(kwargs)
            yield self
        finally:
            for key, value in original_values.items():
                if value is not None:
                    self.set(key, value)
                else:
                    self.remove(key)
    
    def __contains__(self, key: str) -> bool:
        """Verificar si contiene clave."""
        return self.has(key)
    
    def __getitem__(self, key: str) -> Any:
        """
        Obtener valor.
        
        Raises:
            KeyError: Si la clave no existe
        """
        value = self.get(key)
        if value is None and not self.has(key):
            raise KeyError(f"Key '{key}' not found in context")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Establecer valor."""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """
        Eliminar valor.
        
        Raises:
            KeyError: Si la clave no existe
        """
        if not self.remove(key):
            raise KeyError(f"Key '{key}' not found in context")
    
    def __len__(self) -> int:
        """Obtener cantidad de claves."""
        return len(self.keys())


class ContextManager:
    """
    Gestor de contextos para pipelines.
    Optimizado con mejor validación y rendimiento.
    """
    
    def __init__(self) -> None:
        """Inicializar gestor."""
        self._contexts: Dict[str, PipelineContext] = {}
        self._lock = threading.RLock()
    
    def create_context(
        self,
        name: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> PipelineContext:
        """
        Crear nuevo contexto.
        
        Args:
            name: Nombre del contexto
            initial_data: Datos iniciales
            
        Returns:
            Nuevo contexto
            
        Raises:
            ValueError: Si el nombre está vacío o ya existe
        """
        if not name:
            raise ValueError("Context name cannot be empty")
        
        with self._lock:
            if name in self._contexts:
                raise ValueError(f"Context '{name}' already exists")
            
            context = PipelineContext(initial_data)
            self._contexts[name] = context
            return context
    
    def get_context(self, name: str) -> Optional[PipelineContext]:
        """
        Obtener contexto.
        
        Args:
            name: Nombre del contexto
            
        Returns:
            Contexto o None
        """
        if not name:
            return None
        
        with self._lock:
            return self._contexts.get(name)
    
    def remove_context(self, name: str) -> bool:
        """
        Remover contexto.
        
        Args:
            name: Nombre del contexto
            
        Returns:
            True si se removió, False si no existía
        """
        if not name:
            return False
        
        with self._lock:
            if name in self._contexts:
                del self._contexts[name]
                return True
            return False
    
    def clear(self) -> None:
        """Limpiar todos los contextos."""
        with self._lock:
            self._contexts.clear()
    
    def list_contexts(self) -> list[str]:
        """
        Listar nombres de contextos.
        
        Returns:
            Lista de nombres de contextos
        """
        with self._lock:
            return list(self._contexts.keys())
