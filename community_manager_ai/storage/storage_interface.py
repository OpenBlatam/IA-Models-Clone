"""
Storage Interface - Interfaz de Almacenamiento
==============================================

Interfaz abstracta para sistemas de almacenamiento.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class StorageInterface(ABC):
    """Interfaz abstracta para almacenamiento"""
    
    @abstractmethod
    def save(self, key: str, data: Dict[str, Any]) -> bool:
        """Guardar datos"""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Cargar datos"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Eliminar datos"""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """Listar claves"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Verificar si existe una clave"""
        pass




