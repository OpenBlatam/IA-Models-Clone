"""
Repositorio Base

Proporciona funcionalidad común para repositorios de datos.
"""

import logging
from typing import Dict, Any, Optional, List, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Clase base para repositorios"""
    
    def __init__(self, entity_name: str):
        """
        Args:
            entity_name: Nombre de la entidad
        """
        self.entity_name = entity_name
        self.logger = logging.getLogger(f"{__name__}.{entity_name}")
        self._storage: Dict[str, T] = {}
        self.logger.info(f"{entity_name} repository initialized")
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """
        Crea una entidad
        
        Args:
            entity: Entidad a crear
        
        Returns:
            Entidad creada
        """
        pass
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Obtiene una entidad por ID
        
        Args:
            entity_id: ID de la entidad
        
        Returns:
            Entidad o None
        """
        pass
    
    @abstractmethod
    def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualiza una entidad
        
        Args:
            entity_id: ID de la entidad
            updates: Campos a actualizar
        
        Returns:
            True si se actualizó exitosamente
        """
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """
        Elimina una entidad
        
        Args:
            entity_id: ID de la entidad
        
        Returns:
            True si se eliminó exitosamente
        """
        pass
    
    def list_all(self, limit: Optional[int] = None) -> List[T]:
        """
        Lista todas las entidades
        
        Args:
            limit: Límite de resultados
        
        Returns:
            Lista de entidades
        """
        entities = list(self._storage.values())
        if limit:
            return entities[:limit]
        return entities
    
    def count(self) -> int:
        """Cuenta el número de entidades"""
        return len(self._storage)
    
    def exists(self, entity_id: str) -> bool:
        """Verifica si una entidad existe"""
        return entity_id in self._storage


class InMemoryRepository(BaseRepository[T]):
    """Repositorio en memoria (implementación base)"""
    
    def create(self, entity: T) -> T:
        """Crea una entidad en memoria"""
        # Asumimos que la entidad tiene un atributo 'id'
        entity_id = getattr(entity, 'id', None) or getattr(entity, f'{self.entity_name}_id', None)
        if not entity_id:
            raise ValueError(f"Entity must have an 'id' or '{self.entity_name}_id' attribute")
        
        self._storage[entity_id] = entity
        self.logger.debug(f"{self.entity_name} created: {entity_id}")
        return entity
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Obtiene una entidad por ID"""
        return self._storage.get(entity_id)
    
    def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Actualiza una entidad"""
        if entity_id not in self._storage:
            return False
        
        entity = self._storage[entity_id]
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        
        # Actualizar timestamp si existe
        if hasattr(entity, 'updated_at'):
            entity.updated_at = datetime.now()
        
        self.logger.debug(f"{self.entity_name} updated: {entity_id}")
        return True
    
    def delete(self, entity_id: str) -> bool:
        """Elimina una entidad"""
        if entity_id not in self._storage:
            return False
        
        del self._storage[entity_id]
        self.logger.debug(f"{self.entity_name} deleted: {entity_id}")
        return True
    
    def find_by(self, **criteria) -> List[T]:
        """
        Busca entidades por criterios
        
        Args:
            **criteria: Criterios de búsqueda
        
        Returns:
            Lista de entidades que coinciden
        """
        results = []
        for entity in self._storage.values():
            match = True
            for key, value in criteria.items():
                if not hasattr(entity, key) or getattr(entity, key) != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results

