"""
Entity Manager Mixin
====================
Mixin para sistemas que gestionan entidades (CRUD común)
"""

from typing import Dict, List, Optional, Any, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')


class EntityManagerMixin(Generic[T]):
    """
    Mixin para gestión de entidades con operaciones CRUD comunes
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entities: Dict[str, T] = {}
        self._entity_index: Dict[str, List[str]] = {}  # Para búsquedas rápidas
    
    def create_entity(self, entity_id: str, entity: T) -> T:
        """Crear entidad"""
        if entity_id in self._entities:
            raise ValueError(f"Entity {entity_id} already exists")
        
        self._entities[entity_id] = entity
        self._index_entity(entity_id, entity)
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[T]:
        """Obtener entidad por ID"""
        return self._entities.get(entity_id)
    
    def update_entity(self, entity_id: str, entity: T) -> T:
        """Actualizar entidad"""
        if entity_id not in self._entities:
            raise ValueError(f"Entity {entity_id} not found")
        
        self._entities[entity_id] = entity
        self._index_entity(entity_id, entity)
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """Eliminar entidad"""
        if entity_id not in self._entities:
            return False
        
        del self._entities[entity_id]
        self._unindex_entity(entity_id)
        return True
    
    def list_entities(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """Listar entidades"""
        entities = list(self._entities.values())
        if offset > 0:
            entities = entities[offset:]
        if limit:
            entities = entities[:limit]
        return entities
    
    def count_entities(self) -> int:
        """Contar entidades"""
        return len(self._entities)
    
    def search_entities(self, query: str) -> List[T]:
        """
        Buscar entidades
        
        Debe ser sobrescrito por clases específicas para implementar búsqueda
        """
        # Implementación básica - buscar en índices
        matching_ids = set()
        query_lower = query.lower()
        
        for index_key, entity_ids in self._entity_index.items():
            if query_lower in index_key.lower():
                matching_ids.update(entity_ids)
        
        return [self._entities[eid] for eid in matching_ids if eid in self._entities]
    
    def _index_entity(self, entity_id: str, entity: T):
        """Indexar entidad para búsqueda"""
        # Implementación básica - indexar por ID
        # Las clases hijas pueden sobrescribir para indexar por otros campos
        self._entity_index[entity_id] = [entity_id]
    
    def _unindex_entity(self, entity_id: str):
        """Remover entidad del índice"""
        if entity_id in self._entity_index:
            del self._entity_index[entity_id]
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de entidades"""
        return {
            'total_entities': self.count_entities(),
            'indexed_keys': len(self._entity_index)
        }
    
    def clear_entities(self):
        """Limpiar todas las entidades"""
        self._entities.clear()
        self._entity_index.clear()

