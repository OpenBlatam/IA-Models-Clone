"""
Knowledge Graph Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4


class Entity:
    """Entity in knowledge graph"""
    
    def __init__(
        self,
        entity_type: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.entity_type = entity_type
        self.name = name
        self.properties = properties or {}
        self.created_at = datetime.utcnow()


class Relation:
    """Relation between entities"""
    
    def __init__(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.from_entity_id = from_entity_id
        self.to_entity_id = to_entity_id
        self.relation_type = relation_type
        self.properties = properties or {}
        self.created_at = datetime.utcnow()


class KnowledgeGraph:
    """Knowledge graph definition"""
    
    def __init__(self, name: str):
        self.id = str(uuid4())
        self.name = name
        self.entities: List[Entity] = []
        self.relations: List[Relation] = []
        self.created_at = datetime.utcnow()


class KGBase(ABC):
    """Base interface for knowledge graph"""
    
    @abstractmethod
    async def add_entity(self, entity: Entity) -> bool:
        """Add entity to graph"""
        pass
    
    @abstractmethod
    async def add_relation(self, relation: Relation) -> bool:
        """Add relation to graph"""
        pass
    
    @abstractmethod
    async def query(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph"""
        pass
    
    @abstractmethod
    async def get_entity_relations(
        self,
        entity_id: str
    ) -> List[Relation]:
        """Get relations for entity"""
        pass

