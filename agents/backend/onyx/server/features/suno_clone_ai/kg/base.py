"""Base Knowledge Graph - Clase base para KG"""
from abc import ABC, abstractmethod
class BaseKnowledgeGraph(ABC):
    @abstractmethod
    async def add_entity(self, entity: dict): pass
    @abstractmethod
    async def query(self, query: str): pass

