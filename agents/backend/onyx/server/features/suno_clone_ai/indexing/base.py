"""Base Indexer - Clase base para indexadores"""
from abc import ABC, abstractmethod
class BaseIndexer(ABC):
    @abstractmethod
    async def index(self, item: dict): pass
    @abstractmethod
    async def search(self, query: str, limit: int = 10): pass

