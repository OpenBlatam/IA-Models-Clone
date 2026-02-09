"""Base Document Index - Clase base para indexación"""
from abc import ABC, abstractmethod
class BaseDocumentIndex(ABC):
    @abstractmethod
    async def index(self, document: dict): pass
    @abstractmethod
    async def search(self, query: str, limit: int = 10): pass

