"""Base Connector - Clase base para conectores"""
from abc import ABC, abstractmethod
class BaseConnector(ABC):
    @abstractmethod
    async def connect(self, *args, **kwargs): pass

