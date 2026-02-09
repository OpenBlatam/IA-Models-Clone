"""Base Federated Connector - Clase base para conectores federados"""
from abc import ABC, abstractmethod
class BaseFederatedConnector(ABC):
    @abstractmethod
    async def connect(self, *args, **kwargs): pass

