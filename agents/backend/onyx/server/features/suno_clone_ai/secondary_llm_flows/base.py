"""Base Secondary Flow - Clase base para flujos"""
from abc import ABC, abstractmethod
class BaseSecondaryFlow(ABC):
    @abstractmethod
    async def execute(self, *args, **kwargs): pass

