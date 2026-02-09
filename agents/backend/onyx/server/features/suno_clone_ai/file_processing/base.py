"""Base File Processor - Clase base para procesadores"""
from abc import ABC, abstractmethod
class BaseFileProcessor(ABC):
    @abstractmethod
    async def process(self, file_path: str, *args, **kwargs): pass

