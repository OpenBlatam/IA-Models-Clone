"""
Base File Store - Clase base para storage
"""

from abc import ABC, abstractmethod
from typing import Optional, BinaryIO


class BaseFileStore(ABC):
    """Clase base abstracta para almacenamiento de archivos"""

    @abstractmethod
    async def save(self, file_path: str, content: bytes) -> str:
        """Guarda un archivo"""
        pass

    @abstractmethod
    async def get(self, file_path: str) -> Optional[bytes]:
        """Obtiene un archivo"""
        pass

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """Elimina un archivo"""
        pass

