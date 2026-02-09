"""
File Store Service - Servicio de almacenamiento
"""

from typing import Optional
from .base import BaseFileStore
from .local_store import LocalFileStore
from configs.settings import Settings


class FileStoreService:
    """Servicio para gestionar almacenamiento de archivos"""

    def __init__(self, store: Optional[BaseFileStore] = None, settings: Optional[Settings] = None):
        """Inicializa el servicio de almacenamiento"""
        self.settings = settings or Settings()
        self.store = store or LocalFileStore()

    async def save(self, file_path: str, content: bytes) -> str:
        """Guarda un archivo"""
        return await self.store.save(file_path, content)

    async def get(self, file_path: str) -> Optional[bytes]:
        """Obtiene un archivo"""
        return await self.store.get(file_path)

    async def delete(self, file_path: str) -> bool:
        """Elimina un archivo"""
        return await self.store.delete(file_path)

