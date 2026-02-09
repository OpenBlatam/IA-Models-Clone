"""
S3 File Store - Almacenamiento S3
"""

from typing import Optional
from .base import BaseFileStore


class S3FileStore(BaseFileStore):
    """Almacenamiento de archivos en S3"""

    def __init__(self, bucket_name: str):
        """Inicializa el almacenamiento S3"""
        self.bucket_name = bucket_name
        # TODO: Implementar cliente S3

    async def save(self, file_path: str, content: bytes) -> str:
        """Guarda un archivo en S3"""
        # TODO: Implementar guardado en S3
        raise NotImplementedError

    async def get(self, file_path: str) -> Optional[bytes]:
        """Obtiene un archivo de S3"""
        # TODO: Implementar obtención de S3
        raise NotImplementedError

    async def delete(self, file_path: str) -> bool:
        """Elimina un archivo de S3"""
        # TODO: Implementar eliminación en S3
        raise NotImplementedError

