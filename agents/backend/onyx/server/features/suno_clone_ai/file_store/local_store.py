"""
Local File Store - Almacenamiento local
"""

from pathlib import Path
from typing import Optional
import os
from .base import BaseFileStore


class LocalFileStore(BaseFileStore):
    """Almacenamiento de archivos local"""

    def __init__(self, base_path: str = "./storage"):
        """Inicializa el almacenamiento local"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(self, file_path: str, content: bytes) -> str:
        """Guarda un archivo"""
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
        return str(full_path)

    async def get(self, file_path: str) -> Optional[bytes]:
        """Obtiene un archivo"""
        full_path = self.base_path / file_path
        if full_path.exists():
            return full_path.read_bytes()
        return None

    async def delete(self, file_path: str) -> bool:
        """Elimina un archivo"""
        full_path = self.base_path / file_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

