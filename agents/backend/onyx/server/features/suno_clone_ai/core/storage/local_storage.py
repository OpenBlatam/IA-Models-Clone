"""
Implementación de almacenamiento local

Proporciona almacenamiento en sistema de archivos local.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from core.interfaces import IStorageBackend

logger = logging.getLogger(__name__)


class LocalStorage(IStorageBackend):
    """Almacenamiento local en sistema de archivos"""
    
    def __init__(self, base_path: str = "storage"):
        """
        Args:
            base_path: Ruta base para almacenamiento
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized at {self.base_path}")
    
    async def save(self, path: str, data: bytes) -> bool:
        """Guarda datos en el sistema de archivos"""
        try:
            full_path = self.base_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, "wb") as f:
                f.write(data)
            
            logger.debug(f"File saved: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False
    
    async def load(self, path: str) -> Optional[bytes]:
        """Carga datos del sistema de archivos"""
        try:
            full_path = self.base_path / path
            
            if not full_path.exists():
                return None
            
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return None
    
    async def delete(self, path: str) -> bool:
        """Elimina datos del sistema de archivos"""
        try:
            full_path = self.base_path / path
            
            if not full_path.exists():
                return False
            
            full_path.unlink()
            logger.debug(f"File deleted: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def exists(self, path: str) -> bool:
        """Verifica si un archivo existe"""
        full_path = self.base_path / path
        return full_path.exists()
    
    async def list_files(self, prefix: str = "") -> list:
        """Lista archivos con un prefijo"""
        prefix_path = self.base_path / prefix
        if not prefix_path.exists():
            return []
        
        files = []
        for file_path in prefix_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.base_path)
                files.append(str(relative_path))
        
        return files

