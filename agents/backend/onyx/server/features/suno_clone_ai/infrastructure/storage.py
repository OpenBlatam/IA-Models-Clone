"""
Storage Infrastructure
Abstracción para diferentes backends de almacenamiento
"""

import logging
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Tipos de almacenamiento soportados"""
    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class StorageManager(ABC):
    """Interfaz abstracta para gestión de almacenamiento"""
    
    @abstractmethod
    async def upload(self, key: str, file_obj: BinaryIO, metadata: Dict[str, Any] = None) -> bool:
        """Sube un archivo"""
        pass
    
    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Descarga un archivo"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina un archivo"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verifica si un archivo existe"""
        pass
    
    @abstractmethod
    async def get_url(self, key: str, expiration: int = 3600) -> str:
        """Obtiene una URL para acceso temporal"""
        pass


class LocalStorageManager(StorageManager):
    """Implementación de almacenamiento local"""
    
    def __init__(self, base_path: str = "./storage"):
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    async def upload(self, key: str, file_obj: BinaryIO, metadata: Dict[str, Any] = None) -> bool:
        """Sube un archivo localmente"""
        import os
        file_path = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            if isinstance(file_obj, bytes):
                f.write(file_obj)
            else:
                f.write(file_obj.read())
        
        return True
    
    async def download(self, key: str) -> bytes:
        """Descarga un archivo localmente"""
        import os
        file_path = os.path.join(self.base_path, key)
        with open(file_path, 'rb') as f:
            return f.read()
    
    async def delete(self, key: str) -> bool:
        """Elimina un archivo localmente"""
        import os
        file_path = os.path.join(self.base_path, key)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Verifica si un archivo existe localmente"""
        import os
        file_path = os.path.join(self.base_path, key)
        return os.path.exists(file_path)
    
    async def get_url(self, key: str, expiration: int = 3600) -> str:
        """Obtiene una URL local (file://)"""
        import os
        file_path = os.path.join(self.base_path, key)
        return f"file://{os.path.abspath(file_path)}"


class S3StorageManager(StorageManager):
    """Implementación de almacenamiento con S3"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._client = None
    
    async def _get_client(self):
        """Obtiene el cliente S3"""
        if not self._client:
            from aws.services.s3_service import S3Service
            self._client = S3Service(
                bucket_name=self.bucket_name,
                region_name=self.region
            )
        return self._client
    
    async def upload(self, key: str, file_obj: BinaryIO, metadata: Dict[str, Any] = None) -> bool:
        """Sube un archivo a S3"""
        client = await self._get_client()
        await client.upload_file(
            file_obj,
            key,
            metadata=metadata
        )
        return True
    
    async def download(self, key: str) -> bytes:
        """Descarga un archivo de S3"""
        client = await self._get_client()
        return await client.download_file(key)
    
    async def delete(self, key: str) -> bool:
        """Elimina un archivo de S3"""
        client = await self._get_client()
        return await client.delete_file(key)
    
    async def exists(self, key: str) -> bool:
        """Verifica si un archivo existe en S3"""
        client = await self._get_client()
        return await client.file_exists(key)
    
    async def get_url(self, key: str, expiration: int = 3600) -> str:
        """Obtiene una URL pre-firmada de S3"""
        client = await self._get_client()
        return await client.get_presigned_url(key, expiration=expiration)


# Factory function
_storage_manager: Optional[StorageManager] = None


def get_storage() -> Optional[StorageManager]:
    """Obtiene la instancia global del gestor de almacenamiento"""
    return _storage_manager


def create_storage_manager(
    storage_type: StorageType,
    **kwargs
) -> StorageManager:
    """Crea un gestor de almacenamiento según el tipo"""
    if storage_type == StorageType.LOCAL:
        return LocalStorageManager(
            base_path=kwargs.get('base_path', './storage')
        )
    elif storage_type == StorageType.S3:
        return S3StorageManager(
            bucket_name=kwargs.get('bucket_name'),
            region=kwargs.get('region', 'us-east-1')
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")















