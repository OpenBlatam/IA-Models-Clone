"""
File Storage Optimizer
Optimizaciones para almacenamiento de archivos grandes
"""

import logging
import asyncio
import aiofiles
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class FileStorageOptimizer:
    """Optimizador para almacenamiento de archivos"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB
        self.chunk_size = chunk_size
        self._upload_cache: Dict[str, Any] = {}
    
    async def upload_file_chunked(
        self,
        file_path: Path,
        upload_func: callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sube archivo en chunks para mejor rendimiento
        
        Args:
            file_path: Ruta del archivo
            upload_func: Función async que sube un chunk
            *args, **kwargs: Argumentos adicionales
            
        Returns:
            Resultado de la subida
        """
        file_size = file_path.stat().st_size
        total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
        
        results = []
        
        async with aiofiles.open(file_path, 'rb') as f:
            for chunk_num in range(total_chunks):
                chunk_data = await f.read(self.chunk_size)
                
                result = await upload_func(
                    chunk_data,
                    chunk_num=chunk_num,
                    total_chunks=total_chunks,
                    *args,
                    **kwargs
                )
                results.append(result)
        
        return {
            "total_chunks": total_chunks,
            "results": results
        }
    
    async def download_file_chunked(
        self,
        download_func: callable,
        output_path: Path,
        file_size: int = None,
        *args,
        **kwargs
    ):
        """
        Descarga archivo en chunks
        
        Args:
            download_func: Función async que descarga un chunk
            output_path: Ruta de salida
            file_size: Tamaño del archivo (opcional)
            *args, **kwargs: Argumentos adicionales
        """
        async with aiofiles.open(output_path, 'wb') as f:
            chunk_num = 0
            
            while True:
                chunk_data = await download_func(
                    chunk_num=chunk_num,
                    chunk_size=self.chunk_size,
                    *args,
                    **kwargs
                )
                
                if not chunk_data:
                    break
                
                await f.write(chunk_data)
                chunk_num += 1
                
                # Verificar si terminamos
                if file_size and f.tell() >= file_size:
                    break
    
    def optimize_storage_path(
        self,
        file_id: str,
        file_type: str = "audio"
    ) -> Path:
        """
        Optimiza ruta de almacenamiento
        
        Args:
            file_id: ID del archivo
            file_type: Tipo de archivo
            
        Returns:
            Path optimizado
        """
        # Usar hash para distribución
        hash_prefix = file_id[:2]
        hash_subprefix = file_id[2:4]
        
        path = Path(file_type) / hash_prefix / hash_subprefix / file_id
        
        return path
    
    def get_optimal_chunk_size(self, file_size: int) -> int:
        """
        Calcula tamaño óptimo de chunk
        
        Args:
            file_size: Tamaño del archivo
            
        Returns:
            Tamaño óptimo de chunk
        """
        # Chunks más grandes para archivos grandes
        if file_size < 10 * 1024 * 1024:  # < 10MB
            return 512 * 1024  # 512KB
        elif file_size < 100 * 1024 * 1024:  # < 100MB
            return 1024 * 1024  # 1MB
        else:
            return 4 * 1024 * 1024  # 4MB


# Instancia global
_file_storage_optimizer: Optional[FileStorageOptimizer] = None


def get_file_storage_optimizer() -> FileStorageOptimizer:
    """Obtiene el optimizador de almacenamiento"""
    global _file_storage_optimizer
    if _file_storage_optimizer is None:
        _file_storage_optimizer = FileStorageOptimizer()
    return _file_storage_optimizer















