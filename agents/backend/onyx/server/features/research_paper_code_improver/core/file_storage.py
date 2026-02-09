"""
File Storage - Sistema de almacenamiento de archivos (S3-like)
===============================================================
"""

import logging
import os
import hashlib
from typing import Dict, List, Any, Optional, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata de un archivo"""
    key: str
    size: int
    content_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    etag: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "etag": self.etag
        }


class FileStorage:
    """Sistema de almacenamiento de archivos"""
    
    def __init__(self, base_path: str = "./storage"):
        self.base_path = base_path
        self.files: Dict[str, FileMetadata] = {}
        os.makedirs(base_path, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadata desde disco"""
        metadata_file = os.path.join(self.base_path, ".metadata.json")
        if os.path.exists(metadata_file):
            try:
                import json
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    for key, meta_data in data.items():
                        self.files[key] = FileMetadata(
                            key=meta_data["key"],
                            size=meta_data["size"],
                            content_type=meta_data.get("content_type"),
                            created_at=datetime.fromisoformat(meta_data["created_at"]),
                            updated_at=datetime.fromisoformat(meta_data["updated_at"]),
                            metadata=meta_data.get("metadata", {}),
                            etag=meta_data.get("etag")
                        )
            except Exception as e:
                logger.error(f"Error cargando metadata: {e}")
    
    def _save_metadata(self):
        """Guarda metadata en disco"""
        metadata_file = os.path.join(self.base_path, ".metadata.json")
        try:
            import json
            data = {
                key: meta.to_dict() for key, meta in self.files.items()
            }
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """Obtiene la ruta completa de un archivo"""
        # Sanitizar key para evitar path traversal
        safe_key = key.replace("..", "").replace("/", "_")
        return os.path.join(self.base_path, safe_key)
    
    def _calculate_etag(self, file_path: str) -> str:
        """Calcula ETag de un archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def put(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FileMetadata:
        """Guarda un archivo"""
        file_path = self._get_file_path(key)
        
        # Guardar archivo
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(data)
        
        # Calcular ETag
        etag = self._calculate_etag(file_path)
        
        # Crear/actualizar metadata
        now = datetime.now()
        if key in self.files:
            file_meta = self.files[key]
            file_meta.size = len(data)
            file_meta.content_type = content_type or file_meta.content_type
            file_meta.updated_at = now
            file_meta.etag = etag
            if metadata:
                file_meta.metadata.update(metadata)
        else:
            file_meta = FileMetadata(
                key=key,
                size=len(data),
                content_type=content_type,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
                etag=etag
            )
            self.files[key] = file_meta
        
        self._save_metadata()
        logger.info(f"Archivo {key} guardado ({len(data)} bytes)")
        return file_meta
    
    def get(self, key: str) -> Optional[bytes]:
        """Obtiene un archivo"""
        if key not in self.files:
            return None
        
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'rb') as f:
            return f.read()
    
    def get_metadata(self, key: str) -> Optional[FileMetadata]:
        """Obtiene metadata de un archivo"""
        return self.files.get(key)
    
    def delete(self, key: str) -> bool:
        """Elimina un archivo"""
        if key not in self.files:
            return False
        
        file_path = self._get_file_path(key)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        del self.files[key]
        self._save_metadata()
        logger.info(f"Archivo {key} eliminado")
        return True
    
    def list_files(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[FileMetadata]:
        """Lista archivos"""
        files = list(self.files.values())
        
        if prefix:
            files = [f for f in files if f.key.startswith(prefix)]
        
        if limit:
            files = files[:limit]
        
        return files
    
    def copy(self, source_key: str, dest_key: str) -> bool:
        """Copia un archivo"""
        if source_key not in self.files:
            return False
        
        source_path = self._get_file_path(source_key)
        dest_path = self._get_file_path(dest_key)
        
        if not os.path.exists(source_path):
            return False
        
        shutil.copy2(source_path, dest_path)
        
        # Crear metadata para destino
        source_meta = self.files[source_key]
        dest_meta = FileMetadata(
            key=dest_key,
            size=source_meta.size,
            content_type=source_meta.content_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=source_meta.metadata.copy(),
            etag=self._calculate_etag(dest_path)
        )
        self.files[dest_key] = dest_meta
        self._save_metadata()
        
        return True
    
    def get_url(self, key: str, expires_in: Optional[int] = None) -> str:
        """Genera URL para acceso al archivo"""
        # En producción, generar signed URL
        return f"/storage/{key}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacenamiento"""
        total_size = sum(f.size for f in self.files.values())
        total_files = len(self.files)
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files_by_type": self._group_by_type()
        }
    
    def _group_by_type(self) -> Dict[str, int]:
        """Agrupa archivos por tipo"""
        by_type = {}
        for file_meta in self.files.values():
            content_type = file_meta.content_type or "unknown"
            by_type[content_type] = by_type.get(content_type, 0) + 1
        return by_type




