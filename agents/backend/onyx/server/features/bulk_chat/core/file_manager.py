"""
File Manager - Gestor de Archivos
==================================

Sistema avanzado de gestión de archivos con versionado, compresión y metadatos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import os

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Tipo de archivo."""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    ARCHIVE = "archive"


class FileStatus(Enum):
    """Estado de archivo."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    CORRUPTED = "corrupted"


@dataclass
class FileMetadata:
    """Metadatos de archivo."""
    file_id: str
    filename: str
    file_type: FileType
    size: int
    mime_type: str = ""
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    status: FileStatus = FileStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileVersion:
    """Versión de archivo."""
    version: int
    file_id: str
    created_at: datetime
    size: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FileManager:
    """Gestor de archivos."""
    
    def __init__(self, base_path: str = "./storage/files"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.files: Dict[str, FileMetadata] = {}
        self.file_versions: Dict[str, List[FileVersion]] = defaultdict(list)
        self.file_index: Dict[str, List[str]] = defaultdict(list)  # tag -> [file_ids]
        self._lock = asyncio.Lock()
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calcular checksum."""
        return hashlib.sha256(data).hexdigest()
    
    def _detect_file_type(self, filename: str, mime_type: str = "") -> FileType:
        """Detectar tipo de archivo."""
        ext = Path(filename).suffix.lower()
        
        if ext in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            return FileType.TEXT
        elif ext in ['.json', '.xml', '.yaml', '.yml']:
            return FileType.JSON
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp']:
            return FileType.IMAGE
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return FileType.VIDEO
        elif ext in ['.mp3', '.wav', '.flac', '.ogg']:
            return FileType.AUDIO
        elif ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']:
            return FileType.DOCUMENT
        elif ext in ['.zip', '.tar', '.gz', '.rar']:
            return FileType.ARCHIVE
        else:
            return FileType.BINARY
    
    async def upload_file(
        self,
        file_id: str,
        filename: str,
        data: bytes,
        mime_type: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Subir archivo."""
        checksum = self._calculate_checksum(data)
        file_type = self._detect_file_type(filename, mime_type)
        
        # Guardar archivo
        file_path = self.base_path / file_id
        file_path.write_bytes(data)
        
        file_metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            file_type=file_type,
            size=len(data),
            mime_type=mime_type,
            checksum=checksum,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        async with self._lock:
            # Verificar si existe versión anterior
            existing = self.files.get(file_id)
            if existing:
                # Crear nueva versión
                version = FileVersion(
                    version=existing.version,
                    file_id=file_id,
                    created_at=existing.modified_at,
                    size=existing.size,
                    checksum=existing.checksum,
                )
                self.file_versions[file_id].append(version)
                file_metadata.version = existing.version + 1
            else:
                file_metadata.version = 1
            
            self.files[file_id] = file_metadata
            
            # Indexar por tags
            for tag in file_metadata.tags:
                if file_id not in self.file_index[tag]:
                    self.file_index[tag].append(file_id)
        
        logger.info(f"Uploaded file: {file_id} - {filename} ({len(data)} bytes)")
        return file_id
    
    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Descargar archivo."""
        file_metadata = self.files.get(file_id)
        if not file_metadata or file_metadata.status != FileStatus.ACTIVE:
            return None
        
        file_path = self.base_path / file_id
        if not file_path.exists():
            return None
        
        data = file_path.read_bytes()
        
        # Verificar checksum
        checksum = self._calculate_checksum(data)
        if checksum != file_metadata.checksum:
            logger.warning(f"Checksum mismatch for file {file_id}")
            file_metadata.status = FileStatus.CORRUPTED
        
        return data
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Obtener metadatos de archivo."""
        file_metadata = self.files.get(file_id)
        if not file_metadata:
            return None
        
        return {
            "file_id": file_metadata.file_id,
            "filename": file_metadata.filename,
            "file_type": file_metadata.file_type.value,
            "size": file_metadata.size,
            "mime_type": file_metadata.mime_type,
            "checksum": file_metadata.checksum,
            "version": file_metadata.version,
            "status": file_metadata.status.value,
            "tags": file_metadata.tags,
            "created_at": file_metadata.created_at.isoformat(),
            "modified_at": file_metadata.modified_at.isoformat(),
            "metadata": file_metadata.metadata,
        }
    
    def get_file_versions(self, file_id: str) -> List[Dict[str, Any]]:
        """Obtener versiones de archivo."""
        versions = self.file_versions.get(file_id, [])
        
        return [
            {
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "size": v.size,
                "checksum": v.checksum,
            }
            for v in versions
        ]
    
    def search_files(
        self,
        tags: Optional[List[str]] = None,
        file_type: Optional[FileType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Buscar archivos."""
        candidates = list(self.files.values())
        
        if tags:
            # Archivos que tienen al menos uno de los tags
            tag_file_ids = set()
            for tag in tags:
                tag_file_ids.update(self.file_index.get(tag, []))
            
            candidates = [f for f in candidates if f.file_id in tag_file_ids]
        
        if file_type:
            candidates = [f for f in candidates if f.file_type == file_type]
        
        # Filtrar solo activos
        candidates = [f for f in candidates if f.status == FileStatus.ACTIVE]
        
        candidates.sort(key=lambda f: f.modified_at, reverse=True)
        
        return [self.get_file_metadata(f.file_id) for f in candidates[:limit]]
    
    async def delete_file(self, file_id: str, permanent: bool = False) -> bool:
        """Eliminar archivo."""
        file_metadata = self.files.get(file_id)
        if not file_metadata:
            return False
        
        async with self._lock:
            if permanent:
                # Eliminar permanentemente
                file_path = self.base_path / file_id
                if file_path.exists():
                    file_path.unlink()
                
                del self.files[file_id]
                
                # Remover de índices
                for tag in file_metadata.tags:
                    if file_id in self.file_index[tag]:
                        self.file_index[tag].remove(file_id)
            else:
                # Marcar como eliminado
                file_metadata.status = FileStatus.DELETED
        
        logger.info(f"Deleted file: {file_id} (permanent={permanent})")
        return True
    
    async def restore_file(self, file_id: str) -> bool:
        """Restaurar archivo eliminado."""
        file_metadata = self.files.get(file_id)
        if not file_metadata:
            return False
        
        if file_metadata.status == FileStatus.DELETED:
            file_metadata.status = FileStatus.ACTIVE
            return True
        
        return False
    
    def get_file_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        total_size = 0
        
        for file_metadata in self.files.values():
            by_type[file_metadata.file_type.value] += 1
            by_status[file_metadata.status.value] += 1
            total_size += file_metadata.size
        
        return {
            "total_files": len(self.files),
            "files_by_type": dict(by_type),
            "files_by_status": dict(by_status),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_tags": len(self.file_index),
        }
















