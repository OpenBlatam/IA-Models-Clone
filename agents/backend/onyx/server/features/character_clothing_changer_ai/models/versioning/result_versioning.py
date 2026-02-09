"""
Result Versioning System
========================
Sistema de versionado de resultados con historial completo
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class VersionStatus(Enum):
    """Estados de versión"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Version:
    """Versión de resultado"""
    id: str
    result_id: str
    version_number: int
    data: Dict[str, Any]
    created_at: float
    created_by: Optional[str]
    status: VersionStatus
    description: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VersionDiff:
    """Diferencia entre versiones"""
    from_version: int
    to_version: int
    changes: Dict[str, Any]
    added_fields: List[str]
    removed_fields: List[str]
    modified_fields: List[str]


class ResultVersioning:
    """
    Sistema de versionado de resultados
    """
    
    def __init__(self):
        self.versions: Dict[str, List[Version]] = {}  # result_id -> versions
        self.current_versions: Dict[str, int] = {}  # result_id -> current version number
    
    def create_version(
        self,
        result_id: str,
        data: Dict[str, Any],
        created_by: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Version:
        """
        Crear nueva versión
        
        Args:
            result_id: ID del resultado
            data: Datos del resultado
            created_by: Usuario que crea la versión
            description: Descripción de la versión
            tags: Tags de la versión
            metadata: Metadata adicional
        """
        if result_id not in self.versions:
            self.versions[result_id] = []
            self.current_versions[result_id] = 0
        
        version_number = self.current_versions[result_id] + 1
        
        version_id = hashlib.sha256(
            f"{result_id}{version_number}{time.time()}".encode()
        ).hexdigest()[:16]
        
        version = Version(
            id=version_id,
            result_id=result_id,
            version_number=version_number,
            data=data,
            created_at=time.time(),
            created_by=created_by,
            status=VersionStatus.DRAFT,
            description=description,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.versions[result_id].append(version)
        self.current_versions[result_id] = version_number
        
        return version
    
    def get_version(self, result_id: str, version_number: int) -> Optional[Version]:
        """Obtener versión específica"""
        if result_id not in self.versions:
            return None
        
        for version in self.versions[result_id]:
            if version.version_number == version_number:
                return version
        
        return None
    
    def get_current_version(self, result_id: str) -> Optional[Version]:
        """Obtener versión actual"""
        if result_id not in self.current_versions:
            return None
        
        return self.get_version(result_id, self.current_versions[result_id])
    
    def get_all_versions(self, result_id: str) -> List[Version]:
        """Obtener todas las versiones"""
        return self.versions.get(result_id, [])
    
    def update_version(
        self,
        result_id: str,
        version_number: int,
        data: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[VersionStatus] = None
    ) -> Optional[Version]:
        """Actualizar versión existente"""
        version = self.get_version(result_id, version_number)
        if not version:
            return None
        
        if data:
            version.data.update(data)
        if description:
            version.description = description
        if tags:
            version.tags = tags
        if status:
            version.status = status
        
        return version
    
    def publish_version(self, result_id: str, version_number: int) -> bool:
        """Publicar versión"""
        version = self.get_version(result_id, version_number)
        if not version:
            return False
        
        version.status = VersionStatus.PUBLISHED
        return True
    
    def archive_version(self, result_id: str, version_number: int) -> bool:
        """Archivar versión"""
        version = self.get_version(result_id, version_number)
        if not version:
            return False
        
        version.status = VersionStatus.ARCHIVED
        return True
    
    def compare_versions(
        self,
        result_id: str,
        from_version: int,
        to_version: int
    ) -> Optional[VersionDiff]:
        """Comparar dos versiones"""
        from_ver = self.get_version(result_id, from_version)
        to_ver = self.get_version(result_id, to_version)
        
        if not from_ver or not to_ver:
            return None
        
        from_data = from_ver.data
        to_data = to_ver.data
        
        added = [k for k in to_data.keys() if k not in from_data]
        removed = [k for k in from_data.keys() if k not in to_data]
        modified = [
            k for k in from_data.keys()
            if k in to_data and from_data[k] != to_data[k]
        ]
        
        changes = {}
        for key in modified:
            changes[key] = {
                'from': from_data[key],
                'to': to_data[key]
            }
        
        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            changes=changes,
            added_fields=added,
            removed_fields=removed,
            modified_fields=modified
        )
    
    def rollback_to_version(
        self,
        result_id: str,
        version_number: int
    ) -> Optional[Version]:
        """Hacer rollback a una versión anterior"""
        version = self.get_version(result_id, version_number)
        if not version:
            return None
        
        # Crear nueva versión con datos de la versión anterior
        new_version = self.create_version(
            result_id=result_id,
            data=version.data.copy(),
            description=f"Rollback to version {version_number}",
            metadata={
                'rollback_from': self.current_versions[result_id],
                'rollback_to': version_number
            }
        )
        
        return new_version
    
    def get_version_history(self, result_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de versiones"""
        versions = self.get_all_versions(result_id)
        
        return [
            {
                'version_number': v.version_number,
                'created_at': v.created_at,
                'created_by': v.created_by,
                'status': v.status.value,
                'description': v.description,
                'tags': v.tags
            }
            for v in versions
        ]
    
    def search_versions(
        self,
        result_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[VersionStatus] = None,
        created_by: Optional[str] = None
    ) -> List[Version]:
        """Buscar versiones"""
        results = []
        
        for rid, versions in self.versions.items():
            if result_id and rid != result_id:
                continue
            
            for version in versions:
                if tags and not any(tag in version.tags for tag in tags):
                    continue
                if status and version.status != status:
                    continue
                if created_by and version.created_by != created_by:
                    continue
                
                results.append(version)
        
        return results
    
    def delete_version(self, result_id: str, version_number: int) -> bool:
        """Eliminar versión (soft delete)"""
        version = self.get_version(result_id, version_number)
        if not version:
            return False
        
        version.status = VersionStatus.DELETED
        return True


# Instancia global
result_versioning = ResultVersioning()

