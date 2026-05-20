"""
Data Versioning Service - Versionado de datos
==============================================

Sistema para versionar y gestionar historial de cambios en datos.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Tipos de cambio"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"


@dataclass
class DataVersion:
    """Versión de datos"""
    version_id: str
    resource_type: str
    resource_id: str
    data: Dict[str, Any]
    change_type: ChangeType
    changed_by: Optional[str] = None
    change_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionHistory:
    """Historial de versiones"""
    resource_type: str
    resource_id: str
    versions: List[DataVersion]
    current_version: Optional[DataVersion] = None
    total_versions: int = 0


class DataVersioningService:
    """Servicio de versionado de datos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.versions: Dict[str, List[DataVersion]] = {}  # resource_key -> versions
        logger.info("DataVersioningService initialized")
    
    def create_version(
        self,
        resource_type: str,
        resource_id: str,
        data: Dict[str, Any],
        change_type: ChangeType,
        changed_by: Optional[str] = None,
        change_reason: Optional[str] = None
    ) -> DataVersion:
        """Crear nueva versión"""
        resource_key = f"{resource_type}:{resource_id}"
        version_id = f"v{len(self.versions.get(resource_key, [])) + 1}"
        
        version = DataVersion(
            version_id=version_id,
            resource_type=resource_type,
            resource_id=resource_id,
            data=data,
            change_type=change_type,
            changed_by=changed_by,
            change_reason=change_reason,
        )
        
        if resource_key not in self.versions:
            self.versions[resource_key] = []
        
        self.versions[resource_key].append(version)
        
        logger.info(f"Version created: {version_id} for {resource_key}")
        return version
    
    def get_version_history(
        self,
        resource_type: str,
        resource_id: str
    ) -> VersionHistory:
        """Obtener historial de versiones"""
        resource_key = f"{resource_type}:{resource_id}"
        versions = self.versions.get(resource_key, [])
        
        current_version = versions[-1] if versions else None
        
        return VersionHistory(
            resource_type=resource_type,
            resource_id=resource_id,
            versions=versions,
            current_version=current_version,
            total_versions=len(versions),
        )
    
    def get_version(
        self,
        resource_type: str,
        resource_id: str,
        version_id: str
    ) -> Optional[DataVersion]:
        """Obtener versión específica"""
        resource_key = f"{resource_type}:{resource_id}"
        versions = self.versions.get(resource_key, [])
        
        return next((v for v in versions if v.version_id == version_id), None)
    
    def restore_version(
        self,
        resource_type: str,
        resource_id: str,
        version_id: str,
        restored_by: str
    ) -> DataVersion:
        """Restaurar a una versión específica"""
        version = self.get_version(resource_type, resource_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Crear nueva versión con datos restaurados
        restored_version = self.create_version(
            resource_type=resource_type,
            resource_id=resource_id,
            data=version.data,
            change_type=ChangeType.RESTORE,
            changed_by=restored_by,
            change_reason=f"Restored from version {version_id}",
        )
        
        return restored_version
    
    def compare_versions(
        self,
        resource_type: str,
        resource_id: str,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """Comparar dos versiones"""
        version1 = self.get_version(resource_type, resource_id, version_id_1)
        version2 = self.get_version(resource_type, resource_id, version_id_2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Comparar datos (simplificado)
        differences = []
        all_keys = set(version1.data.keys()) | set(version2.data.keys())
        
        for key in all_keys:
            val1 = version1.data.get(key)
            val2 = version2.data.get(key)
            
            if val1 != val2:
                differences.append({
                    "field": key,
                    "old_value": val1,
                    "new_value": val2,
                })
        
        return {
            "version1": version_id_1,
            "version2": version_id_2,
            "differences": differences,
            "total_differences": len(differences),
        }




