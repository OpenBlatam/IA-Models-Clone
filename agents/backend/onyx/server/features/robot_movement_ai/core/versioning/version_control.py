"""
Version Control System
=======================

Sistema de control de versiones para configuraciones y datos.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """Versión."""
    version_id: str
    entity_type: str
    entity_id: str
    data: Dict[str, Any]
    created_by: Optional[str] = None
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionControl:
    """
    Control de versiones.
    
    Gestiona versiones de entidades del sistema.
    """
    
    def __init__(self, storage_path: str = "data/versions"):
        """
        Inicializar control de versiones.
        
        Args:
            storage_path: Ruta de almacenamiento
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, List[Version]] = {}  # entity_key -> versions
    
    def _get_entity_key(self, entity_type: str, entity_id: str) -> str:
        """Obtener clave de entidad."""
        return f"{entity_type}:{entity_id}"
    
    def create_version(
        self,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        created_by: Optional[str] = None,
        message: str = "",
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Version:
        """
        Crear versión.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            data: Datos de la versión
            created_by: ID del creador
            message: Mensaje de commit
            parent_version: Versión padre
            metadata: Metadata adicional
            
        Returns:
            Versión creada
        """
        entity_key = self._get_entity_key(entity_type, entity_id)
        
        # Generar ID de versión
        data_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:8]
        version_id = f"{entity_id}_{data_hash}"
        
        version = Version(
            version_id=version_id,
            entity_type=entity_type,
            entity_id=entity_id,
            data=data,
            created_by=created_by,
            message=message,
            parent_version=parent_version,
            metadata=metadata or {}
        )
        
        if entity_key not in self.versions:
            self.versions[entity_key] = []
        self.versions[entity_key].append(version)
        
        # Guardar en archivo
        self._save_version(version)
        
        logger.info(f"Created version {version_id} for {entity_type}:{entity_id}")
        
        return version
    
    def _save_version(self, version: Version) -> None:
        """Guardar versión en archivo."""
        try:
            version_file = self.storage_path / f"{version.version_id}.json"
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "version_id": version.version_id,
                    "entity_type": version.entity_type,
                    "entity_id": version.entity_id,
                    "data": version.data,
                    "created_by": version.created_by,
                    "message": version.message,
                    "timestamp": version.timestamp,
                    "parent_version": version.parent_version,
                    "metadata": version.metadata
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving version: {e}")
    
    def get_versions(
        self,
        entity_type: str,
        entity_id: str,
        limit: int = 100
    ) -> List[Version]:
        """
        Obtener versiones de entidad.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            limit: Límite de resultados
            
        Returns:
            Lista de versiones
        """
        entity_key = self._get_entity_key(entity_type, entity_id)
        versions = self.versions.get(entity_key, [])
        
        # Ordenar por timestamp (más reciente primero)
        versions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return versions[:limit]
    
    def get_version(
        self,
        entity_type: str,
        entity_id: str,
        version_id: str
    ) -> Optional[Version]:
        """
        Obtener versión específica.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            version_id: ID de versión
            
        Returns:
            Versión o None
        """
        versions = self.get_versions(entity_type, entity_id)
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    def get_latest_version(
        self,
        entity_type: str,
        entity_id: str
    ) -> Optional[Version]:
        """
        Obtener última versión.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            
        Returns:
            Última versión o None
        """
        versions = self.get_versions(entity_type, entity_id, limit=1)
        return versions[0] if versions else None
    
    def diff_versions(
        self,
        entity_type: str,
        entity_id: str,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            version_id1: ID de versión 1
            version_id2: ID de versión 2
            
        Returns:
            Diferencias entre versiones
        """
        version1 = self.get_version(entity_type, entity_id, version_id1)
        version2 = self.get_version(entity_type, entity_id, version_id2)
        
        if not version1 or not version2:
            return {"error": "One or both versions not found"}
        
        def get_diff(dict1: Dict, dict2: Dict, path: str = "") -> Dict[str, Any]:
            """Calcular diferencias recursivamente."""
            diff = {
                "added": {},
                "removed": {},
                "modified": {}
            }
            
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in dict1:
                    diff["added"][current_path] = dict2[key]
                elif key not in dict2:
                    diff["removed"][current_path] = dict1[key]
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        nested_diff = get_diff(dict1[key], dict2[key], current_path)
                        diff["added"].update(nested_diff["added"])
                        diff["removed"].update(nested_diff["removed"])
                        diff["modified"].update(nested_diff["modified"])
                    else:
                        diff["modified"][current_path] = {
                            "old": dict1[key],
                            "new": dict2[key]
                        }
            
            return diff
        
        return get_diff(version1.data, version2.data)
    
    def restore_version(
        self,
        entity_type: str,
        entity_id: str,
        version_id: str,
        created_by: Optional[str] = None
    ) -> Optional[Version]:
        """
        Restaurar versión.
        
        Args:
            entity_type: Tipo de entidad
            entity_id: ID de entidad
            version_id: ID de versión a restaurar
            created_by: ID del usuario que restaura
            
        Returns:
            Nueva versión creada o None
        """
        version = self.get_version(entity_type, entity_id, version_id)
        if not version:
            return None
        
        # Crear nueva versión con datos restaurados
        restored = self.create_version(
            entity_type=entity_type,
            entity_id=entity_id,
            data=version.data,
            created_by=created_by,
            message=f"Restored from version {version_id}",
            parent_version=version_id,
            metadata={"restored_from": version_id}
        )
        
        return restored


# Instancia global
_version_control: Optional[VersionControl] = None


def get_version_control(storage_path: str = "data/versions") -> VersionControl:
    """Obtener instancia global del control de versiones."""
    global _version_control
    if _version_control is None:
        _version_control = VersionControl(storage_path=storage_path)
    return _version_control






