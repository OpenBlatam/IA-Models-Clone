"""
Model Versioning System - Sistema de versionado de modelos
============================================================
"""

import logging
import os
import json
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Versión de modelo"""
    version: str
    model_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    is_active: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "is_active": self.is_active,
            "tags": self.tags
        }


class ModelVersioningSystem:
    """Sistema de versionado de modelos"""
    
    def __init__(self, version_dir: str = "./model_versions"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.load_versions()
    
    def create_version(
        self,
        model_path: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """Crea una nueva versión del modelo"""
        if version is None:
            version = self._generate_version()
        
        # Calcular checksum
        checksum = self._calculate_checksum(model_path)
        
        # Copiar modelo a directorio de versiones
        version_path = self.version_dir / f"model_v{version}.pt"
        shutil.copy(model_path, version_path)
        
        # Crear versión
        model_version = ModelVersion(
            version=version,
            model_path=str(version_path),
            metadata=metadata or {},
            checksum=checksum,
            tags=tags or []
        )
        
        self.versions[version] = model_version
        self.save_versions()
        
        logger.info(f"Versión {version} creada: {version_path}")
        return model_version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Obtiene una versión específica"""
        return self.versions.get(version)
    
    def list_versions(
        self,
        tags: Optional[List[str]] = None,
        active_only: bool = False
    ) -> List[ModelVersion]:
        """Lista versiones"""
        versions = list(self.versions.values())
        
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        if active_only:
            versions = [v for v in versions if v.is_active]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def activate_version(self, version: str) -> bool:
        """Activa una versión"""
        if version not in self.versions:
            return False
        
        # Desactivar todas las versiones
        for v in self.versions.values():
            v.is_active = False
        
        # Activar versión solicitada
        self.versions[version].is_active = True
        self.save_versions()
        
        logger.info(f"Versión {version} activada")
        return True
    
    def delete_version(self, version: str) -> bool:
        """Elimina una versión"""
        if version not in self.versions:
            return False
        
        model_version = self.versions[version]
        
        # Eliminar archivo
        if os.path.exists(model_version.model_path):
            os.remove(model_version.model_path)
        
        # Eliminar de registro
        del self.versions[version]
        self.save_versions()
        
        logger.info(f"Versión {version} eliminada")
        return True
    
    def _generate_version(self) -> str:
        """Genera número de versión"""
        if not self.versions:
            return "1.0.0"
        
        # Obtener última versión
        versions = sorted(self.versions.keys(), key=lambda v: [int(x) for x in v.split('.')])
        last_version = versions[-1]
        parts = last_version.split('.')
        
        # Incrementar patch
        parts[2] = str(int(parts[2]) + 1)
        return '.'.join(parts)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcula checksum de archivo"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def verify_version(self, version: str) -> bool:
        """Verifica integridad de versión"""
        if version not in self.versions:
            return False
        
        model_version = self.versions[version]
        
        if not os.path.exists(model_version.model_path):
            return False
        
        current_checksum = self._calculate_checksum(model_version.model_path)
        return current_checksum == model_version.checksum
    
    def save_versions(self):
        """Guarda registro de versiones"""
        versions_file = self.version_dir / "versions.json"
        versions_data = {
            version: version_obj.to_dict()
            for version, version_obj in self.versions.items()
        }
        
        with open(versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
    
    def load_versions(self):
        """Carga registro de versiones"""
        versions_file = self.version_dir / "versions.json"
        
        if not versions_file.exists():
            return
        
        with open(versions_file, 'r') as f:
            versions_data = json.load(f)
        
        for version, data in versions_data.items():
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            self.versions[version] = ModelVersion(**data)




