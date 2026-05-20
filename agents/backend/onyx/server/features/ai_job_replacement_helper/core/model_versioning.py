"""
Model Versioning Service - Versionado de modelos
=================================================

Sistema para versionar y gestionar múltiples versiones de modelos.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Versión de modelo"""
    version: str
    model_path: str
    created_at: datetime
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    checksum: Optional[str] = None


class ModelVersioningService:
    """Servicio de versionado de modelos"""
    
    def __init__(self, version_dir: str = "model_versions"):
        """Inicializar servicio"""
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.metadata_file = self.version_dir / "versions.json"
        self._load_versions()
        logger.info(f"ModelVersioningService initialized in {version_dir}")
    
    def _load_versions(self) -> None:
        """Cargar versiones desde disco"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for version_id, version_data in data.items():
                        version_data['created_at'] = datetime.fromisoformat(
                            version_data['created_at']
                        )
                        self.versions[version_id] = ModelVersion(**version_data)
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
    
    def _save_versions(self) -> None:
        """Guardar versiones a disco"""
        try:
            data = {}
            for version_id, version in self.versions.items():
                version_dict = {
                    "version": version.version,
                    "model_path": version.model_path,
                    "created_at": version.created_at.isoformat(),
                    "description": version.description,
                    "metrics": version.metrics,
                    "config": version.config,
                    "tags": version.tags,
                    "checksum": version.checksum,
                }
                data[version_id] = version_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions: {e}")
    
    def _calculate_checksum(self, filepath: str) -> str:
        """Calcular checksum de archivo"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_version(
        self,
        model_path: str,
        version: Optional[str] = None,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """Crear nueva versión de modelo"""
        if version is None:
            version = f"v{len(self.versions) + 1}"
        
        # Calcular checksum
        checksum = self._calculate_checksum(model_path)
        
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            created_at=datetime.now(),
            description=description,
            metrics=metrics or {},
            config=config or {},
            tags=tags or [],
            checksum=checksum,
        )
        
        self.versions[version] = model_version
        self._save_versions()
        
        logger.info(f"Model version {version} created")
        return model_version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Obtener versión específica"""
        return self.versions.get(version)
    
    def list_versions(
        self,
        tag: Optional[str] = None,
        sort_by: str = "created_at"
    ) -> List[ModelVersion]:
        """Listar versiones"""
        versions = list(self.versions.values())
        
        # Filtrar por tag
        if tag:
            versions = [v for v in versions if tag in v.tags]
        
        # Ordenar
        if sort_by == "created_at":
            versions.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "version":
            versions.sort(key=lambda x: x.version, reverse=True)
        
        return versions
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Comparar dos versiones"""
        v1 = self.versions.get(version1)
        v2 = self.versions.get(version2)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_diff": {},
            "same_checksum": v1.checksum == v2.checksum,
        }
        
        # Comparar métricas
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            val1 = v1.metrics.get(metric, 0)
            val2 = v2.metrics.get(metric, 0)
            comparison["metrics_diff"][metric] = val2 - val1
        
        return comparison
    
    def delete_version(self, version: str) -> bool:
        """Eliminar versión"""
        if version not in self.versions:
            return False
        
        del self.versions[version]
        self._save_versions()
        logger.info(f"Version {version} deleted")
        return True




