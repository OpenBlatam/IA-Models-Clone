"""
Model Versioning
================

Sistema de versionado de modelos.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Versión de modelo."""
    version: str
    model_path: Path
    metadata: Dict[str, Any]
    created_at: str
    checksum: str
    tags: List[str] = None
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.tags is None:
            self.tags = []


class ModelVersioning:
    """
    Sistema de versionado de modelos.
    """
    
    def __init__(self, registry_dir: Path):
        """
        Inicializar sistema de versionado.
        
        Args:
            registry_dir: Directorio del registro de modelos
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.registry_dir / "versions.json"
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Cargar versiones desde disco."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    for version, version_data in data.items():
                        version_data['model_path'] = Path(version_data['model_path'])
                        self.versions[version] = ModelVersion(**version_data)
            except Exception as e:
                logger.warning(f"Error cargando versiones: {e}")
    
    def _save_versions(self) -> None:
        """Guardar versiones en disco."""
        try:
            data = {}
            for version, model_version in self.versions.items():
                version_dict = asdict(model_version)
                version_dict['model_path'] = str(version_dict['model_path'])
                data[version] = version_dict
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando versiones: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcular checksum de archivo."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(
        self,
        model_path: Path,
        metadata: Dict[str, Any],
        version: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Registrar nueva versión de modelo.
        
        Args:
            model_path: Ruta del modelo
            metadata: Metadatos del modelo
            version: Versión (opcional, auto-generada si None)
            tags: Tags (opcional)
            
        Returns:
            ModelVersion registrada
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Generar versión si no se proporciona
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular checksum
        checksum = self._calculate_checksum(model_path)
        
        # Copiar modelo al registro
        registry_model_path = self.registry_dir / f"model_{version}.pth"
        shutil.copy2(model_path, registry_model_path)
        
        # Crear versión
        model_version = ModelVersion(
            version=version,
            model_path=registry_model_path,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            tags=tags or []
        )
        
        # Guardar
        self.versions[version] = model_version
        self._save_versions()
        
        logger.info(f"Modelo registrado: versión {version}")
        
        return model_version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """
        Obtener versión específica.
        
        Args:
            version: Versión a obtener
            
        Returns:
            ModelVersion o None
        """
        return self.versions.get(version)
    
    def list_versions(self, tag: Optional[str] = None) -> List[ModelVersion]:
        """
        Listar versiones.
        
        Args:
            tag: Filtrar por tag (opcional)
            
        Returns:
            Lista de versiones
        """
        versions = list(self.versions.values())
        
        if tag:
            versions = [v for v in versions if tag in v.tags]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def get_latest(self, tag: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Obtener última versión.
        
        Args:
            tag: Filtrar por tag (opcional)
            
        Returns:
            Última ModelVersion o None
        """
        versions = self.list_versions(tag=tag)
        return versions[0] if versions else None
    
    def tag_version(self, version: str, tag: str) -> None:
        """
        Agregar tag a versión.
        
        Args:
            version: Versión
            tag: Tag a agregar
        """
        if version in self.versions:
            if tag not in self.versions[version].tags:
                self.versions[version].tags.append(tag)
                self._save_versions()
    
    def delete_version(self, version: str) -> bool:
        """
        Eliminar versión.
        
        Args:
            version: Versión a eliminar
            
        Returns:
            True si se eliminó, False si no existe
        """
        if version in self.versions:
            model_version = self.versions[version]
            if model_version.model_path.exists():
                model_version.model_path.unlink()
            del self.versions[version]
            self._save_versions()
            return True
        return False


# Instancia global
_global_versioning: Optional[ModelVersioning] = None


def get_versioning(registry_dir: Optional[Path] = None) -> ModelVersioning:
    """
    Obtener instancia global del sistema de versionado.
    
    Args:
        registry_dir: Directorio del registro (opcional)
        
    Returns:
        Instancia del sistema de versionado
    """
    global _global_versioning
    
    if _global_versioning is None:
        if registry_dir is None:
            registry_dir = Path.cwd() / "model_registry"
        _global_versioning = ModelVersioning(registry_dir)
    
    return _global_versioning

