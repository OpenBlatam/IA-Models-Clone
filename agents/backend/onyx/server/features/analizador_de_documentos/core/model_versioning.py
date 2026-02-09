"""
Sistema de Versionado de Modelos
=================================

Sistema para gestionar versiones de modelos entrenados.
"""

import logging
import json
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Estados de modelo"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Versión de modelo"""
    version: str
    model_name: str
    model_path: str
    status: ModelStatus
    created_at: str
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    training_config: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelVersionManager:
    """
    Gestor de versiones de modelos
    
    Proporciona:
    - Versionado semántico
    - Gestión de estados
    - Metadatos y tags
    - Métricas de rendimiento
    - Rollback a versiones anteriores
    """
    
    def __init__(self, models_dir: str = "models"):
        """Inicializar gestor"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.versions_file = self.models_dir / "versions.json"
        self.versions: Dict[str, List[ModelVersion]] = {}
        self._load_versions()
        logger.info("ModelVersionManager inicializado")
    
    def _load_versions(self):
        """Cargar versiones desde disco"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_name, versions in data.items():
                        self.versions[model_name] = [
                            ModelVersion(
                                version=v['version'],
                                model_name=v['model_name'],
                                model_path=v['model_path'],
                                status=ModelStatus(v['status']),
                                created_at=v['created_at'],
                                metadata=v.get('metadata', {}),
                                performance_metrics=v.get('performance_metrics'),
                                training_config=v.get('training_config'),
                                tags=v.get('tags', [])
                            )
                            for v in versions
                        ]
            except Exception as e:
                logger.error(f"Error cargando versiones: {e}")
                self.versions = {}
    
    def _save_versions(self):
        """Guardar versiones en disco"""
        try:
            data = {}
            for model_name, versions in self.versions.items():
                data[model_name] = [asdict(v) for v in versions]
            
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando versiones: {e}")
    
    def register_version(
        self,
        model_name: str,
        model_path: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Registrar nueva versión
        
        Args:
            model_name: Nombre del modelo
            model_path: Ruta al modelo
            version: Versión (auto-incrementa si None)
            metadata: Metadatos adicionales
            tags: Tags para categorización
        
        Returns:
            ModelVersion creada
        """
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        # Generar versión si no se proporciona
        if version is None:
            existing_versions = [v.version for v in self.versions[model_name]]
            version = self._generate_version(existing_versions)
        
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            model_path=model_path,
            status=ModelStatus.READY,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        self.versions[model_name].append(model_version)
        self._save_versions()
        
        logger.info(f"Versión {version} registrada para modelo {model_name}")
        
        return model_version
    
    def _generate_version(self, existing_versions: List[str]) -> str:
        """Generar nueva versión semántica"""
        if not existing_versions:
            return "1.0.0"
        
        # Extraer números de versión
        versions = []
        for v in existing_versions:
            try:
                parts = v.split('.')
                if len(parts) == 3:
                    versions.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except:
                continue
        
        if not versions:
            return "1.0.0"
        
        # Encontrar última versión
        latest = max(versions)
        # Incrementar patch
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
    
    def get_version(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Obtener versión específica"""
        if model_name not in self.versions:
            return None
        
        if version:
            for v in self.versions[model_name]:
                if v.version == version:
                    return v
            return None
        else:
            # Devolver última versión
            versions = self.versions[model_name]
            if versions:
                return max(versions, key=lambda v: v.created_at)
            return None
    
    def list_versions(self, model_name: Optional[str] = None) -> Dict[str, List[ModelVersion]]:
        """Listar versiones"""
        if model_name:
            return {model_name: self.versions.get(model_name, [])}
        return self.versions.copy()
    
    def update_version_status(
        self,
        model_name: str,
        version: str,
        status: ModelStatus
    ) -> bool:
        """Actualizar estado de versión"""
        version_obj = self.get_version(model_name, version)
        if version_obj:
            version_obj.status = status
            self._save_versions()
            return True
        return False
    
    def deploy_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Desplegar versión (marcar como deployed)"""
        # Desplegar otras versiones
        if model_name in self.versions:
            for v in self.versions[model_name]:
                if v.status == ModelStatus.DEPLOYED:
                    v.status = ModelStatus.READY
        
        return self.update_version_status(model_name, version, ModelStatus.DEPLOYED)
    
    def get_deployed_version(self, model_name: str) -> Optional[ModelVersion]:
        """Obtener versión desplegada"""
        if model_name not in self.versions:
            return None
        
        for v in self.versions[model_name]:
            if v.status == ModelStatus.DEPLOYED:
                return v
        
        return None
    
    def archive_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Archivar versión"""
        return self.update_version_status(model_name, version, ModelStatus.ARCHIVED)
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Comparar dos versiones"""
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)
        
        if not v1 or not v2:
            return {"error": "Versión no encontrada"}
        
        comparison = {
            "version1": asdict(v1),
            "version2": asdict(v2),
            "differences": {}
        }
        
        # Comparar métricas si existen
        if v1.performance_metrics and v2.performance_metrics:
            comparison["differences"]["metrics"] = {}
            for metric in set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys()):
                val1 = v1.performance_metrics.get(metric, 0)
                val2 = v2.performance_metrics.get(metric, 0)
                if val1 != val2:
                    comparison["differences"]["metrics"][metric] = {
                        "v1": val1,
                        "v2": val2,
                        "diff": val2 - val1
                    }
        
        return comparison


# Instancia global
_model_version_manager: Optional[ModelVersionManager] = None


def get_model_version_manager(models_dir: str = "models") -> ModelVersionManager:
    """Obtener instancia global del gestor"""
    global _model_version_manager
    if _model_version_manager is None:
        _model_version_manager = ModelVersionManager(models_dir)
    return _model_version_manager

