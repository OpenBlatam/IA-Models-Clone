"""
Model Versioning - Modular Version Management
=============================================

Gestión modular de versiones de modelos.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Versión de modelo."""
    version: str
    model_path: str
    config_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    created_at: str = ""
    description: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción."""
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class VersionManager:
    """
    Gestor de versiones de modelos.
    
    Maneja versionado semántico y tracking
    de diferentes versiones de modelos.
    """
    
    def __init__(self, registry_path: str = "model_registry.json"):
        """
        Inicializar gestor de versiones.
        
        Args:
            registry_path: Ruta al registro de versiones
        """
        self.registry_path = Path(registry_path)
        self.versions: Dict[str, ModelVersion] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Cargar registro de versiones."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.versions = {
                        version: ModelVersion(**info)
                        for version, info in data.items()
                    }
                logger.info(f"Loaded {len(self.versions)} model versions")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.versions = {}
        else:
            self.versions = {}
    
    def _save_registry(self):
        """Guardar registro de versiones."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            version: asdict(model_version)
            for version, model_version in self.versions.items()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Registry saved to {self.registry_path}")
    
    def register_version(
        self,
        version: str,
        model_path: str,
        config_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Registrar nueva versión.
        
        Args:
            version: Versión (ej: "1.0.0")
            model_path: Ruta al modelo
            config_path: Ruta a configuración (opcional)
            metrics: Métricas (opcional)
            description: Descripción
            tags: Tags (opcional)
            metadata: Metadata adicional (opcional)
            
        Returns:
            Versión registrada
        """
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            config_path=config_path,
            metrics=metrics,
            description=description,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.versions[version] = model_version
        self._save_registry()
        
        logger.info(f"Registered model version: {version}")
        return model_version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """
        Obtener versión específica.
        
        Args:
            version: Versión
            
        Returns:
            Versión o None
        """
        return self.versions.get(version)
    
    def list_versions(self) -> List[str]:
        """Listar todas las versiones."""
        return sorted(self.versions.keys(), reverse=True)
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """Obtener última versión."""
        versions = self.list_versions()
        return self.versions.get(versions[0]) if versions else None
    
    def get_best_version(self, metric: str = 'val_loss') -> Optional[ModelVersion]:
        """
        Obtener mejor versión según métrica.
        
        Args:
            metric: Nombre de la métrica
            
        Returns:
            Mejor versión o None
        """
        best_version = None
        best_value = None
        
        for version, model_version in self.versions.items():
            if model_version.metrics and metric in model_version.metrics:
                value = model_version.metrics[metric]
                if best_value is None or value < best_value:  # Asumir menor es mejor
                    best_value = value
                    best_version = model_version
        
        return best_version
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
        metric: str = 'val_loss'
    ) -> Dict[str, Any]:
        """
        Comparar dos versiones.
        
        Args:
            version1: Primera versión
            version2: Segunda versión
            metric: Métrica para comparar
            
        Returns:
            Resultados de comparación
        """
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            return {'error': 'One or both versions not found'}
        
        result = {
            'version1': version1,
            'version2': version2,
            'metric': metric
        }
        
        if v1.metrics and v2.metrics:
            val1 = v1.metrics.get(metric)
            val2 = v2.metrics.get(metric)
            
            if val1 is not None and val2 is not None:
                result['value1'] = val1
                result['value2'] = val2
                result['difference'] = val2 - val1
                result['improvement'] = val1 > val2  # Asumir menor es mejor
        
        return result








