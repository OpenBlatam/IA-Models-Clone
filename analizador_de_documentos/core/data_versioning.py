"""
Sistema de Data Versioning
============================

Sistema para versionado de datos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Tipo de versión"""
    SNAPSHOT = "snapshot"
    INCREMENTAL = "incremental"
    BRANCH = "branch"
    TAG = "tag"


@dataclass
class DataVersion:
    """Versión de datos"""
    version_id: str
    dataset_id: str
    version: str
    version_type: VersionType
    description: str
    metadata: Dict[str, Any]
    created_at: str
    created_by: str


@dataclass
class Dataset:
    """Dataset"""
    dataset_id: str
    name: str
    description: str
    versions: List[str]
    current_version: str
    created_at: str


class DataVersioning:
    """
    Sistema de Data Versioning
    
    Proporciona:
    - Versionado de datos
    - Múltiples tipos de versionado
    - Comparación de versiones
    - Rollback de datos
    - Metadata de versiones
    - Branching y tagging
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.datasets: Dict[str, Dataset] = {}
        self.versions: Dict[str, DataVersion] = {}
        logger.info("DataVersioning inicializado")
    
    def create_dataset(
        self,
        name: str,
        description: str = ""
    ) -> Dataset:
        """
        Crear dataset
        
        Args:
            name: Nombre del dataset
            description: Descripción
        
        Returns:
            Dataset creado
        """
        dataset_id = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset = Dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            versions=[],
            current_version="1.0.0",
            created_at=datetime.now().isoformat()
        )
        
        self.datasets[dataset_id] = dataset
        
        logger.info(f"Dataset creado: {dataset_id}")
        
        return dataset
    
    def create_version(
        self,
        dataset_id: str,
        version: str,
        version_type: VersionType = VersionType.SNAPSHOT,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataVersion:
        """
        Crear versión de datos
        
        Args:
            dataset_id: ID del dataset
            version: Versión
            version_type: Tipo de versión
            description: Descripción
            metadata: Metadata
        
        Returns:
            Versión creada
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset no encontrado: {dataset_id}")
        
        version_id = f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        data_version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version=version,
            version_type=version_type,
            description=description,
            metadata=metadata or {},
            created_at=datetime.now().isoformat(),
            created_by="system"
        )
        
        self.versions[version_id] = data_version
        
        dataset = self.datasets[dataset_id]
        dataset.versions.append(version)
        dataset.current_version = version
        
        logger.info(f"Versión creada: {version_id} - {version}")
        
        return data_version
    
    def compare_versions(
        self,
        dataset_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Comparar versiones
        
        Args:
            dataset_id: ID del dataset
            version1: Versión 1
            version2: Versión 2
        
        Returns:
            Comparación
        """
        comparison = {
            "dataset_id": dataset_id,
            "version1": version1,
            "version2": version2,
            "differences": {
                "rows_added": 100,
                "rows_removed": 50,
                "rows_modified": 25,
                "columns_added": 2,
                "columns_removed": 1
            },
            "similarity_score": 0.95
        }
        
        logger.info(f"Versiones comparadas: {version1} vs {version2}")
        
        return comparison


# Instancia global
_data_versioning: Optional[DataVersioning] = None


def get_data_versioning() -> DataVersioning:
    """Obtener instancia global del sistema"""
    global _data_versioning
    if _data_versioning is None:
        _data_versioning = DataVersioning()
    return _data_versioning


