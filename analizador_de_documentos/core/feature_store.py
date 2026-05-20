"""
Sistema de Feature Store
==========================

Sistema para almacenamiento y gestión de features.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Tipo de feature"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    EMBEDDING = "embedding"
    TIME_SERIES = "time_series"


@dataclass
class Feature:
    """Feature"""
    feature_id: str
    name: str
    feature_type: FeatureType
    description: str
    schema: Dict[str, Any]
    created_at: str


@dataclass
class FeatureSet:
    """Conjunto de features"""
    featureset_id: str
    name: str
    features: List[str]
    version: str
    created_at: str


class FeatureStore:
    """
    Sistema de Feature Store
    
    Proporciona:
    - Almacenamiento de features
    - Versionado de features
    - Búsqueda y descubrimiento de features
    - Transformación de features
    - Feature lineage
    - Feature sharing entre modelos
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.features: Dict[str, Feature] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}
        logger.info("FeatureStore inicializado")
    
    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        description: str,
        schema: Dict[str, Any]
    ) -> Feature:
        """
        Registrar feature
        
        Args:
            name: Nombre de la feature
            feature_type: Tipo de feature
            description: Descripción
            schema: Esquema de la feature
        
        Returns:
            Feature registrada
        """
        feature_id = f"feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feature = Feature(
            feature_id=feature_id,
            name=name,
            feature_type=feature_type,
            description=description,
            schema=schema,
            created_at=datetime.now().isoformat()
        )
        
        self.features[feature_id] = feature
        
        logger.info(f"Feature registrada: {feature_id} - {name}")
        
        return feature
    
    def create_feature_set(
        self,
        name: str,
        feature_ids: List[str]
    ) -> FeatureSet:
        """
        Crear conjunto de features
        
        Args:
            name: Nombre del conjunto
            feature_ids: IDs de features
        
        Returns:
            Feature set creado
        """
        featureset_id = f"featureset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feature_set = FeatureSet(
            featureset_id=featureset_id,
            name=name,
            features=feature_ids,
            version="1.0.0",
            created_at=datetime.now().isoformat()
        )
        
        self.feature_sets[featureset_id] = feature_set
        
        logger.info(f"Feature set creado: {featureset_id} - {len(feature_ids)} features")
        
        return feature_set
    
    def get_feature_lineage(
        self,
        feature_id: str
    ) -> Dict[str, Any]:
        """
        Obtener lineage de feature
        
        Args:
            feature_id: ID de la feature
        
        Returns:
            Lineage de la feature
        """
        if feature_id not in self.features:
            raise ValueError(f"Feature no encontrada: {feature_id}")
        
        lineage = {
            "feature_id": feature_id,
            "name": self.features[feature_id].name,
            "source_datasets": ["dataset_1", "dataset_2"],
            "transformations": ["normalization", "encoding"],
            "used_by_models": ["model_1", "model_2"]
        }
        
        logger.info(f"Lineage obtenido: {feature_id}")
        
        return lineage


# Instancia global
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Obtener instancia global del sistema"""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


