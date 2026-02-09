"""
Feature Store - Almacén de features para ML
============================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata de feature"""
    name: str
    feature_type: str  # "numerical", "categorical", "text", "embedding"
    shape: tuple
    dtype: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "feature_type": self.feature_type,
            "shape": self.shape,
            "dtype": self.dtype,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "tags": self.tags
        }


class FeatureStore:
    """Almacén de features"""
    
    def __init__(self, store_path: str = "./feature_store"):
        self.store_path = store_path
        self.features: Dict[str, Any] = {}
        self.metadata: Dict[str, FeatureMetadata] = {}
        os.makedirs(store_path, exist_ok=True)
    
    def store_feature(
        self,
        name: str,
        feature: Union[np.ndarray, pd.Series, List],
        feature_type: str = "numerical",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> FeatureMetadata:
        """Almacena una feature"""
        # Convertir a numpy array
        if isinstance(feature, pd.Series):
            feature_array = feature.values
        elif isinstance(feature, list):
            feature_array = np.array(feature)
        else:
            feature_array = feature
        
        # Guardar feature
        feature_path = os.path.join(self.store_path, f"{name}.npy")
        np.save(feature_path, feature_array)
        
        # Crear metadata
        metadata = FeatureMetadata(
            name=name,
            feature_type=feature_type,
            shape=feature_array.shape,
            dtype=str(feature_array.dtype),
            description=description,
            tags=tags or []
        )
        
        self.features[name] = feature_path
        self.metadata[name] = metadata
        
        logger.info(f"Feature {name} almacenada: shape={feature_array.shape}")
        return metadata
    
    def get_feature(self, name: str) -> Optional[np.ndarray]:
        """Obtiene una feature"""
        if name not in self.features:
            return None
        
        feature_path = self.features[name]
        if not os.path.exists(feature_path):
            logger.error(f"Archivo de feature no encontrado: {feature_path}")
            return None
        
        try:
            feature = np.load(feature_path)
            return feature
        except Exception as e:
            logger.error(f"Error cargando feature {name}: {e}")
            return None
    
    def get_feature_metadata(self, name: str) -> Optional[FeatureMetadata]:
        """Obtiene metadata de una feature"""
        return self.metadata.get(name)
    
    def list_features(
        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureMetadata]:
        """Lista features con filtros"""
        features = list(self.metadata.values())
        
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]
        
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]
        
        return features
    
    def delete_feature(self, name: str) -> bool:
        """Elimina una feature"""
        if name not in self.features:
            return False
        
        feature_path = self.features[name]
        if os.path.exists(feature_path):
            os.remove(feature_path)
        
        del self.features[name]
        if name in self.metadata:
            del self.metadata[name]
        
        logger.info(f"Feature {name} eliminada")
        return True
    
    def get_feature_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de una feature"""
        feature = self.get_feature(name)
        if feature is None:
            return None
        
        stats = {
            "shape": feature.shape,
            "dtype": str(feature.dtype),
            "size": feature.size,
            "memory_mb": feature.nbytes / 1024**2
        }
        
        if feature.dtype in [np.float32, np.float64, np.int32, np.int64]:
            stats.update({
                "mean": float(np.mean(feature)),
                "std": float(np.std(feature)),
                "min": float(np.min(feature)),
                "max": float(np.max(feature))
            })
        
        return stats




