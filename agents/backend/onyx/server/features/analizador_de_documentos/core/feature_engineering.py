"""
Sistema de Feature Engineering Automático
===========================================

Sistema para generación automática de características.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Tipo de característica"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TEMPORAL = "temporal"
    EMBEDDING = "embedding"


@dataclass
class Feature:
    """Característica generada"""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    importance: float
    created_at: str


class AutomatedFeatureEngineering:
    """
    Sistema de Feature Engineering Automático
    
    Proporciona:
    - Generación automática de características
    - Transformaciones inteligentes
    - Selección de características
    - Feature importance
    - Interacciones entre características
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.features: Dict[str, Feature] = {}
        self.transformations: List[Dict[str, Any]] = []
        logger.info("AutomatedFeatureEngineering inicializado")
    
    def generate_features(
        self,
        data: List[Dict[str, Any]],
        target_column: Optional[str] = None
    ) -> List[Feature]:
        """
        Generar características automáticamente
        
        Args:
            data: Datos de entrada
            target_column: Columna objetivo (opcional)
        
        Returns:
            Lista de características generadas
        """
        features = []
        
        if not data:
            return features
        
        # Analizar primera muestra para inferir tipos
        sample = data[0]
        
        for key, value in sample.items():
            if key == target_column:
                continue
            
            feature_type = self._infer_feature_type(value)
            feature_id = f"feat_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            feature = Feature(
                feature_id=feature_id,
                feature_name=key,
                feature_type=feature_type,
                importance=0.5,  # Se calculará después
                created_at=datetime.now().isoformat()
            )
            
            features.append(feature)
            
            # Generar características derivadas
            if feature_type == FeatureType.NUMERICAL:
                # Características derivadas: log, sqrt, square
                derived_features = [
                    Feature(
                        feature_id=f"{feature_id}_log",
                        feature_name=f"{key}_log",
                        feature_type=FeatureType.NUMERICAL,
                        importance=0.3,
                        created_at=datetime.now().isoformat()
                    ),
                    Feature(
                        feature_id=f"{feature_id}_sqrt",
                        feature_name=f"{key}_sqrt",
                        feature_type=FeatureType.NUMERICAL,
                        importance=0.3,
                        created_at=datetime.now().isoformat()
                    )
                ]
                features.extend(derived_features)
        
        self.features.update({f.feature_id: f for f in features})
        
        logger.info(f"Características generadas: {len(features)}")
        
        return features
    
    def _infer_feature_type(self, value: Any) -> FeatureType:
        """Inferir tipo de característica"""
        if isinstance(value, (int, float)):
            return FeatureType.NUMERICAL
        elif isinstance(value, str):
            if len(value) > 100:
                return FeatureType.TEXT
            else:
                return FeatureType.CATEGORICAL
        else:
            return FeatureType.CATEGORICAL
    
    def select_features(
        self,
        features: List[Feature],
        method: str = "importance",
        top_k: int = 10
    ) -> List[Feature]:
        """
        Seleccionar características más importantes
        
        Args:
            features: Lista de características
            method: Método de selección
            top_k: Número de características a seleccionar
        
        Returns:
            Características seleccionadas
        """
        # Ordenar por importancia
        sorted_features = sorted(
            features,
            key=lambda x: x.importance,
            reverse=True
        )
        
        selected = sorted_features[:top_k]
        
        logger.info(f"Características seleccionadas: {len(selected)} de {len(features)}")
        
        return selected
    
    def create_interactions(
        self,
        features: List[Feature],
        max_interactions: int = 5
    ) -> List[Feature]:
        """
        Crear interacciones entre características
        
        Args:
            features: Lista de características
            max_interactions: Número máximo de interacciones
        
        Returns:
            Características de interacción
        """
        interactions = []
        
        # Crear interacciones entre características numéricas
        numerical_features = [f for f in features if f.feature_type == FeatureType.NUMERICAL]
        
        for i, feat1 in enumerate(numerical_features[:max_interactions]):
            for feat2 in numerical_features[i+1:max_interactions]:
                interaction = Feature(
                    feature_id=f"interaction_{feat1.feature_id}_{feat2.feature_id}",
                    feature_name=f"{feat1.feature_name}_x_{feat2.feature_name}",
                    feature_type=FeatureType.NUMERICAL,
                    importance=0.4,
                    created_at=datetime.now().isoformat()
                )
                interactions.append(interaction)
        
        logger.info(f"Interacciones creadas: {len(interactions)}")
        
        return interactions


# Instancia global
_feature_engineering: Optional[AutomatedFeatureEngineering] = None


def get_feature_engineering() -> AutomatedFeatureEngineering:
    """Obtener instancia global del sistema"""
    global _feature_engineering
    if _feature_engineering is None:
        _feature_engineering = AutomatedFeatureEngineering()
    return _feature_engineering


