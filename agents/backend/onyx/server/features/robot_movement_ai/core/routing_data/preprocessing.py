"""
Route Preprocessing
==================

Módulos para preprocesamiento de datos de enrutamiento.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    """Configuración de preprocesamiento."""
    scaler_type: str = "standard"  # standard, minmax, robust
    normalize_features: bool = True
    normalize_targets: bool = True
    feature_dim: int = 20
    target_dim: int = 4


class RoutePreprocessor:
    """
    Preprocesador para datos de enrutamiento.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializar preprocesador.
        
        Args:
            config: Configuración (opcional)
        """
        self.config = config or PreprocessingConfig()
        
        # Scalers
        scaler_classes = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler
        }
        
        scaler_class = scaler_classes.get(self.config.scaler_type, StandardScaler)
        
        if self.config.normalize_features:
            self.feature_scaler = scaler_class()
        else:
            self.feature_scaler = None
        
        if self.config.normalize_targets:
            self.target_scaler = scaler_class()
        else:
            self.target_scaler = None
        
        self._fitted = False
    
    def fit(self, features: List[np.ndarray], targets: List[np.ndarray]):
        """
        Ajustar scalers a los datos.
        
        Args:
            features: Lista de features
            targets: Lista de targets
        """
        if self.feature_scaler:
            features_array = np.array(features)
            self.feature_scaler.fit(features_array)
        
        if self.target_scaler:
            targets_array = np.array(targets)
            self.target_scaler.fit(targets_array)
        
        self._fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transformar features.
        
        Args:
            features: Features a transformar
            
        Returns:
            Features transformadas
        """
        if not self._fitted:
            return features
        
        if self.feature_scaler:
            # Asegurar que es 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = self.feature_scaler.transform(features)
            if features.shape[0] == 1:
                features = features.flatten()
        
        return features
    
    def transform_target(self, target: np.ndarray) -> np.ndarray:
        """
        Transformar target.
        
        Args:
            target: Target a transformar
            
        Returns:
            Target transformado
        """
        if not self._fitted:
            return target
        
        if self.target_scaler:
            # Asegurar que es 2D
            if target.ndim == 1:
                target = target.reshape(1, -1)
            target = self.target_scaler.transform(target)
            if target.shape[0] == 1:
                target = target.flatten()
        
        return target
    
    def inverse_transform_target(self, target: np.ndarray) -> np.ndarray:
        """
        Transformación inversa de target.
        
        Args:
            target: Target transformado
            
        Returns:
            Target original
        """
        if not self._fitted or not self.target_scaler:
            return target
        
        if target.ndim == 1:
            target = target.reshape(1, -1)
        target = self.target_scaler.inverse_transform(target)
        if target.shape[0] == 1:
            target = target.flatten()
        
        return target


class FeatureExtractor:
    """
    Extractor de features de rutas.
    """
    
    @staticmethod
    def extract_route_features(
        route_data: Dict[str, Any],
        max_features: int = 20
    ) -> np.ndarray:
        """
        Extraer features de una ruta.
        
        Args:
            route_data: Datos de la ruta
            max_features: Número máximo de features
            
        Returns:
            Array de features
        """
        features = []
        
        # Features básicas
        features.extend([
            route_data.get("distance", 0.0),
            route_data.get("time", 0.0),
            route_data.get("cost", 0.0),
            route_data.get("capacity", 1.0),
            route_data.get("current_load", 0.0),
        ])
        
        # Features de nodos
        node_features = route_data.get("node_features", [])
        features.extend(node_features[:10])
        
        # Features de aristas
        edge_features = route_data.get("edge_features", [])
        features.extend(edge_features[:5])
        
        # Padding/truncate
        while len(features) < max_features:
            features.append(0.0)
        
        return np.array(features[:max_features], dtype=np.float32)
    
    @staticmethod
    def extract_targets(
        route_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extraer targets de una ruta.
        
        Args:
            route_data: Datos de la ruta
            
        Returns:
            Array de targets
        """
        return np.array([
            route_data.get("predicted_time", 0.0),
            route_data.get("predicted_cost", 0.0),
            route_data.get("predicted_load", 0.0),
            route_data.get("success_probability", 0.0)
        ], dtype=np.float32)


