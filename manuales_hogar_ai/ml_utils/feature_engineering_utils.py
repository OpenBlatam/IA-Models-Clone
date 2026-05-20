"""
Feature Engineering Utils - Utilidades de Feature Engineering
==============================================================

Utilidades para feature engineering avanzado.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

# Intentar importar sklearn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    logger.warning("sklearn not available, some feature engineering features will be limited")


class FeatureScaler:
    """
    Escalador de features.
    """
    
    def __init__(self, method: str = "standard"):
        """
        Inicializar escalador.
        
        Args:
            method: Método ('standard', 'minmax', 'robust')
        """
        if not _has_sklearn:
            raise ImportError("sklearn is required for FeatureScaler")
        
        self.method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Ajustar escalador.
        
        Args:
            X: Datos
        """
        self.scaler.fit(X)
        self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transformar datos.
        
        Args:
            X: Datos
            
        Returns:
            Datos escalados
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajustar y transformar.
        
        Args:
            X: Datos
            
        Returns:
            Datos escalados
        """
        return self.scaler.fit_transform(X)


class FeatureSelector:
    """
    Selector de features.
    """
    
    def __init__(
        self,
        method: str = "kbest",
        k: int = 10
    ):
        """
        Inicializar selector.
        
        Args:
            method: Método ('kbest', 'pca')
            k: Número de features
        """
        if not _has_sklearn:
            raise ImportError("sklearn is required for FeatureSelector")
        
        self.method = method
        self.k = k
        
        if method == "kbest":
            self.selector = SelectKBest(f_classif, k=k)
        elif method == "pca":
            self.selector = PCA(n_components=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        """
        Ajustar selector.
        
        Args:
            X: Features
            y: Targets (opcional)
        """
        self.selector.fit(X, y)
        self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transformar datos.
        
        Args:
            X: Datos
            
        Returns:
            Features seleccionadas
        """
        if not self.fitted:
            raise ValueError("Selector not fitted")
        return self.selector.transform(X)
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Ajustar y transformar.
        
        Args:
            X: Features
            y: Targets (opcional)
            
        Returns:
            Features seleccionadas
        """
        return self.selector.fit_transform(X, y)


class FeatureTransformer:
    """
    Transformador de features.
    """
    
    def __init__(self):
        """Inicializar transformador."""
        self.transformations: List[Callable] = []
    
    def add_transformation(self, transform: Callable) -> 'FeatureTransformer':
        """
        Agregar transformación.
        
        Args:
            transform: Función de transformación
            
        Returns:
            Self para chaining
        """
        self.transformations.append(transform)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplicar transformaciones.
        
        Args:
            X: Datos
            
        Returns:
            Datos transformados
        """
        result = X
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Llamar transformador."""
        return self.transform(X)


class PolynomialFeatures:
    """
    Features polinomiales.
    """
    
    def __init__(self, degree: int = 2):
        """
        Inicializar features polinomiales.
        
        Args:
            degree: Grado del polinomio
        """
        self.degree = degree
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transformar a features polinomiales.
        
        Args:
            X: Datos
            
        Returns:
            Features polinomiales
        """
        from itertools import combinations_with_replacement
        
        n_samples, n_features = X.shape
        features = [X]
        
        for degree in range(2, self.degree + 1):
            for indices in combinations_with_replacement(range(n_features), degree):
                feature = np.prod(X[:, indices], axis=1, keepdims=True)
                features.append(feature)
        
        return np.hstack(features)


class InteractionFeatures:
    """
    Features de interacción.
    """
    
    def __init__(self, max_interactions: int = 2):
        """
        Inicializar features de interacción.
        
        Args:
            max_interactions: Máximo número de interacciones
        """
        self.max_interactions = max_interactions
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Crear features de interacción.
        
        Args:
            X: Datos
            
        Returns:
            Features de interacción
        """
        from itertools import combinations
        
        n_samples, n_features = X.shape
        features = [X]
        
        for r in range(2, min(self.max_interactions + 1, n_features + 1)):
            for indices in combinations(range(n_features), r):
                interaction = np.prod(X[:, indices], axis=1, keepdims=True)
                features.append(interaction)
        
        return np.hstack(features)


def create_feature_pipeline(
    scaler_method: str = "standard",
    selector_method: Optional[str] = None,
    k: int = 10
) -> Tuple[FeatureScaler, Optional[FeatureSelector]]:
    """
    Crear pipeline de features.
    
    Args:
        scaler_method: Método de escalado
        selector_method: Método de selección (opcional)
        k: Número de features
        
    Returns:
        Tupla (scaler, selector)
    """
    scaler = FeatureScaler(scaler_method)
    selector = FeatureSelector(selector_method, k) if selector_method else None
    
    return scaler, selector




