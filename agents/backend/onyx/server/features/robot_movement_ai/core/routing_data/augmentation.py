"""
Route Data Augmentation
=======================

Técnicas de data augmentation para datos de enrutamiento.
"""

import numpy as np
from typing import List, Dict, Any
import random


class RouteAugmentation:
    """
    Técnicas de data augmentation para rutas.
    """
    
    @staticmethod
    def add_noise(features: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Agregar ruido gaussiano a features.
        
        Args:
            features: Features originales
            noise_level: Nivel de ruido
            
        Returns:
            Features con ruido
        """
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    @staticmethod
    def scale_features(features: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
        """
        Escalar features aleatoriamente.
        
        Args:
            features: Features originales
            scale_range: Rango de escalado (min, max)
            
        Returns:
            Features escaladas
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return features * scale
    
    @staticmethod
    def augment_batch(features: List[np.ndarray], targets: List[np.ndarray], 
                     augmentation_prob: float = 0.5) -> tuple:
        """
        Aumentar batch de datos.
        
        Args:
            features: Lista de features
            targets: Lista de targets
            augmentation_prob: Probabilidad de aplicar augmentation
            
        Returns:
            (features_augmented, targets)
        """
        augmented_features = []
        
        for feat in features:
            if random.random() < augmentation_prob:
                # Aplicar augmentation aleatoria
                if random.random() < 0.5:
                    feat = RouteAugmentation.add_noise(feat)
                else:
                    feat = RouteAugmentation.scale_features(feat)
            
            augmented_features.append(feat)
        
        return augmented_features, targets


