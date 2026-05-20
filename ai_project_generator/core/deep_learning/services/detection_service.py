"""
Detection Service - Servicio de detección
==========================================

Servicio especializado para detectar características de Deep Learning.
Encapsula toda la lógica de detección en un servicio independiente.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Dict, Any

from ..utils.detectors import (
    detect_framework,
    detect_model_type,
    detect_fine_tuning_technique,
    detect_deep_learning_features,
    DeepLearningFeatures,
    FrameworkType,
    ModelType,
    FineTuningTechnique
)

logger = logging.getLogger(__name__)


def _validate_keywords(keywords: Dict[str, Any]) -> None:
    """
    Valida keywords (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Raises:
        ValueError: Si keywords es inválido
        TypeError: Si no es un diccionario
    """
    if not isinstance(keywords, dict):
        raise TypeError("keywords must be a dictionary")


class DetectionService:
    """
    Servicio de detección.
    
    Encapsula toda la lógica de detección de características de Deep Learning.
    Proporciona una interfaz limpia y fácil de usar.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializar servicio de detección."""
        self.logger = logging.getLogger(f"{__name__}.DetectionService")
    
    def detect_all_features(self, keywords: Dict[str, Any]) -> DeepLearningFeatures:
        """
        Detectar todas las características de Deep Learning.
        
        Args:
            keywords: Keywords extraídos del proyecto
            
        Returns:
            Características detectadas
            
        Raises:
            ValueError: Si keywords es inválido
            TypeError: Si keywords no es un diccionario
        """
        _validate_keywords(keywords)
        
        try:
            features = detect_deep_learning_features(keywords)
            self.logger.info(f"Detected features: {features.to_dict()}")
            return features
        except Exception as e:
            self.logger.error(f"Error detecting features: {e}", exc_info=True)
            raise
    
    def detect_framework(self, keywords: Dict[str, Any]) -> str:
        """
        Detectar framework.
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            Framework detectado (valor del enum)
            
        Raises:
            ValueError: Si keywords es inválido
            TypeError: Si keywords no es un diccionario
        """
        _validate_keywords(keywords)
        
        try:
            framework = detect_framework(keywords)
            self.logger.debug(f"Detected framework: {framework.value}")
            return framework.value
        except Exception as e:
            self.logger.error(f"Error detecting framework: {e}", exc_info=True)
            return FrameworkType.PYTORCH.value
    
    def detect_model_type(self, keywords: Dict[str, Any]) -> str:
        """
        Detectar tipo de modelo.
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            Tipo de modelo detectado (valor del enum)
            
        Raises:
            ValueError: Si keywords es inválido
            TypeError: Si keywords no es un diccionario
        """
        _validate_keywords(keywords)
        
        try:
            model_type = detect_model_type(keywords)
            self.logger.debug(f"Detected model type: {model_type.value}")
            return model_type.value
        except Exception as e:
            self.logger.error(f"Error detecting model type: {e}", exc_info=True)
            return ModelType.BASE.value
    
    def detect_fine_tuning_technique(self, keywords: Dict[str, Any]) -> str:
        """
        Detectar técnica de fine-tuning.
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            Técnica de fine-tuning detectada (valor del enum)
            
        Raises:
            ValueError: Si keywords es inválido
            TypeError: Si keywords no es un diccionario
        """
        _validate_keywords(keywords)
        
        try:
            technique = detect_fine_tuning_technique(keywords)
            self.logger.debug(f"Detected fine-tuning technique: {technique.value}")
            return technique.value
        except Exception as e:
            self.logger.error(f"Error detecting fine-tuning technique: {e}", exc_info=True)
            return FineTuningTechnique.FULL_FINETUNING.value
